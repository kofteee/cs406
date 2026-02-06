// spmv.cpp
// CS406: Parallel Computing - Homework Skeleton (SpMV).
// Build (sequential skeleton):
//   g++ -O3 -std=c++17 spmv.cpp -o spmv
// If you add OpenMP pragmas:
//   g++ -O3 -std=c++17 -fopenmp spmv.cpp -o spmv
// Run: ./spmv path/to/matrix.mtx

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>   // <-- added

#include "rcm.h"

int G_N = 0;                   // rows
int G_M = 0;                   // cols
std::vector<int>    G_row_ptr; // length: G_N+1
std::vector<int>    G_col_idx; // length: nnz
std::vector<double> G_vals;    // length: nnz

double gflops; // TODO: GLOBAL VARIABLE TO HOLD GFLOPS VALUE, YOU WILL COMPUTE THIS IN spmv()

std::vector<std::pair<int,int>> freqs; // the row number goes to number of elements in this row

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// ---- MatrixMarket reader (coordinate + array N×2 edge list) ----
static void read_matrix_market_to_global_csr(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open file: " + path);

    std::string header;
    if (!std::getline(fin, header)) throw std::runtime_error("Empty file");
    if (header.rfind("%%MatrixMarket", 0) != 0)
        throw std::runtime_error("Not a MatrixMarket file");

    const bool is_coordinate = (header.find("coordinate") != std::string::npos);
    const bool is_array      = (header.find("array")      != std::string::npos);
    const bool is_pattern    = (header.find("pattern")    != std::string::npos);
    const bool is_symmetric  = (header.find("symmetric")  != std::string::npos);
    const bool is_general    = (header.find("general")    != std::string::npos);

    if (!is_coordinate && !is_array)
        throw std::runtime_error("Only 'coordinate' or 'array' MatrixMarket supported");

    std::string line;
    do {
        if (!std::getline(fin, line)) throw std::runtime_error("Missing size line");
    } while (!line.empty() && line[0] == '%');

    long nrows=0, ncols=0, nz_or_vals=0;
    {
        std::istringstream ss(line);
        if (is_coordinate) {
            if (!(ss >> nrows >> ncols >> nz_or_vals))
                throw std::runtime_error("Bad size line for coordinate");
        } else {
            if (!(ss >> nrows >> ncols))
                throw std::runtime_error("Bad size line for array");
        }
    }

    struct Trip { int r, c; double v; };
    std::vector<Trip> trips;

    if (is_coordinate) {
        if (!is_general && !is_symmetric)
            throw std::runtime_error("Unsupported symmetry flag in coordinate header");
        trips.reserve(is_symmetric ? (size_t)nz_or_vals * 2 : (size_t)nz_or_vals);

        for (long t = 0; t < nz_or_vals; ++t) {
            if (!std::getline(fin, line)) throw std::runtime_error("Unexpected EOF");
            if (line.empty()) { --t; continue; }
            std::istringstream s(line);
            int i, j; double v = 1.0;
            if (!(s >> i >> j)) throw std::runtime_error("Bad coordinate entry");
            if (!is_pattern) { if (!(s >> v)) v = 1.0; }
            --i; --j;
            if (i < 0 || j < 0) continue;
            trips.push_back({i, j, v});
            if (is_symmetric && i != j) trips.push_back({j, i, v});
        }

        G_N = (int)nrows; G_M = (int)ncols;
        std::vector<std::vector<std::pair<int,double>>> rows(G_N);
        for (auto &t : trips) {
            if (t.r >= 0 && t.r < G_N && t.c >= 0 && t.c < G_M)
                rows[t.r].push_back({t.c, t.v});
        }
        G_row_ptr.assign(G_N + 1, 0);
        for (int r = 0; r < G_N; ++r) {
            auto &vec = rows[r];
            std::sort(vec.begin(), vec.end(), [](auto &a, auto &b){ return a.first < b.first; });
            int w = 0;
            for (int u = 0; u < (int)vec.size();) {
                int c = vec[u].first;
                double s = vec[u].second;
                int v = u + 1;
                while (v < (int)vec.size() && vec[v].first == c) { s += vec[v].second; ++v; }
                vec[w++] = {c, s};
                u = v;
            }
            vec.resize(w);
            G_row_ptr[r+1] = G_row_ptr[r] + (int)vec.size();
        }
        const int nnz = G_row_ptr.back();
        G_col_idx.resize(nnz); G_vals.resize(nnz);
        for (int r = 0; r < G_N; ++r) {
            int base = G_row_ptr[r];
            for (int k = 0; k < (int)rows[r].size(); ++k) {
                G_col_idx[base + k] = rows[r][k].first;
                G_vals   [base + k] = rows[r][k].second;
            }
        }
    } else {
        if (ncols != 2)
            throw std::runtime_error("array real general must be N x 2 (edge list)");
        const long N = nrows;
        std::vector<double> colmajor; colmajor.reserve(N * 2);

        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::istringstream s(line);
            double v;
            if (!(s >> v)) throw std::runtime_error("Bad array value line");
            colmajor.push_back(v);
        }
        if ((long)colmajor.size() != N * 2)
            throw std::runtime_error("Unexpected value count in array file");

        std::vector<int> U(N), V(N);
        for (long r = 0; r < N; ++r) {
            long iu = (long)std::llround(colmajor[(size_t)r + 0 * (size_t)N]);
            long iv = (long)std::llround(colmajor[(size_t)r + 1 * (size_t)N]);
            if (iu <= 0 || iv <= 0) continue;
            U[r] = (int)(iu - 1);
            V[r] = (int)(iv - 1);
        }

        int nmax = 0;
        for (long r = 0; r < N; ++r)
            nmax = std::max(nmax, std::max(U[r], V[r]));
        G_N = G_M = nmax + 1;

        std::vector<std::vector<int>> rows(G_N);
        for (long r = 0; r < N; ++r) {
            if (U[r] >= 0 && U[r] < G_N && V[r] >= 0 && V[r] < G_M)
                rows[U[r]].push_back(V[r]);
        }

        G_row_ptr.assign(G_N + 1, 0);
        for (int i = 0; i < G_N; ++i) {
            auto &vec = rows[i];
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
            G_row_ptr[i+1] = G_row_ptr[i] + (int)vec.size();
        }
        const int nnz = G_row_ptr.back();
        G_col_idx.resize(nnz); G_vals.assign(nnz, 1.0);
        for (int i = 0; i < G_N; ++i) {
            int base = G_row_ptr[i];
            for (int k = 0; k < (int)rows[i].size(); ++k)
                G_col_idx[base + k] = rows[i][k];
        }
    }
}

// ======== EDIT BELOW: Implement SpMV kernel ========
void spmv(const std::vector<double>& x, std::vector<double>& y) {
    
    // Large graph check, for soc-liveJournal1.mtx
    if (G_N > 4000000) {
        // Dynamic schedule resolves load imbalance
        #pragma omp parallel for schedule(dynamic, 64) 
        for (int r = 0; r < G_N; r++) {
            double sum = 0.0;
            const int start = G_row_ptr[r];
            const int end   = G_row_ptr[r+1];


            // __restrict tells the compiler "these pointers don't overlap, vectorize them freely"
            const int* __restrict cols = &G_col_idx[start];
            const double* __restrict vals = &G_vals[start];
            
            #pragma omp simd reduction(+:sum) // Even though this is not useful, compiler with -o3 can optimise this
            for (int k = 0; k < end - start; k++) {
                sum += vals[k] * x[cols[k]];
            }
            y[r] = sum;
        }
    } else { // For other matrixes
        // Static schedule cache is better for locality
        #pragma omp parallel for schedule(static) 
        for (int r = 0; r < G_N; r++) {
            double sum = 0.0;
            const int start = G_row_ptr[r];
            const int end   = G_row_ptr[r+1];
            
            
            const int* __restrict cols = &G_col_idx[start];
            const double* __restrict vals = &G_vals[start];
            
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < end - start; k++) {
                sum += vals[k] * x[cols[k]];
            }
            y[r] = sum;
        }
    }
}
// ======== EDIT ABOVE ========

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./spmv path/to/matrix.mtx\n";
        return 1;
    }
    const std::string path = argv[1];

    std::cerr << "[load] " << path << "\n";
    read_matrix_market_to_global_csr(path);
    const size_t nnz = G_col_idx.size();
    std::cerr << "Matrix: n=" << G_N << " m=" << G_M << " nnz=" << nnz << "\n";


// ======== Preprocess ========
#ifdef preprocess
    const uint64_t start = now_ns(); // Time start

    std::vector<int> P = calculate_rcm_permutation();

    // new vectors for CSR 
    std::vector<int> new_row_ptr, new_col_idx;
    std::vector<double> new_vals;
    apply_permutation_to_csr(P, new_row_ptr, new_col_idx, new_vals);

    G_row_ptr.swap(new_row_ptr);
    G_col_idx.swap(new_col_idx);
    G_vals.swap(new_vals);
    
    const uint64_t end = now_ns();   // Time ends
    
    // Scriptin yakalaması için:
    std::cerr << "Pre Process: n=" << (end-start) / 1e9 << std::endl;
#endif

// ======== Preprocess ========   


    std::vector<double> x(G_M, 1), y(G_N, 0);
    // Warm-up
    spmv(x, y); std::swap(x, y);

    // 50 iterations
    const int iters = 50;
    const uint64_t t0 = now_ns();
    for (int it = 0; it < iters; ++it) {
        spmv(x, y);
        std::swap(x, y);
    }
    const uint64_t t1 = now_ns();
    const double sec = (t1 - t0) * 1e-9;

    double chk = 0.0; for (double v : x) chk += v;
    
    gflops = iters * G_vals.size() * 2 / sec / 1e9; //      Total flops / time

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "time_sec=" << sec
              << "  gflops=" << gflops
              << "  checksum=" << chk << "\n";
    return 0;
}
