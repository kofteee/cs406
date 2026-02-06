// spmv_cuda.cu — CS406 HW2 CUDA Skeleton
//
// Build (example):
//   nvcc -O3 -std=c++17 spmv_cuda.cu rcm.cpp -o spmv_cuda
// Run:
//   ./spmv_cuda path/to/matrix.mtx
//
// You will implement:
//   - device_build_from_global_csr()
//   - spmv_cuda_kernel(...) (may be a controller or call other kernels)
//   - spmv(...): any launches/transfers you need
//
// We provide: CPU MatrixMarket → CSR loader into global CPU arrays.

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
#include <cmath>
#include <cuda_runtime.h>

#include "rcm.h"

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(err)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} } while(0)

// ---------------- CPU-side global CSR ----------------
int G_N = 0;                   // rows
int G_M = 0;                   // cols
std::vector<int>    G_row_ptr; // length: G_N+1
std::vector<int>    G_col_idx; // length: nnz
std::vector<double> G_vals;    // length: nnz
std::vector<double> x, y;      // I declare them globally
double sec;
double gflops = 0.0; // set by your implementation if you choose

// ---------------- Device pointers (you will populate) ----------------
static int    *d_row_ptr = nullptr;
static int    *d_col_idx = nullptr;
static double *d_vals    = nullptr;
static double *d_x       = nullptr;
static double *d_y = nullptr;

// ---------------- MatrixMarket → CSR loader (CPU) ----------------
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

// --------------- You will build device state here -------------------
static void device_build_from_global_csr() {
    int nnz = G_col_idx.size();

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (G_N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,    nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,       G_M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,       G_N * sizeof(double)));
    
    
    CUDA_CHECK(cudaMemcpy(d_row_ptr, G_row_ptr.data(), (G_N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, G_col_idx.data(), nnz * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,    G_vals.data(),    nnz * sizeof(double),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,       x.data(),         G_M * sizeof(double),    cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_y, 0, G_N * sizeof(double)));
}

// --------------- You will implement your CUDA kernel(s) -------------
template <int THREADS_PER_ROW>
__global__ void 
__launch_bounds__(128) // Decrease register pressure with bound
spmv_csr_vector_tuned_kernel(
    int num_rows,
    const double* __restrict__ vals,
    const int* __restrict__ col_idx,
    const int* __restrict__ row_ptr,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    // Block size 128 
    const int ROWS_PER_BLOCK = blockDim.x / THREADS_PER_ROW;
    
    int thread_in_block = threadIdx.x;
    int row_in_block    = thread_in_block / THREADS_PER_ROW;
    int lane_id         = thread_in_block % THREADS_PER_ROW;
    int row             = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    if (row < num_rows) {
        // Reading row pointers with __ldg 
        int start = __ldg(&row_ptr[row]);
        int end   = __ldg(&row_ptr[row + 1]);

        double sum = 0.0;

        // UNROLL: (Instruction Level Parallelism)
        #pragma unroll
        for (int k = start + lane_id; k < end; k += THREADS_PER_ROW) {
            // All readings with __ldg  (Texture Cache Path)
            int col = __ldg(&col_idx[k]);
            double val = __ldg(&vals[k]);
            
            // Fused Multiply-Add (FMA) 
            sum += val * __ldg(&x[col]);
        }

        // Warp Shuffle Reduction
        unsigned int mask = 0xffffffff;

        #pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        if (lane_id == 0) {
            // Overwrite
            y[row] = sum;
        }
    }
}

// --------------- Host wrapper that calls your kernel(s) -------------
// Includes event timing around YOUR launch(es).
static void spmv(const std::vector<double>& x_host, std::vector<double>& y_host) {
    // You will do any H2D/D2H copies you need.

    const size_t nnz = G_col_idx.size();
    double avg_nnz = (double)nnz / G_N;
    
    // Block size 128
    int threads_per_block = 128; 

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Purpose: increasing GPU Memory Clock to 3505/5005 MHz
    // Warm-up
    for(int k=0; k < 200; k++) {
        
         const int TPR = 8;
         int rows_per_block = threads_per_block / TPR;
         int blocks = (G_N + rows_per_block - 1) / rows_per_block;
         spmv_csr_vector_tuned_kernel<TPR><<<blocks, threads_per_block>>>(
                G_N, d_vals, d_col_idx, d_row_ptr, d_x, d_y);
    }
    int NUM_ITER = 50; 
    CUDA_CHECK(cudaEventRecord(start));

    // YOU WILL LAUNCH YOUR KERNEL(S) HERE.
    // Example: spmv_cuda_kernel<<<grid_dim, block_dim>>>(...);

    for(int i = 0; i < NUM_ITER; i++) {
        // TPR --> THREADS PER ROW
        
        // 1. Very Sparse -> TPR 2 (netherlands)
        if (avg_nnz <= 2.5) {
            const int TPR = 2;
            int rows_per_block = threads_per_block / TPR;
            int blocks = (G_N + rows_per_block - 1) / rows_per_block;
            spmv_csr_vector_tuned_kernel<TPR><<<blocks, threads_per_block>>>(
                G_N, d_vals, d_col_idx, d_row_ptr, d_x, d_y);
        }
        // 2. Sparse (e.g. Delaunay) -> TPR 4
        else if (avg_nnz <= 7.0) { 
            const int TPR = 4;
            int rows_per_block = threads_per_block / TPR;
            int blocks = (G_N + rows_per_block - 1) / rows_per_block;
            spmv_csr_vector_tuned_kernel<TPR><<<blocks, threads_per_block>>>(
                G_N, d_vals, d_col_idx, d_row_ptr, d_x, d_y);
        }
        // 3. Medium/High Density (LiveJournal SHOULD fall here) -> TPR 32
        // Previously, this was TPR=8 and threshold was avg <= 16.0.
        // LiveJournal (~14 nnz) was falling into that category and causing a bottleneck.
        // Now, we try everything greater than 7.0 directly with TPR 32.
        else { // soc-LiveJournal
            const int TPR = 32;
            int rows_per_block = threads_per_block / TPR;
            int blocks = (G_N + rows_per_block - 1) / rows_per_block;
            spmv_csr_vector_tuned_kernel<TPR><<<blocks, threads_per_block>>>(
                G_N, d_vals, d_col_idx, d_row_ptr, d_x, d_y);
        }
        
        std::swap(d_x, d_y);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    sec = milliseconds * 1e-3;

    if (sec > 0.0) {
        // One SpMV: ≈ 2*nnz FLOPs (mul+add)
        gflops = (2.0 * (double)nnz) * (double)NUM_ITER / (sec * 1e9);
    }
    // You will ensure y_host is filled if you want a non-zero checksum.
    // Copying from d_x because of the final swap operation
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_x, G_N * sizeof(double), cudaMemcpyDeviceToHost));
}
// --------------- Cleanup (we do this) -------------------------------
static void device_free_all() {
    if (d_row_ptr) {CUDA_CHECK(cudaFree(d_row_ptr)); d_row_ptr = nullptr;}
    if (d_col_idx) {CUDA_CHECK(cudaFree(d_col_idx)); d_col_idx = nullptr;}
    if (d_vals)    {CUDA_CHECK(cudaFree(d_vals));    d_vals    = nullptr;}
    if (d_x)       {CUDA_CHECK(cudaFree(d_x));       d_x       = nullptr;}
    if (d_y)       {CUDA_CHECK(cudaFree(d_y));       d_y       = nullptr;}
}

// ------------------------------- main -------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./spmv_cuda path/to/matrix.mtx\n";
        return 1;
    }
    const std::string path = argv[1];

    std::cerr << "[load] " << path << "\n";
    read_matrix_market_to_global_csr(path);
    const size_t nnz = G_col_idx.size();
    std::cerr << "Matrix: n=" << G_N << " m=" << G_M << " nnz=" << nnz << "\n";

    if (G_N == 0 || G_M == 0 || nnz == 0) {
        std::cerr << "Empty/invalid matrix.\n";
        return 1;
    }

///////////////////////// 
//////// RCM
/////////////////////////
    std::vector<int> P = calculate_rcm_permutation();

    std::vector<int> new_row_ptr, new_col_idx;
    std::vector<double> new_vals;
    apply_permutation_to_csr(P, new_row_ptr, new_col_idx, new_vals);

    G_row_ptr.swap(new_row_ptr);
    G_col_idx.swap(new_col_idx);
    G_vals.swap(new_vals);
///////////////////////// 
//////// RCM
/////////////////////////
        
    x.assign(G_M, 1.0); 
    y.assign(G_N, 0.0);

    std::vector<double> x(G_M, 1.0), y(G_N, 0.0);

    device_build_from_global_csr();
    spmv(x, y);

    double chk = 0.0; for (double v : y) chk += v;
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6) << "secs=" << sec
              << "  gflops=" << gflops
              << "  checksum=" << chk << "\n";

    device_free_all();
    return 0;
}
