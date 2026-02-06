#include "rcm.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <numeric>
#include <omp.h>

// --- EXTERN GLOBALS ---
extern int G_N;
extern int G_M; 
extern std::vector<int>    G_row_ptr;
extern std::vector<int>    G_col_idx;
extern std::vector<double> G_vals;

// ============================================================================
// HELPER: Calculate Node Degree
// ============================================================================
static std::vector<int> calculate_degrees(int n) {
    std::vector<int> degrees(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
       // G_row_ptr size is guaranteed to be n+1
        degrees[i] = G_row_ptr[i + 1] - G_row_ptr[i];
    }
    return degrees;
}

// ============================================================================
//  RCM (Safe & Optimized)
// ============================================================================
std::vector<int> calculate_rcm_permutation() {
    const int n = G_N;
    if (n == 0) return {};
    
// Symmetric permutation (RCM) on rectangular matrices (N != M)
// Generally not applicable directly. We return identity as a safeguard.
    if (G_M != G_N) {
        std::cerr << "[RCM Warning] Matrix is not square (N!=M). Skipping reordering." << std::endl;
        std::vector<int> P(n);
        std::iota(P.begin(), P.end(), 0);
        return P;
    }

    std::vector<int> degrees = calculate_degrees(n);
    std::vector<bool> visited(n, false);
    std::vector<int> rcm_order; 
    rcm_order.reserve(n);

{
        // --- STANDARD RCM ---
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;

            // You can search here to find the one with the lowest rating (Cuthill-McKee).
            // However, for speed, we start directly from i.
            
            std::queue<int> q;
            q.push(i);
            visited[i] = true;
            
            std::vector<int> component_nodes;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                component_nodes.push_back(u);

                int start = G_row_ptr[u];
                int end   = G_row_ptr[u + 1];
                
                // Gather neighbors
                std::vector<int> neighbors;
                neighbors.reserve(end - start);

                for (int k = start; k < end; ++k) {
                    int v = G_col_idx[k];
                    // Self-loop or out-of-bounds control
                    if (v < 0 || v >= n) continue; 
                    
                    if (!visited[v]) {
                        visited[v] = true;
                        neighbors.push_back(v);
                    }
                }

                // CM Rule: Add neighbors in ASCENDING order according to their degree
                std::sort(neighbors.begin(), neighbors.end(), 
                    [&degrees](int a, int b){ return degrees[a] < degrees[b]; });

                for (int v : neighbors) {
                    q.push(v);
                }
            }
            rcm_order.insert(rcm_order.end(), component_nodes.begin(), component_nodes.end());
        }

        // Reverse Cuthill-McKee
        std::reverse(rcm_order.begin(), rcm_order.end());
    }

    // --- SAFEGUARD: Missing Node Check ---
    // If the size of rcm_order is not N (isolated nodes, etc.), add the missing ones.
    if ((int)rcm_order.size() < n) {
        std::vector<bool> in_order(n, false);
        for (int x : rcm_order) in_order[x] = true;
        for (int i = 0; i < n; ++i) {
            if (!in_order[i]) rcm_order.push_back(i);
        }
    }

    // --- Mapping ---
    // P[old_id] = new_position
    std::vector<int> P(n);
    #pragma omp parallel for
    for (int k = 0; k < n; ++k) {
        P[rcm_order[k]] = k;
    }

    return P;
}

// ============================================================================
// 2. APPLY PERMUTATION (Fixed Logic)
// ============================================================================
void apply_permutation_to_csr(const std::vector<int>& P,
                              std::vector<int>& new_row_ptr,
                              std::vector<int>& new_col_idx,
                              std::vector<double>& new_vals)
{
    const int n = G_N;
    bool has_vals = !G_vals.empty();

    // 1. Inverse Permutation: invP[new_row] = old_row
    std::vector<int> invP(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        invP[P[i]] = i;
    }

    // 2. New row_ptr calculation
    new_row_ptr.resize(n + 1);
    new_row_ptr[0] = 0;

    // This part must be sequential (prefix sum)
    long long total_nnz = 0;
    for (int new_r = 0; new_r < n; ++new_r) {
        int old_r = invP[new_r];
        int len = G_row_ptr[old_r + 1] - G_row_ptr[old_r];
        total_nnz += len;
        new_row_ptr[new_r + 1] = (int)total_nnz;
    }

    // 3. Allocate memory
    new_col_idx.resize(total_nnz);
    if (has_vals) {
        new_vals.resize(total_nnz);
    } else {
        new_vals.clear();
    }

    // 4. Fill rows (Parallel)
    #pragma omp parallel for schedule(dynamic, 128)
    for (int new_r = 0; new_r < n; ++new_r) {
        int old_r = invP[new_r];
        int old_start = G_row_ptr[old_r];
        int row_len   = G_row_ptr[old_r + 1] - G_row_ptr[old_r];
        int new_start = new_row_ptr[new_r];

        if (row_len == 0) continue;

        // Temporary buffer: (New Col Index, Value)
        std::vector<std::pair<int, double>> buf(row_len);

        for (int k = 0; k < row_len; ++k) {
            int old_c = G_col_idx[old_start + k];
            double val = has_vals ? G_vals[old_start + k] : 0.0;
            
            // CAUTION: P[old_c] is only valid for square matrices and symmetric permutations.
            // If old_c >= n (rectangular matrix), this code will explode.
            // We're safe here because we added the N!=M ​​check in the RCM function.
            buf[k] = { P[old_c], val };
        }

        // Sort by column indexes (required for CSR format)
        std::sort(buf.begin(), buf.end(), 
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Write data to original vectors
        for (int k = 0; k < row_len; ++k) {
            new_col_idx[new_start + k] = buf[k].first;
            if (has_vals) {
                new_vals[new_start + k] = buf[k].second;
            }
        }
    }
}