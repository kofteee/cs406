#ifndef RCM_ORDERING_H
#define RCM_ORDERING_H

#include <vector>

/**
 * Calculates RCM (Reverse Cuthill–McKee) permutation.
 * Reads the global vectors G_N, G_row_ptr, and G_col_idx (as externs) from spmv.cpp.
 * @return P where P[old_index] = new_index
 */
std::vector<int> calculate_rcm_permutation();

/**
 * Apply a row/column permutation to a CSR matrix.
 * Input CSR is read from globals G_N, G_row_ptr, G_col_idx, G_vals.
 * The function fills new_row_ptr/new_col_idx/new_vals with the permuted CSR
 * such that the new matrix corresponds to A_perm = P * A * P^T,
 * where P is the permutation matrix for mapping old -> new indices.
 *
 * Note: Caller should swap the returned vectors into the globals if they want
 * to replace the matrix in-place.
 */
void apply_permutation_to_csr(const std::vector<int>& P,
                              std::vector<int>& new_row_ptr,
                              std::vector<int>& new_col_idx,
                              std::vector<double>& new_vals);

#endif // RCM_ORDERING_H