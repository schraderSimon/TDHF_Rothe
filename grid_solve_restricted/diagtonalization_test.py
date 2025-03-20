import numpy as np

def full_diagonalization_smallest_eigenvalues(S, imin):
    """
    For each i >= imin, form the principal submatrix S_i (S with the i-th row and
    column removed) and compute its eigenvalues by full diagonalization.
    Return an array with the smallest eigenvalue for each S_i.
    """
    n = S.shape[0]
    smallest_eigs = []
    for i in range(imin, n):
        S_i = np.delete(np.delete(S, i, axis=0), i, axis=1)
        lam_i = np.linalg.eigvalsh(S_i)
        smallest_eigs.append(lam_i[0])
    return np.array(smallest_eigs)

# --- Example test code ---
if __name__ == '__main__':
    # Create a random Hermitian matrix S for testing.
    n = 10
    A = np.random.randn(n, n)
    S = A @ A.T  # S is now symmetric (real Hermitian)
    
    # Set the starting index for deletion
    imin = 5
    
    # Compute the smallest eigenvalues via the secular equation method.
    # Compute the smallest eigenvalues via full diagonalization for comparison.
    full_eigs = full_diagonalization_smallest_eigenvalues(S, imin)
    print("\nSmallest eigenvalues using full diagonalization (for i>=imin):")
    for idx, eig in enumerate(full_eigs, start=imin):
        print(f"i = {idx}: mu = {eig:.6e}")
