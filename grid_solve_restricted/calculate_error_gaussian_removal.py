import numpy as np
import argparse
import sys
import os
from scipy.optimize import minimize
import gauss_Rothe as GR
from quadratures import gaussian_quadrature, trapezoidal_quadrature
def full_diagonalization_smallest_eigenvalues(S, imin):
    """
    For each i >= imin, form the principal submatrix S_i (S with the i-th row and
    column removed) and compute its eigenvalues by full diagonalization.
    Return an array with the smallest eigenvalue for each S_i.
    """
    n = S.shape[0]
    largest_elements=[]
    smallest_eigs = []
    for i in range(imin, n):
        S_i = np.delete(np.delete(S, i, axis=0), i, axis=1)
        largest_element=np.max(np.abs(S_i[imin:,imin:]-np.diag(np.diag(S_i[imin:,imin:]))))
        lam_i = np.linalg.eigvalsh(S_i)
        smallest_eigs.append(lam_i[0])
        largest_elements.append(largest_element)
    return np.array(smallest_eigs),np.array(largest_elements)


def load_input_file(input_file, start_time, norbs):
    """
    Loads the input file and extracts the state time, linear coefficients, and Gaussian nonlinear parameters.
    """
    data = np.load(input_file, allow_pickle=True)
    times = data["times"]
    # Select the state closest to start_time.
    index = np.abs(times - start_time).argmin()
    t_state = times[index]

    nbasis_arr = data["nbasis"]
    ngauss = int(nbasis_arr[index])

    # Retrieve the state from the "params" key.
    state = data["params"][index]
    total_len = len(state)
    ngauss_wrong = total_len // (norbs * 2 + 4)

    lincoeff_real = state[:ngauss * norbs]
    lincoeff_imag = state[ngauss_wrong * norbs:(ngauss + ngauss_wrong) * norbs]
    lincoeff_initial = (lincoeff_real + 1j * lincoeff_imag).reshape((ngauss, norbs))

    if ngauss_wrong - ngauss > 0:
        gaussian_params = state[ngauss_wrong * norbs * 2: -4 * (ngauss_wrong - ngauss)]
    else:
        gaussian_params = state[ngauss * norbs * 2:]
    gaussian_params = gaussian_params.reshape((ngauss, 4))
    return t_state, lincoeff_initial, gaussian_params


class GaussianRemovalCalculator:
    def __init__(self, lincoeffs, gaussian_params, molecule="LiH", penalty_constant=1e-2):
        """
        The constructor now only requires the linear coefficients and Gaussian nonlinear parameters.
        """
        self.lincoeffs = lincoeffs
        self.params = gaussian_params
        self.molecule = molecule
        self.penalty_constant = penalty_constant
        self.points = None
        self.orbitals_full = None

    def create_grid(self, a=-100, b=100):
        inner_grid = 10
        grid_a = a
        grid_b = b
        points_inner, weights_inner = gaussian_quadrature(-inner_grid, inner_grid, 24 * inner_grid + 1)
        grid_spacing = 0.4
        num_points = int((grid_b - inner_grid) / grid_spacing)
        points_outer1, weights_outer1 = trapezoidal_quadrature(grid_a, -inner_grid, num_points)
        points_outer2, weights_outer2 = trapezoidal_quadrature(inner_grid, grid_b, num_points)
        points = np.concatenate((points_outer1, points_inner, points_outer2))
        weights = np.concatenate((weights_outer1, weights_inner, weights_outer2))
        self.points = points
        GR.points = points if GR.points is None else GR.points
        self.weights = weights
        self.sqrt_weights = np.sqrt(weights)
        return self.points

    def compute_orbitals(self, params, lincoeffs):
        orbitals = GR.make_orbitals(lincoeffs, params)
        return orbitals

    def compute_basis_overlap(self, params):
        functions, _ = GR.setupfunctions(params, self.points)
        scaled_functions = functions * self.sqrt_weights
        S = np.dot(scaled_functions, np.conjugate(scaled_functions).T)
        return S, functions

    def re_fit_without_gaussian(self, params, lincoeffs, remove_index):
        new_params = np.delete(params, remove_index, axis=0)
        deleted_params = params[remove_index]
        results = GR.setupfunctionsandDerivs(new_params, GR.points)
        functions_new = results[0]
        X = functions_new.T * self.sqrt_weights.reshape(-1, 1)
        X_dag = X.conj().T
        XTX = X_dag @ X
        old_action = self.compute_orbitals(params, lincoeffs) * self.sqrt_weights
        n_orbs = old_action.shape[0]
        new_lincoeffs_fit = np.zeros((new_params.shape[0], n_orbs), dtype=np.complex128)
        orbital_errors = np.zeros(n_orbs)
        XTX_inv = np.linalg.inv(XTX + np.eye(XTX.shape[0]) * 1e-14)
        for orbital_index in range(n_orbs):
            Y = old_action[orbital_index]
            XTy = X_dag @ Y
            c = XTX_inv @ XTy
            Y_new = X @ c
            norm = np.linalg.norm(Y - Y_new)
            orbital_errors[orbital_index] = norm
        return new_params, new_lincoeffs_fit, orbital_errors, deleted_params

    def cost_and_grad(self, x, existing_params, F_old, X_old, old_action, nfrozen,
                      num_new, a_min_threshold, max_overlap,
                      return_overlap=False):
        lambd = 1e-16
        candidate = x.reshape((num_new, 4))
        F_new, _, dF_new_a, dF_new_b, dF_new_p, dF_new_q = GR.setupfunctionsandDerivs(candidate, self.points)[:6]
        X_new = F_new.T * self.sqrt_weights.reshape(-1, 1)
        X_total = np.concatenate((X_old, X_new), axis=1)
        threshold=1e-7
        U,Sigma,Vdagger=np.linalg.svd(X_total,full_matrices=False)
        Sigma_invvals=np.where(np.abs(Sigma) > threshold**2, 1.0 / Sigma, 0.0)
        Sigma_inv=np.diag(Sigma_invvals)
        full_pseudoinverse=Vdagger.conj().T@Sigma_inv@U.conj().T
        
        function_derivs = []
        for i in range(num_new):
            function_derivs.extend([dF_new_a[i], dF_new_b[i], dF_new_p[i], dF_new_q[i]])
        n_orbs = old_action.shape[0]
        new_lincoeff = np.zeros((nfrozen + num_new, n_orbs), dtype=np.complex128)
        orbital_errors = np.zeros(n_orbs)
        zs = np.zeros((n_orbs, len(self.points)), dtype=np.complex128)
        overlap_error = 0.0

        for orbital_index in range(n_orbs):
            Y = old_action[orbital_index]
            new_lincoeff[:, orbital_index] = full_pseudoinverse@Y
            Y_new = X_total @ new_lincoeff[:, orbital_index]
            zs[orbital_index] = Y - Y_new
            norm = np.linalg.norm(zs[orbital_index])
            orbital_errors[orbital_index] = norm
            overlap_error += norm ** 2

        grad = np.zeros_like(x)
        Xders = np.array(function_derivs).T * self.sqrt_weights.reshape(-1, 1)
        tempmat=U@Sigma_inv@Vdagger
        for orbital_index in range(old_action.shape[0]):
            c=new_lincoeff[:,orbital_index]
            r=zs[orbital_index]
            for k in range(len(x)):
                #See https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/
                col_index = nfrozen + k // 4
                Xder_k=Xders[:,k]
                Dkc = Xder_k * c[col_index]
                ak=Dkc-U@(U.conj().T@Dkc)
                DkTconj_r = np.zeros_like(c)
                DkTconj_r[col_index] = np.vdot(Xder_k, r)
                bk=tempmat@DkTconj_r
                gradvec=-ak-bk
                grad[k]+=2*np.real(np.vdot(r,gradvec))
        total_error = overlap_error


        #Penalty for a values
        for k in range(num_new):
            a_val = candidate[k, 0]
            if np.abs(a_val) >= a_min_threshold:
                pen = 0.0
                grad_term = 0.0
            else:
                pc = 10 * self.penalty_constant
                ratio = a_min_threshold / np.abs(a_val)
                pen = pc * (ratio - 1) ** 2
                grad_term = -2 * pc * a_min_threshold * (ratio - 1) / (np.abs(a_val) ** 2) * np.sign(a_val)
            total_error += pen
            grad[4 * k] += grad_term

        new_params = np.concatenate((existing_params[:nfrozen, :], candidate), axis=0)
        overlapmatrix_full = self.compute_basis_overlap(new_params)[0]
        onf = overlapmatrix_newfunctions = overlapmatrix_full[nfrozen:, nfrozen:]
        oon = overlapmatrix_oldnew = overlapmatrix_full[:nfrozen, nfrozen:]
        ovlp_max = np.max([np.max(oon), np.max(onf - np.diag(np.diag(onf)))])
        prefac = self.penalty_constant / (1 - max_overlap ** 2)
        # Penalty for overlap
        for i in range(num_new):
            for j in range(i):
                if np.abs(overlapmatrix_newfunctions[i, j]) > max_overlap:
                    penalty_add = prefac * (np.abs(overlapmatrix_newfunctions[i, j]) ** 2 - max_overlap ** 2)
                    total_error += penalty_add
                    for k in range(4):
                        grad[4 * i + k] += 2 * prefac * np.real(
                            overlapmatrix_newfunctions[i, j] *
                            np.sum(function_derivs[k][i] * self.sqrt_weights *
                                   np.conj(F_new[j] * self.sqrt_weights))
                        )
                        grad[4 * j + k] += 2 * prefac * np.real(
                            overlapmatrix_newfunctions[i, j] *
                            np.sum(function_derivs[k][j] * self.sqrt_weights *
                                   np.conj(F_new[i] * self.sqrt_weights))
                        )
        for i in range(nfrozen):
            for j in range(num_new):
                if np.abs(overlapmatrix_oldnew[i, j]) > max_overlap:
                    penalty_add = prefac * (np.abs(overlapmatrix_oldnew[i, j]) ** 2 - max_overlap ** 2)
                    total_error += penalty_add
                    for k in range(4):
                        grad[4 * j + k] += 2 * prefac * np.real(
                            overlapmatrix_oldnew[i, j] *
                            np.sum(function_derivs[k][j] * self.sqrt_weights *
                                   np.conj(F_old[i]) * self.sqrt_weights)
                        )
        overlapmatrix_eigvals,overlapmatrix_eigvecs=np.linalg.eigh(overlapmatrix_full)
        total_error = np.real(total_error)
        grad = np.real(grad)
        ovlp_max = np.real(ovlp_max)
        if return_overlap:
            return total_error, grad, ovlp_max
        return total_error, grad

    def add_new_gaussians(self, existing_params, original_orbitals, num_new=3, initial_guess=None, num_unfrozen_old=0,maxiter=100,max_ovlp=0.975,a_min_threshold=0.1):
        initial_guess_full = np.zeros(num_new * 4)
        max_overlap = max_ovlp

        for i in range(num_new):
            if initial_guess is not None:
                initial_guess_full[4 * i] = initial_guess[0] + (np.random.rand()) * 0.1
            else:
                initial_guess_full[4 * i] = a_min_threshold + (np.random.rand()) * 0.1

            if initial_guess is not None:
                initial_guess_full[4 * i + 1] = initial_guess[1] + (np.random.rand() - 0.5) * abs(initial_guess[1])
                initial_guess_full[4 * i + 2] = initial_guess[2] + (np.random.rand() - 0.5) * abs(initial_guess[2])
                initial_guess_full[4 * i + 3] = initial_guess[3] + (np.random.rand() - 0.5) * abs(initial_guess[3]) 
            else:
                initial_guess_full[4 * i + 1] = 0.0
                initial_guess_full[4 * i + 2] = 0.0
                initial_guess_full[4 * i + 3] = 0.0

        initial_guess = initial_guess_full

        nfrozen = existing_params.shape[0] - num_unfrozen_old
        F_old, _, _, _, _, _ = GR.setupfunctionsandDerivs(existing_params[:nfrozen, :], self.points)[:6]
        X_old = F_old.T * self.sqrt_weights.reshape(-1, 1)
        old_action = original_orbitals * self.sqrt_weights
        num_new = num_new + num_unfrozen_old
        initial_guess = np.concatenate((existing_params[nfrozen:, :].flatten(), initial_guess), axis=0)

        _, grad0,_ = self.cost_and_grad(initial_guess, existing_params, F_old, X_old,
                                          old_action, nfrozen, num_new, 
                                          a_min_threshold, max_overlap,return_overlap=True)
        hess_inv0 = np.diag(1 / (np.abs(grad0) + 1e-14))
        res = minimize(self.cost_and_grad, initial_guess, args=(existing_params, F_old, X_old,
                                                                  old_action, nfrozen, num_new,
                                                                  a_min_threshold, max_overlap),
                       method='BFGS', jac=True, options={"maxiter":maxiter,'gtol': 1e-8, "hess_inv0": hess_inv0})
        new_candidate = res.x.reshape((num_new, 4))
        new_candidate[:, 0] = np.abs(new_candidate[:, 0])
        # Recalculate without penalty.
        penalty_constant_temp = self.penalty_constant
        self.penalty_constant = 0.0
        new_cost_npenalty, _, ovlp_max = self.cost_and_grad(res.x, existing_params, F_old, X_old,
                                                                     old_action, nfrozen, num_new, 
                                                                     a_min_threshold, max_overlap,
                                                                     return_overlap=True)
        self.penalty_constant = penalty_constant_temp
        return res, new_candidate, new_cost_npenalty, ovlp_max

    def run(self,num_sugg,maxiter,max_ovlp_first=0.8, max_ovlp_second=0.95,eigval_issue=False,index=None,num_new=1):
        ngauss_frozen = 20 if self.molecule == "LiH" else 34
        self.create_grid(-400, 400)
        orbitals_full = self.compute_orbitals(self.params, self.lincoeffs)
        self.orbitals_full = orbitals_full
        S, _ = self.compute_basis_overlap(self.params)
        S_off = S.copy()
        np.fill_diagonal(S_off, 0)
        S_off[:ngauss_frozen, :ngauss_frozen] = 0
        abs_S_off = np.abs(S_off)
        max_index = np.unravel_index(np.argmax(abs_S_off), abs_S_off.shape)
        i_max_1, j_max_1 = max_index
        i_max=np.min([i_max_1,j_max_1])
        j_max=np.max([i_max_1,j_max_1]) #J is always the larger index
        print("Maximal off-diagonal overlap element: {:.4f}".format(abs_S_off[i_max, j_max]))
        if i_max<ngauss_frozen and j_max<ngauss_frozen:
            print("Both Gaussians are frozen. No need to reoptimize.")
            return self.params, self.lincoeffs, abs_S_off[i_max, j_max], 0,1
        new_params_j, _,errors_j, deleted_params_j = self.re_fit_without_gaussian(self.params, self.lincoeffs, j_max)
        new_params_i, _,errors_i, deleted_params_i = self.re_fit_without_gaussian(self.params, self.lincoeffs, i_max)
        if sum(errors_i) < sum(errors_j) and i_max>ngauss_frozen: # i needs to be unfrozen
            new_params_j = new_params_i #We remove the one with the smaller error
            errors_j = errors_i
            deleted_params_j=deleted_params_i
        print(i_max, j_max,sum(errors_j))
        best = 1e100
        best_params = None
        if index is not None:
            new_params_j,_,errors_j,deleted_params_j=self.re_fit_without_gaussian(self.params,self.lincoeffs,index)
            print("Overlap matrix wants us to remove Gaussian ",index)
        best_threshold=1e-6
        for k in range(num_sugg):
            _, new_candidate, new_cost, ovlp_max_candidate = self.add_new_gaussians(
                new_params_j, orbitals_full, num_new=num_new, initial_guess=deleted_params_j,maxiter=maxiter,max_ovlp=max_ovlp_first,a_min_threshold=0.12)
            print(new_cost)
            if new_cost < best and ovlp_max_candidate < 0.98:
                best = new_cost
                best_params = new_candidate
                if best <best_threshold:
                    break
        print("Best params cost: {:.6e}".format(best))
        print("We removed params ",deleted_params_j)
        print("We got params",best_params)
        all_new_params = np.concatenate((new_params_j, best_params), axis=0)
        
        n_unfrozen = all_new_params.shape[0] - ngauss_frozen
        _, final_new_candidate, final_cost, final_ovlp_max = self.add_new_gaussians(
            all_new_params, self.orbitals_full, num_new=0, num_unfrozen_old=n_unfrozen,maxiter=maxiter,max_ovlp=max_ovlp_second,a_min_threshold=0.10)
        nfrozen = all_new_params.shape[0] - n_unfrozen
        final_nonlinear_params = np.concatenate((all_new_params[:nfrozen, :], final_new_candidate), axis=0)

        functions, _ = GR.setupfunctions(final_nonlinear_params, self.points)
        X_total = functions.T * self.sqrt_weights.reshape(-1, 1)
        X_total_dag = X_total.conj().T
        XTX = X_total_dag @ X_total
        XTX_inv = np.linalg.inv(XTX + np.eye(XTX.shape[0]) * 1e-14)
        n_orbs = self.orbitals_full.shape[0]
        best_linear = np.zeros((final_nonlinear_params.shape[0], n_orbs), dtype=np.complex128)
        for orbital_index in range(n_orbs):
            Y = self.orbitals_full[orbital_index] * self.sqrt_weights
            XTY = X_total_dag @ Y
            best_linear[:, orbital_index] = XTX_inv @ XTY

        S_final, _ = self.compute_basis_overlap(final_nonlinear_params)
        S_final_off = S_final.copy()
        np.fill_diagonal(S_final_off, 0)
        largest_overlap = np.max(np.abs(S_final_off))

        return final_nonlinear_params, best_linear, largest_overlap, final_cost,num_new

def test_gradient(calc):
    """
    Test the analytical gradient computed in cost_and_grad by comparing
    it to a finite-difference approximation.
    """
    # Use the existing (frozen) parameters.
    existing_params = calc.params  # shape (ngauss, 4)
    nfrozen = existing_params.shape[0]  # all current Gaussians are frozen in this test
    num_new = 1  # test one new Gaussian
    a_min_threshold = 0.1
    max_overlap = 0.975

    # Compute F_old, X_old and old_action from the frozen functions.
    F_old, _, _, _, _, _ = GR.setupfunctionsandDerivs(existing_params[:nfrozen, :], calc.points)[:6]
    X_old = F_old.T * calc.sqrt_weights.reshape(-1, 1)
    orbitals_full = calc.compute_orbitals(calc.params, calc.lincoeffs)
    old_action = orbitals_full * calc.sqrt_weights

    # Choose a random candidate parameter vector (flattened: 4 parameters per new Gaussian).
    x0 = np.random.randn(num_new * 4) * 0.01  # small random perturbation

    # Compute analytical cost and gradient.
    cost, grad_analytical = calc.cost_and_grad(
        x0, existing_params, F_old, X_old, old_action,
        nfrozen, num_new, a_min_threshold, max_overlap, return_overlap=False
    )

    print("Analytical gradient:")
    print(grad_analytical)

    # Compute finite-difference gradient.
    epsilon = 1e-8
    grad_fd = np.zeros_like(x0)
    for i in range(len(x0)):
        x_plus = np.copy(x0)
        x_minus = np.copy(x0)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        cost_plus, _ = calc.cost_and_grad(
            x_plus, existing_params, F_old, X_old, old_action,
            nfrozen, num_new, a_min_threshold, max_overlap, return_overlap=False
        )
        cost_minus, _ = calc.cost_and_grad(
            x_minus, existing_params, F_old, X_old, old_action,
            nfrozen, num_new, a_min_threshold, max_overlap, return_overlap=False
        )
        grad_fd[i] = (cost_plus - cost_minus) / (2 * epsilon)

    print("Finite-difference gradient:")
    print(grad_fd)
    diff_norm = np.linalg.norm(grad_fd - grad_analytical)
    print("Gradient difference norm:", diff_norm)
def main():
    parser = argparse.ArgumentParser(
        description="Calculate final improvement in Gaussian removal and refitting."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file (npz) with wavefunction parameters ('params', 'lincoeff', 'times', and 'nbasis').")
    parser.add_argument("--start_time", type=float, default=0.0,
                        help="Starting time to select the state from the input file (default: 0.0)")
    args = parser.parse_args()
    if "LiH2" in args.input_file:
        molecule = "LiH2"
        norbs = 4
    else:
        molecule = "LiH"
        norbs = 2
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    t_state, lincoeff_initial, gaussian_params = load_input_file(args.input_file, args.start_time, norbs)
    print("Selected state time:", t_state)
    
    calc = GaussianRemovalCalculator(
        lincoeffs=lincoeff_initial,
        gaussian_params=gaussian_params,
        molecule=molecule,
        penalty_constant=1e-2
    )
    # Create grid and compute orbitals, etc.
    calc.create_grid(-200, 200)
    calc.orbitals_full = calc.compute_orbitals(calc.params, calc.lincoeffs)
    
    # Run the gradient test.
    print("=== Testing gradient with finite differences ===")
    test_gradient(calc)
    
    # Continue with the usual optimization.
    final_nonlinear, best_linear, largest_overlap, final_cost = calc.run(num_sugg=10, maxiter=10)
    print("Final optimized nonlinear Gaussian parameters:")
    print(final_nonlinear)
    print("Optimized linear coefficients:")
    print(best_linear)
    print("Largest off-diagonal overlap element: {:.4f}".format(largest_overlap))
    print("Final minimized error: {:.6e}".format(final_cost))


if __name__ == "__main__":
    main()
