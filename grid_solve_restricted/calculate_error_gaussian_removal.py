import numpy as np
import argparse
import sys
import os
from scipy.optimize import minimize
import gauss_Rothe as GR
from quadratures import gaussian_quadrature, trapezoidal_quadrature


class GaussianRemovalCalculator:
    def __init__(self, input_file, start_time, norbs, molecule="LiH",penalty_constant=1e-3):
        self.input_file = input_file
        self.start_time = start_time
        self.norbs = norbs
        self.molecule = molecule
        # These will be set later:
        self.points = None
        self.lincoeffs = None
        self.params = None
        self.orbitals_full = None
        self.penalty_constant = penalty_constant
    def load_input_file(self):
        data = np.load(self.input_file, allow_pickle=True)
        times = data["times"]
        # Select the index of the state closest to the requested start_time.
        index = np.abs(times - self.start_time).argmin()
        t_state = times[index]

        nbasis_arr = data["nbasis"]
        ngauss = int(nbasis_arr[index])

        # Retrieve the saved state from the "params" key.
        state = data["params"][index]
        total_len = len(state)
        # The state is stored as:
        #   [ lincoeff_real (ngauss * norbs),
        #     lincoeff_imag (ngauss_wrong * norbs),
        #     Gaussian parameters (rest) ]
        ngauss_wrong = total_len // (self.norbs * 2 + 4)

        lincoeff_real = state[:ngauss * self.norbs]
        lincoeff_imag = state[ngauss_wrong * self.norbs:(ngauss + ngauss_wrong) * self.norbs]
        lincoeff_initial = (lincoeff_real + 1j * lincoeff_imag).reshape((ngauss, self.norbs))

        if ngauss_wrong - ngauss > 0:
            gaussian_params = state[ngauss_wrong * self.norbs * 2: -4 * (ngauss_wrong - ngauss)]
        else:
            gaussian_params = state[ngauss * self.norbs * 2:]
        gaussian_params = gaussian_params.reshape((ngauss, 4))
        return t_state, lincoeff_initial, gaussian_params

    def create_grid(self, a=-100, b=100, num_points=2000):
        inner_grid=10
        grid_b=b
        grid_a=a
        points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,24*inner_grid+1)
        grid_spacing=0.4#0.4
        num_points=int((grid_b-inner_grid)/grid_spacing)
        points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, num_points)
        points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, num_points)
        points=np.concatenate((points_outer1,points_inner,points_outer2))
        weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))

        self.points = points
        GR.points = points  # set the grid in gauss_Rothe
        self.weights=weights
        self.sqrt_weights=np.sqrt(self.weights)
        return self.points

    def compute_orbitals(self, params, lincoeffs):
        GR.points = self.points
        orbitals = GR.make_orbitals(lincoeffs, params)
        return orbitals

    def compute_basis_overlap(self, params):
        functions, _ = GR.setupfunctions(params, self.points)
        scaled_functions=functions*self.sqrt_weights
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
        F_total = np.concatenate((F_old, F_new), axis=0)
        X_new = F_new.T *  self.sqrt_weights.reshape(-1, 1)
        X_total = np.concatenate((X_old, X_new), axis=1)
        X_total_dag = X_total.conj().T
        XTX = X_total_dag @ X_total
        XTX_inv = np.linalg.inv(XTX + lambd * np.eye(XTX.shape[0]))
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
            XTY = X_total_dag @ Y
            new_lincoeff[:, orbital_index] = XTX_inv @ XTY
            Y_new = X_total @ new_lincoeff[:, orbital_index]
            zs[orbital_index] = Y - Y_new
            norm = np.linalg.norm(zs[orbital_index])
            orbital_errors[orbital_index] = norm
            overlap_error += norm ** 2

        grad = np.zeros_like(x)
        Xders_base = np.zeros_like(X_total)
        Xders = np.array(function_derivs).T * self.sqrt_weights.reshape(-1, 1)

        for i in range(len(x)):
            Xder = Xders_base.copy()
            candidate_col_index = nfrozen + (i // 4)
            Xder[:, candidate_col_index] = Xders[:, i]
            Xder_dag = Xder.conj().T
            for orbital_index in range(n_orbs):
                Y = old_action[orbital_index]
                invmat = XTX_inv
                XTYder = Xder_dag @ Y
                XTY = X_total_dag @ Y
                matrix_der = -invmat @ ((X_total_dag @ Xder) + (Xder_dag @ X_total)) @ invmat
                cder = matrix_der @ XTY + invmat @ XTYder
                gradvec = - (Xder @ new_lincoeff[:, orbital_index] + X_total @ cder)
                grad[i] += 2 * np.real(np.vdot(zs[orbital_index], gradvec))

        total_error = overlap_error

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
        onf=overlapmatrix_newfunctions = overlapmatrix_full[nfrozen:, nfrozen:]
        oon=overlapmatrix_oldnew = overlapmatrix_full[:nfrozen, nfrozen:]
        ovlp_max=np.max([np.max(oon),np.max(onf-np.diag(np.diag(onf)))])
        prefac = self.penalty_constant / (1 - max_overlap ** 2)

        for i in range(num_new):
            for j in range(i):
                if np.abs(overlapmatrix_newfunctions[i, j]) > max_overlap:
                    penalty_add = prefac * (np.abs(overlapmatrix_newfunctions[i, j]) ** 2 - max_overlap ** 2)
                    total_error += penalty_add
                    for k in range(4):
                        grad[4 * i + k] += 2 * prefac * np.real(
                            overlapmatrix_newfunctions[i, j] *
                            np.sum(function_derivs[k][i]*self.sqrt_weights * np.conj(F_new[j]*self.sqrt_weights))
                        )
                        grad[4 * j + k] += 2 * prefac * np.real(
                            overlapmatrix_newfunctions[i, j] *
                            np.sum(function_derivs[k][j]*self.sqrt_weights * np.conj(F_new[i]*self.sqrt_weights))
                        )
        for i in range(nfrozen):
            for j in range(num_new):
                if np.abs(overlapmatrix_oldnew[i, j]) > max_overlap:
                    penalty_add = prefac * (np.abs(overlapmatrix_oldnew[i, j]) ** 2 - max_overlap ** 2)
                    total_error += penalty_add
                    for k in range(4):
                        grad[4 * j + k] += 2 * prefac * np.real(
                            overlapmatrix_oldnew[i, j] *
                            np.sum(function_derivs[k][j]*self.sqrt_weights * np.conj(F_old[i])*self.sqrt_weights)
                        )
        total_error = np.real(total_error)
        grad = np.real(grad)
        ovlp_max = np.real(ovlp_max)
        if return_overlap:
            return total_error, grad, ovlp_max
        return total_error, grad

    def add_new_gaussians(self, existing_params, original_orbitals, num_new=3, initial_guess=None, num_unfrozen_old=0):
        initial_guess_full = np.zeros(num_new * 4)
        if initial_guess is not None:
            a_min_threshold = initial_guess[0] + 0.003
        else:
            a_min_threshold = 0.1005
        max_overlap = 0.975

        for i in range(num_new):
            if initial_guess is not None:
                initial_guess_full[4 * i] = initial_guess[0] + (np.random.rand()) * 0.1
            else:
                initial_guess_full[4 * i] = 0.102 + (np.random.rand()) * 0.1
            if initial_guess_full[4 * i] < a_min_threshold:
                initial_guess_full[4 * i] = a_min_threshold

            if initial_guess is not None:
                initial_guess_full[4 * i + 1] = initial_guess[1] + (np.random.rand() - 0.5) * abs(initial_guess[1])
                initial_guess_full[4 * i + 2] = initial_guess[2] + (np.random.rand() - 0.5) * abs(initial_guess[2])
                initial_guess_full[4 * i + 3] = initial_guess[3] + (np.random.rand() - 0.5) * abs(initial_guess[3]) * 1e-1
            else:
                initial_guess_full[4 * i + 1] = 0.0
                initial_guess_full[4 * i + 2] = 0.0
                initial_guess_full[4 * i + 3] = 0.0

        initial_guess = initial_guess_full


        nfrozen = existing_params.shape[0] - num_unfrozen_old
        F_old, _, _, _, _, _ = GR.setupfunctionsandDerivs(existing_params[:nfrozen, :], self.points)[:6]
        X_old = F_old.T  * self.sqrt_weights.reshape(-1, 1)
        old_action = original_orbitals  * self.sqrt_weights
        n_orbs = old_action.shape[0]

        num_new = num_new + num_unfrozen_old
        initial_guess = np.concatenate((existing_params[nfrozen:, :].flatten(), initial_guess), axis=0)

        cost0, grad0 = self.cost_and_grad(initial_guess, existing_params, F_old, X_old,
                                          old_action, nfrozen, num_new, 
                                          a_min_threshold, max_overlap)
        hess_inv0 = np.diag(1 / (np.abs(grad0) + 1e-14))
        res = minimize(self.cost_and_grad, initial_guess, args=(existing_params, F_old, X_old,
                                                                  old_action, nfrozen, num_new,
                                                                  a_min_threshold, max_overlap),
                       method='BFGS', jac=True, options={'gtol': 1e-8, "hess_inv0": hess_inv0})
        new_candidate = res.x.reshape((num_new, 4))
        new_candidate[:, 0] = np.abs(new_candidate[:, 0])
        new_cost = res.fun
        # Recalculate without penalty
        penalty_constant_temp = self.penalty_constant
        self.penalty_constant = 0.0
        new_cost_npenalty, new_grad2, ovlp_max = self.cost_and_grad(res.x, existing_params, F_old, X_old,
                                                                     old_action, nfrozen, num_new, 
                                                                     a_min_threshold, max_overlap,
                                                                     return_overlap=True)
        self.penalty_constant = penalty_constant_temp
        return res, new_candidate, new_cost_npenalty, ovlp_max

    def run(self):
        # Load state and create grid.
        t_state, lincoeffs, params = self.load_input_file()
        self.lincoeffs = lincoeffs
        self.params = params
        self.create_grid(-100, 100, num_points=2000)

        # Compute original orbitals.
        orbitals_full = self.compute_orbitals(self.params, self.lincoeffs)
        self.orbitals_full = orbitals_full

        # Compute overlap matrix and select index with maximum off-diagonal overlap.
        S, _ = self.compute_basis_overlap(self.params)
        S_off = S.copy()
        np.fill_diagonal(S_off, 0)
        abs_S_off = np.abs(S_off)
        max_index = np.unravel_index(np.argmax(abs_S_off), abs_S_off.shape)
        i_max, j_max = max_index
        print("Maximal off-diagonal overlap element: {:.4f}".format(abs_S_off[i_max, j_max]))
        # Use removal with index j_max for further optimization.
        new_params_j, _, _, deleted_params = self.re_fit_without_gaussian(self.params, self.lincoeffs, j_max)

        # Optimize new candidate Gaussian parameters over several trials.
        best = 1e100
        best_params = None
        for k in range(20):
            res_opt, new_candidate, new_cost, ovlp_max_candidate = self.add_new_gaussians(
                new_params_j, orbitals_full, num_new=2, initial_guess=deleted_params)
            print(new_cost,ovlp_max_candidate)
            if new_cost < best and ovlp_max_candidate < 0.98:
                best = new_cost
                best_params = new_candidate
                if best<1e-6:
                    break
        if best_params is None:
            best_params = np.empty((0, 4))
        print("Best params cost: {:.6e}".format(best))
        # Combine parameters from removal and new candidate.
        all_new_params = np.concatenate((new_params_j, best_params), axis=0)
        ngauss_frozen = 20 if self.molecule == "LiH" else 34
        n_unfrozen = all_new_params.shape[0] - ngauss_frozen
        res_final, final_new_candidate, final_cost, final_ovlp_max = self.add_new_gaussians(
            all_new_params, self.orbitals_full, num_new=0, num_unfrozen_old=n_unfrozen)
        nfrozen = all_new_params.shape[0] - n_unfrozen
        final_nonlinear_params = np.concatenate((all_new_params[:nfrozen, :], final_new_candidate), axis=0)

        # Compute best-fit linear coefficients for the final set of nonlinear parameters.
        functions, _ = GR.setupfunctions(final_nonlinear_params, self.points)
        X_total = functions.T  * self.sqrt_weights.reshape(-1, 1)
        X_total_dag = X_total.conj().T
        XTX = X_total_dag @ X_total
        XTX_inv = np.linalg.inv(XTX + np.eye(XTX.shape[0]) * 1e-14)
        n_orbs = self.orbitals_full.shape[0]
        best_linear = np.zeros((final_nonlinear_params.shape[0], n_orbs), dtype=np.complex128)
        for orbital_index in range(n_orbs):
            Y = self.orbitals_full[orbital_index]  * self.sqrt_weights
            XTY = X_total_dag @ Y
            best_linear[:, orbital_index] = XTX_inv @ XTY

        # Compute the largest off-diagonal overlap element for the final basis.
        S_final, _ = self.compute_basis_overlap(final_nonlinear_params)
        S_final_off = S_final.copy()
        np.fill_diagonal(S_final_off, 0)
        largest_overlap = np.max(np.abs(S_final_off))

        return final_nonlinear_params, best_linear, largest_overlap, final_cost


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
        molecule="LiH2"
        norbs=4
    else:
        molecule="LiH"
        norbs=2
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    calc = GaussianRemovalCalculator(
        input_file=args.input_file,
        start_time=args.start_time,
        norbs=norbs,
        molecule=molecule,
        penalty_constant=1e-2
    )
    final_nonlinear, best_linear, largest_overlap, final_cost = calc.run()
    print("Final optimized nonlinear Gaussian parameters:")
    print(final_nonlinear)
    print("Optimized linear coefficients:")
    print(best_linear)
    print("Largest off-diagonal overlap element: {:.4f}".format(largest_overlap))
    print("Final minimized error: {:.6e}".format(final_cost))


if __name__ == "__main__":
    main()
