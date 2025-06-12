# wf2grid.py
import sys, glob, math
import numpy as np
from numpy import abs,sqrt,exp,pi
F_inp   = float(sys.argv[1])          # laser-field index (0,1,4,…)
maxiter = int(sys.argv[2])            # SCF iterations used in the run
molec   = sys.argv[3]                 # “LiH” | “LiH2”
freeze  = sys.argv[4]                 # “freeze” or “nofreeze” (kept for the filename)
eps     = float(sys.argv[5])          # total Rothe target ε (same you gave the propagator)
optflag = sys.argv[6]                 # “True” | “False”
method  = sys.argv[7]                 # “HF” | “DFT”

initlen = 20 if molec == "LiH" else 34                      # fixed by the original input file
E0      = math.sqrt(F_inp / (3.50944758e2))                # same conversion used in propagation
globpat = f"WF_{method}_{molec}_{E0:.4f}_*_%d_{eps:.3e}.npz"%maxiter  # glob pattern for wave-function files
print("Looking for", globpat)
wf_file = sorted(glob.glob(globpat))[0]                    # first (only) match → wave-function file

data      = np.load(wf_file)
params_ts = data["params"]            # one big vector per saved time
nbasis_ts = data["nbasis"]            # nbasis at each save point
tidx      = -2                        # second-to-last time-step

nb        = int(nbasis_ts[tidx])
norbs     = 2 if molec == "LiH" else 4
full      = params_ts[tidx]

C_real = full[:nb * norbs]
C_imag = full[nb * norbs:2 * nb * norbs]
C      = (C_real + 1j * C_imag).reshape(nb, norbs)         # linear coeffs  (nbasis × norbs)
Gpars  = full[2 * nb * norbs:].reshape(nb, 4)              # Gaussian a,b,p,q per row
print(Gpars.shape)
print(C.shape)
if method=="DFT" and F_inp==4:
    x= np.arange(-500.0, 500.25, 0.25)  # requested grid (3601 pts)
elif F_inp==4:
    x = np.arange(-450.0, 450.25, 0.25)                       # requested grid (3601 pts)
else:
    x = np.arange(-320.0, 320.25, 0.25)                       # requested grid (1601 pts)
def g(par, x):
    a, b, p,q = par
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    return gaussval

basis = np.array([g(Gpars[j], x) for j in range(nb)])      # nbasis × Ngrid
print(basis.shape)
orbit = (C.T @ basis).T                                    # Ngrid × norbs  (match original layout)
Mmax=basis.shape[0]  # number of basis functions used in the calculation
out = f"orbitals_{method}_{molec}_F{F_inp}_{maxiter}_{eps:.0e}_Mmax=%d.npz"% Mmax
np.savez(out, grids=x, orbitals=orbit)
print("Wrote", out)
