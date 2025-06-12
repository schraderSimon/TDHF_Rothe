#!/usr/bin/env python3
"""
Plots orbital densities plus their absolute deviation from a reference.

Order & colours:
  • zero-file first      → yellow-ish
  • highest ε  next      → blue
  • middle  ε            → green
  • lowest  ε            → red
"""

import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# 1. CLI & file discovery
# ──────────────────────────────────────────────────────────────────────
F_inp  = float(sys.argv[1])      # 0, 1, 4 …
molec  = sys.argv[2]             # LiH | LiH2
method = sys.argv[3]             # HF  | DFT

pattern = f"orbitals_{method}_{molec}_F{F_inp}_*.npz"
files   = glob.glob(pattern)
if not files:
    print("No files matching", pattern)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 2. custom sort
# ──────────────────────────────────────────────────────────────────────
def sort_key(fp):
    base   = os.path.basename(fp).replace(".npz", "")
    p      = base.split("_")
    param1 = int(p[-3])               # 0 or 300
    param2 = float(p[-2])             # 1e+01 → 10.0
    return (0,) if param1 == 0 else (1, -param2)

files = sorted(files, key=sort_key)
print("Sorted files:\n  " + "\n  ".join(files))

# ──────────────────────────────────────────────────────────────────────
# 3. reference density
# ──────────────────────────────────────────────────────────────────────
ref_path = f"../grid-methods/examples/orbitals_{molec}_{int(F_inp)}_{method}.npz"
ref      = np.load(ref_path)
x_ref    = ref["grids"]
psi_ref  = ref["orbitals"]

rho_ref  = np.zeros_like(x_ref)
mult     = 2 if method == "DFT" else 8
for i in range(psi_ref.shape[1]):
    rho_ref += mult * np.abs(psi_ref[:, i])**2

# ──────────────────────────────────────────────────────────────────────
# 4. figure & palette
# ──────────────────────────────────────────────────────────────────────
fig, (ax_dens, ax_diff) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 5), )

palette = [
    (0.9, 0.44823529411764707, 0.049411764705882356),  # yellow-ish
    "blue",
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # green
    "red",
]

# reference (top only)
ax_dens.plot(
    x_ref,
    rho_ref,
    label="Grid",
    color="black",
    linewidth=1,
)

# ──────────────────────────────────────────────────────────────────────
# 5. loop over data files
# ──────────────────────────────────────────────────────────────────────
idx=0
for colour, fp in zip(palette, files):
    
    d   = np.load(fp)
    x   = d["grids"]
    dx  = x[1] - x[0]
    psi = d["orbitals"]

    rho = np.sum(2 * np.abs(psi)**2, axis=1)   # density

    # label uses the Mmax token
    mmax = os.path.basename(fp).split("_")[-1].split("=")[1]
    mmax=mmax.split(".")[0]  # remove .npz
    if idx==0:
        label="frozen"
    else:
        label= r"$M_{max.}$=%d"%int(mmax)
    ax_dens.plot(x, rho, label=label, color=colour)

    # absolute difference curve
    ax_diff.plot(x, np.abs(rho - rho_ref), color=colour)

    # integrated difference (still printed to console)
    integral = np.abs(rho - rho_ref).sum() * dx
    norm=np.abs(rho_ref).sum() * dx
    print(f"{fp:>50s}   ∫|Δρ| dx = {integral:.3e}, norm: %.2f"%(norm))
    idx+=1
# ──────────────────────────────────────────────────────────────────────
# 6. cosmetics
# ──────────────────────────────────────────────────────────────────────
for ax in (ax_dens, ax_diff):
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 2)
    ax.set_xlim(-400, 410) if F_inp == 4 else ax.set_xlim(-320, 320)
ax_dens.set_ylabel(r"Electronic density $\rho(x)$")
ax_diff.set_ylabel(r"$|\rho_{\rm gauss}(x) - \rho_{\rm grid}(x)|$")
ax_diff.set_xlabel("Position $x$ [a.u.]")
plt.suptitle(r"Densities - %s, %s, $I_0= %d\times10^{14}$ W/cm$^2$" % (molec, method, F_inp))
#Legend with reduced gap between l
ax_dens.legend(
    loc="upper right",
    ncol=2,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=1.0,
)
plt.tight_layout()
plt.savefig(f"plots/orbitals_{method}_{molec}_F{F_inp}.pdf", bbox_inches="tight")
plt.show()
