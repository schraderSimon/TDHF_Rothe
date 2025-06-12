#!/usr/bin/env python3
"""
Poster plot  –  two stacked panels
──────────────────────────────────
(1) Electronic densities  (log-scale)
(2) HHG spectra           (log-scale, two best runs + grid reference)

Invocation order is **exactly** the same as in your very first script:

    python poster_plot.py  <F>  <molecule>  [HF|DFT]

Examples
--------
    python poster_plot.py  4   LiH2  HF
    python poster_plot.py  1   LiH   DFT
"""

# Std-lib
import os, re, sys, glob, itertools
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.fftpack

# ─────────────────────────────────────────────────────────────
#  1. Command-line arguments  (same order as your originals!)
# ─────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    sys.exit("Syntax:  poster_plot.py  <F>  <molecule>  [HF|DFT]")

F_inp   = float(sys.argv[1])          # laser-intensity index: 0, 1, 4 …
molec   = sys.argv[2]                 # LiH   | LiH2
method  = sys.argv[3] if len(sys.argv) > 3 else "HF"

print(f"→ Molecule : {molec}")
print(f"→ Method   : {method}")
print(f"→ F        : {F_inp}")

# ─────────────────────────────────────────────────────────────
#  2. Palette & helper
# ─────────────────────────────────────────────────────────────
PALETTE = [
    (0.90, 0.45, 0.05),   # yellow-orange (“frozen”)
    "#3693d4",            # tab10 blue
    "#48bd48",            # tab10 green
    "#fc5151",            # tab10 red
]

def compute_hhg_spectrum(t, dip, hann_window=True):
    """Return (ω, |D(ω)|²) from dipole-vs-time array."""
    d = np.asarray(dip, float) - dip[0]
    if hann_window:
        win = np.sin(np.pi * t / t[-1])**2
        F   = scipy.fftpack.fftshift(scipy.fftpack.fft(d * win))
    else:
        F   = scipy.fftpack.fftshift(scipy.fftpack.fft(d))
    Px = np.abs(F)**2
    dt = t[1] - t[0]
    ω  = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t))) * 2*np.pi/dt
    return ω, Px

omega_ref = 0.06075   # same normalisation you used

# ─────────────────────────────────────────────────────────────
#  3. Density data  ─ find & sort exactly like your first script
# ─────────────────────────────────────────────────────────────
dens_pat   = f"orbitals_{method}_{molec}_F{F_inp}_*.npz"  # ← float “4.0”
dens_files = glob.glob(dens_pat)
if not dens_files:
    sys.exit(f"No density files match {dens_pat}")

def sort_key(fp):
    base   = os.path.basename(fp).replace(".npz", "").split("_")
    p1     = int(base[-3])          # 0 (frozen) or 300 …
    p2     = float(base[-2])        # ε
    return (0,) if p1 == 0 else (1, -p2)

dens_files = sorted(dens_files, key=sort_key)
print("→ Density files (sorted):")
for f in dens_files:
    print("   ", f)

# Load grid-reference density – *int(F_inp)* (4, not 4.0)
ref_path = f"../grid-methods/examples/orbitals_{molec}_{int(F_inp)}_{method}.npz"
try:
    ref      = np.load(ref_path)
    x_ref    = ref["grids"]
    psi_ref  = ref["orbitals"]
    mult     = 2 if method == "DFT" else 8
    rho_ref  = (mult * np.abs(psi_ref)**2).sum(axis=1)
    ref_ok   = True
    print(f"→ Reference density file: {ref_path}")
except Exception as e:
    print("   ! Could not load grid density reference:", e)
    ref_ok = False

# Map mmax → colour so HHG can reuse the same shades
mmax_to_colour = {}
for col, fp in zip(PALETTE, dens_files):
    mmax = int(fp.split("=")[-1].split(".")[0])
    mmax_to_colour[mmax] = col

# ─────────────────────────────────────────────────────────────
#  4. HHG data  – exactly your old parsing logic
# ─────────────────────────────────────────────────────────────
fieldstrength = 0.0534 * np.sqrt(F_inp)   # F_inp is I0 (W/cm²) index
wfname_head   = f"WF_{method}_{molec}_{fieldstrength:.4f}_"

WF_regex = re.compile(
    rf"^{wfname_head}(?:\d+_)*?(\d+)_(\d+)_(\d\.\d+e[+-]\d+)\.npz$"
)

runs = []
for fn in filter(lambda f: f.startswith(wfname_head), os.listdir(".")):
    m = WF_regex.match(fn)
    if not m:
        continue
    n_init, n_iter, eps = map(float, m.groups())
    try:
        d = np.load(fn)
        t, dip, errs = d["times"], d["xvals"], d["rothe_errors"]
        ω, Px        = compute_hhg_spectrum(t, dip)
        Px          *= (ω / omega_ref)**2
        nG           = len(d["params"][-1]) // (8 if molec=="LiH" else 12)
        runs.append(
            dict(fn=fn, t=t, dip=dip, ω=ω, Px=Px,
                 err_total=np.sum(errs),
                 mmax=nG)                      # mmax ≈ nG
        )
    except Exception as E:
        print("   ! Could not read", fn, ":", E)

if not runs:
    sys.exit("No WF_* files found – nothing to plot.")

# Two **best** (lowest Rothe error) runs
runs.sort(key=lambda d: d["err_total"])
best_runs = runs[:3]

print("→ HHG: two best runs:")
for r in best_runs:
    print("   ", r["fn"], "  (mmax =", r["mmax"], ")")

# Grid-reference HHG (optional)
grid_hhg_ok = False
try:
    base = "../grid-methods/examples/"
    grid_fn = (f"{base}grid_solution_{molec}_{fieldstrength:.3f}"
               f"{'_DFT' if method=='DFT' else ''}.npz")
    g = np.load(grid_fn)
    ω_grid, Px_grid = compute_hhg_spectrum(g["times"], g["xvals"])
    Px_grid        *= (ω_grid / omega_ref)**2
    grid_hhg_ok = True
    print(f"→ HHG grid reference: {os.path.basename(grid_fn)}")
except Exception as e:
    print("   ! No HHG grid reference:", e)

grid_ok= grid_hhg_ok and ref_ok
# ====================================================
# 5.  P L O T T I N G  (legend a bit higher)
# ====================================================


ticksize=20
labelsize=22
legendsize=20
lw=2.2



fig, (ax_dens, ax_hhg) = plt.subplots(
    2, 1, figsize=(10, 4.5),
    gridspec_kw={'height_ratios': [1, 1], 'hspace': -0.5},
    constrained_layout=True
)

# ── densities ───────────────────────────────────────
ax_dens.plot(x_ref, rho_ref, c="k", lw=lw+0.5, label="Grid")
for idx, (fp, col) in enumerate(zip(dens_files, PALETTE)):
    d      = np.load(fp)
    x      = d["grids"]
    rho    = (2 * np.abs(d["orbitals"])**2).sum(axis=1)
    mmax   = int(fp.split("=")[-1].split(".")[0])
    lbl    = "frozen" if idx == 0 else rf"$M_{{max}}={mmax}$"
    ax_dens.plot(x, rho, c=col, lw=lw, label=lbl,ls=(0, (2, 1)))

ax_dens.set_yscale("log")
ax_dens.set_ylim(1e-7, 2)
ax_dens.set_xlim(-400, 410) if F_inp == 4 else ax_dens.set_xlim(-230, 270)
ax_dens.set_ylabel(r"$\rho(x,t=t_f)$", fontsize=labelsize)
ax_dens.set_xlabel(r"$x$ (a.u.)", fontsize=labelsize, labelpad=4)
ax_dens.tick_params(labelsize=ticksize)
ax_dens.set_yticks([1e-7, 1e-4, 1e-1])
# ── HHG spectra ─────────────────────────────────────
if grid_ok:
    ax_hhg.plot(ω_grid/omega_ref, Px_grid, c="k", lw=lw+0.5, label="Grid")

for run in best_runs[::-1]:
    colour = mmax_to_colour.get(run["mmax"], "#1f77b4")
    lbl    = rf"$M_{{max}}={run['mmax']}$"
    ax_hhg.plot(run["ω"]/omega_ref, run["Px"],
                c=colour, lw=lw, ls=(0, (2, 1)), label=lbl)

ax_hhg.set(xscale="linear", yscale="log",
           xlim=(1, 90) if F_inp != 1 else (1, 60))
if grid_ok:
    tail = Px_grid[len(Px_grid)//2:]
    ax_hhg.set_ylim(0.75*tail[3:3*90].min(), tail.max())

ax_hhg.set_xlabel("Harmonic order", fontsize=labelsize, labelpad=4)
ax_hhg.set_ylabel("Intensity",     fontsize=labelsize)
ax_hhg.tick_params(labelsize=ticksize)
ax_hhg.set_yticks([1e-4, 1e0,1e4])
# ── legend: centred, *slightly* above axes (5 columns) ──
handles, labels = ax_dens.get_legend_handles_labels()
leg = fig.legend(handles, labels,
                 loc="upper center",
                 bbox_to_anchor=(0.55, 1.15),   # ← 1.10 instead of 1.02
                 ncol=5,
                 fontsize=legendsize,
                 framealpha=0.6,
                 columnspacing=0.6,
                 handlelength=1.0,
                 handletextpad=0.25,
                 borderaxespad=0.3)

# ====================================================
# 6.  S A V E   (include legend in bbox)
# ====================================================
out = f"plots/POSTER_{molec}_F{F_inp}_{method}.svg"
fig.savefig(out,
            dpi=300,
            bbox_inches='tight',
            bbox_extra_artists=[leg],   # ← ensures legend is included
            pad_inches=0.02)
print("→ Saved", out)
plt.show()
