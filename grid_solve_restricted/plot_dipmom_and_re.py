
import os, re, sys
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def compute_hhg_spectrum(t, dip, hann_window=True):
    """Return (ω, |D(ω)|²) for a dipole time series."""
    d  = np.asarray(dip) - dip[0]
    if hann_window:
        win = np.sin(np.pi * t / t[-1])**2
        F   = scipy.fftpack.fftshift(scipy.fftpack.fft(d * win))
    else:
        F   = scipy.fftpack.fftshift(scipy.fftpack.fft(d))
    Px = np.abs(F)**2
    dt = t[1] - t[0]
    ω  = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t))) * 2 * np.pi / dt
    return ω, Px

if len(sys.argv) < 3:
    sys.exit("Syntax:  plot_hhg.py  <molecule>  <I0(W/cm2)>  [HF|DFT]")

molecule          = sys.argv[1]
Wcm2=int(sys.argv[2])
fieldstrength_int = np.sqrt(int(sys.argv[2]))            # √I0 (W½ cm⁻¹)
fieldstrength     = 0.0534 * fieldstrength_int           # a.u.
method            = sys.argv[3] if len(sys.argv) > 3 else "HF"
omega_ref         = 0.06075                              # for scaling

print(f"→ Molecule  : {molecule}")
print(f"→ Method    : {method}")
print(f"→ E₀ (a.u.) : {fieldstrength:.4f}")

grid_solution_exists = False
try:
    base = "/home/simonsch/projects/TDHF/grid-methods/examples/"
    grid_file = (f"{base}grid_solution_{molecule}_{fieldstrength:.3f}.npz"
                 if method == "HF"
                 else f"{base}grid_solution_{molecule}_{fieldstrength:.3f}_DFT.npz")
    g = np.load(grid_file)
    t_grid, dip_grid = g["times"], g["xvals"]
    ω_grid, Px_grid  = compute_hhg_spectrum(t_grid, dip_grid)
    Px_grid         *= (ω_grid / omega_ref)**2
    grid_solution_exists = True
    print(f"→ Grid reference found: {os.path.basename(grid_file)}")
except Exception:
    print("→ No grid reference – continuing without it.")

pat_filestr = f"WF_{method}_{molecule}_{fieldstrength:.4f}_"
files = [f for f in os.listdir(".") if f.startswith(pat_filestr)]

regex = re.compile(
    rf"^WF_{method}_{molecule}_{fieldstrength:.4f}_(?:\d+_)*?"
    r"(\d+)_(\d+)_(\d\.\d+e[+-]\d+)\.npz$"
)

t_list, dip_list, err_list  = [], [], []
ω_list, Px_list             = [], []
Ninit, Nmax, epsilons       = [], [], []
final_err                   = []
gaussians, maxiters         = [], []

for f in files:
    m = regex.match(f)
    if not m:
        print(f"Skipping {f} (name mismatch)")
        continue
    n_init, n_iter, eps = int(m[1]), int(m[2]), float(m[3])

    try:
        d = np.load(f)
        t, dip  = d["times"], d["xvals"]
        errs    = d["rothe_errors"]
        params  = d["params"]

        # gaussian count depends on molecule
        if molecule == "LiH":
            nG = len(params[-1]) // 8
        elif molecule == "LiH2":
            nG = len(params[-1]) // 12
        else:
            nG = len(params[-1]) // 8  # fallback

        ω, Px = compute_hhg_spectrum(t, dip)
        Px   *= (ω / omega_ref)**2

        if t[-1] > 300:         
            ω_list .append(ω)
            Px_list.append(Px)
        else:
            ω_list .append(None)
            Px_list.append(None)

        t_list .append(t)
        dip_list.append(dip)
        err_list.append(errs)
        final_err .append(float(np.sum(errs)))
        Ninit    .append(n_init)
        Nmax     .append(n_iter)
        epsilons .append(eps)
        gaussians.append(nG)
        maxiters .append(n_iter)
        print("Number of Gaussians: %d,rothe error: %.3f, epsilon_h: %.2e, maxiter: %d"%(nG, final_err[-1],eps, n_iter))
    except Exception as E:
        print(f"Cannot read {f} – {E}")

if not t_list:
    sys.exit("No Rothe files found – nothing to plot.")

idx_sorted = sorted(range(len(t_list)), key=lambda i: final_err[i])[::-1]  # low→high
if molecule == "LiH" and method == "HF" and Wcm2 == 1:
    #Swap first and second
    idx_sorted[0], idx_sorted[1] = idx_sorted[1], idx_sorted[0]
palette    = ["red","seagreen","mediumorchid","darkgoldenrod"][::-1]
palette=plt.get_cmap('tab10').colors[:4]
palette=list(palette)
palette[-1]="red"
palette[0]="blue"#plt.get_cmap('tab10').colors[6]
import matplotlib.colors as mcolors
temp=plt.get_cmap('tab10').colors[7]
temp="#ff7f0e"
color = mcolors.to_rgb("#ff7f0e")         # (1.0, 0.498, 0.055)
darker = tuple(c * 0.9 for c in color)    # e.g., ~20% darker
palette[1]="blue"
palette[0]=darker
color_map  = {i: palette[k % len(palette)] for k,i in enumerate(idx_sorted)}
print(palette)

qual_pairs = [(i, final_err[i]) for i in idx_sorted if ω_list[i] is not None]
qual_pairs.sort(key=lambda p: p[1])          # ascending error
low_idx  = [i for i,_ in qual_pairs[:2]]
high_idx = [i for i,_ in qual_pairs[-2:]]

x_min, x_max = (1, 60) if abs(fieldstrength - 0.0534) < 1e-5 else (1, 90)

def hhg_ylim():
    if not grid_solution_exists:
        return None, None
    tail = Px_grid[len(Px_grid)//2:]
    y_max = 6 * np.nanmax(tail[3:])
    y_min = 0.5 * np.nanmin(tail[3:3*x_max])
    return y_min, y_max

y_min, y_max = hhg_ylim()
fig1, ax = plt.subplots(1, 2, figsize=(12, 5))

if grid_solution_exists:
    ax[0].plot(t_grid, dip_grid, c="k", lw=1.2, label="Grid")
for i in idx_sorted:
    ax[0].plot(t_list[i], dip_list[i],
               c=color_map[i],  lw=1.0, alpha=0.9,
               label=rf"$M_{{max}}={gaussians[i]}$")
ax[0].set(xlabel="Time (a.u.)", ylabel="Dipole moment (a.u.)",
          title="Dipole moments")
ax[0].legend(fontsize=8, framealpha=0.6)

if grid_solution_exists:
    ax[1].plot(t_grid[1:], dip_grid[1:]-dip_grid[:-1],
               c="k", lw=1.2, label="Grid")
for i in idx_sorted:
    ax[1].plot(t_list[i][1:],
               dip_list[i][1:]-dip_list[i][:-1],
               c=color_map[i],  lw=1.0, alpha=0.9,
               label=rf"$M_{{max}}={gaussians[i]}$")
ax[1].set(xlabel="Time (a.u.)", ylabel="Δ Dipole",
          title="Dipole derivative")
ax[1].legend(fontsize=8, framealpha=0.6)

fig1.tight_layout()
fig2, ax_e = plt.subplots(figsize=(6, 4))
for i in idx_sorted:
    ax_e.plot(t_list[i], err_list[i],
              c=color_map[i], lw=1.2,
              label=rf"$M_{{max}}={gaussians[i]}$")
ax_e.set(xlabel="Time (a.u.)", ylabel="Rothe error",
         title="Rothe error vs time")
ax_e.legend(fontsize=8, framealpha=0.6)
fig2.tight_layout()



fig3 = plt.figure(figsize=(1.2*3.4, 1.2*7))          # ~2‑column width
gs   = gridspec.GridSpec(3, 1, height_ratios=[0.4, 0.3, 0.3],
                         hspace=0.35)

axA = fig3.add_subplot(gs[0, 0])   # Dipole (legend lives above this axis)
axC = fig3.add_subplot(gs[1, 0])   # Low‑quality HHG
axB = fig3.add_subplot(gs[2, 0])   # High‑quality HHG

#Dipole moment
display_idx = list(dict.fromkeys(low_idx + high_idx))   # grid + 4 runs
lw_ref=1.1
lw_Rothe=1
if grid_solution_exists:
    axA.plot(t_grid, dip_grid, c="k", lw=lw_ref, label="Grid")

for j,i in enumerate(idx_sorted):
    if j==0:
        lbl="frozen"
    else:
        lbl = (rf"$M_{{max}}={gaussians[i]}$"
            if i in display_idx else "_nolegend_")
    axA.plot(t_list[i],
             dip_list[i],
             c=color_map[i],
             lw=lw_Rothe, alpha=0.8,#ls=(0, (2, 1)),
             label=lbl)
axA.set(
    xlabel="Time (a.u.)",
    ylabel="Dipole moment (a.u.)",
    xlim=(-2, 312),
)
axA.set_title("Dipole moments",fontsize=11)

handles, labels = axA.get_legend_handles_labels()

fig3.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.57, 0.95),  # ← was 0.995; move legend *down*
    ncol=3,
    fontsize=10,
    framealpha=0.6,
    columnspacing=0.5,
    borderpad=0.2
)
def hhg_panel(ax, indices, title):
    if grid_solution_exists:
        ax.plot(ω_grid / omega_ref, Px_grid,
                c="k", lw=lw_ref, label="_nolegend_")
    for i in indices[::-1]:
        if ω_list[i] is None:
            continue
        ax.plot(ω_list[i] / omega_ref,
                Px_list[i],
                c=color_map[i],
                lw=lw_Rothe, alpha=0.98,
                ls=	(0, (2, 1)),
                label="_nolegend_")
    ax.set(
        xscale="linear",
        yscale="log",
        xlim=(x_min, x_max),
        xlabel="Harmonic order",
        ylabel="Intensity (arb. units)",
    )
    ax.set_title(title,fontsize=11)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

#High quality HHG
hhg_panel(axB, low_idx,  "HHG spectra – additional Gaussians")

# C) Low quality HHG
hhg_panel(axC, high_idx, "HHG spectra – no additional Gaussians")
fig3.suptitle(
    f"HHG – {molecule}, {method}, $I_0={Wcm2}\\times 10^{{14}}$ W/cm$^2$",
    fontsize=13,
    x=0.57,
    y=0.975                     # ← control vertical placement
)
fig3.subplots_adjust(left=0.18, right=0.98, top=0.85, bottom=0.05)
fig3.savefig(f"plots/HHG_{molecule}_{fieldstrength:.4f}_{method}.pdf")
#plt.show()