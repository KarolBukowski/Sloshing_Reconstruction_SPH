"""
main_3D.py
----------
Sparse reconstruction of the 3D sloshing free-surface from a few wave gauges.

Context
-------
SPH simulations of a rectangular sloshing tank (0.45 m x 0.30 m) are run at
several single-axis (X or Y) and combined (X+Y) forcing frequencies. Each case
gives the free-surface elevation over a full 2D spatial grid at many time steps.
The idea is to learn the dominant spatial shapes from all cases together, then
show that a handful of well-placed gauges on the physical experimental grid is
enough to recover the full 2D surface at any time.

Pipeline
--------
STAGE 1 - Load and preprocess
    - Load all CSV files; drop the boundary columns (x = +/-0.225 m,
      y = +/-0.15 m) where the SPH free surface is not well defined.
    - Remove the mean elevation and cut the initial transient (t < T_START).
    - Save the cleaned fluctuation arrays to the NPZ file.
    - Save a 3D animated GIF of the free surface for one example case.
    - Plot: time series at one (x, y) location.

STAGE 2 - POD and sensor placement
    - Combine all cases into one big data matrix (about 14 751 interior points)
      and decompose it to find the r dominant spatial shapes (POD modes).
    - From those shapes, automatically select p gauge locations out of the
      345 allowed positions on the 2D experimental grid (23 x 15, QR pivoting).
    - Compute the operator that reconstructs the full 2D field from p readings.
    - Plot: energy content of the modes, 2D mode colour maps, sensor locations
      overlaid on the modes, reconstruction operator colour maps.

STAGE 3 - Reconstruction and error evaluation
    - For the chosen case, simulate reading only the p gauge values.
    - Reconstruct the full 2D free surface at every time step, with and without
      added noise on the gauge readings.
    - Repeat for several (r, p) combinations and compare the errors.
    - Plot: side-by-side colour map comparison, time series, error vs (r, p).
    - Save GIFs with three panels: true field, clean rec, noisy rec.

Output files
------------
  outputs_3D/sloshing_3d_data.npz              preprocessed data
  outputs_3D/figures/*.png                     all static figures
  outputs_3D/visualisation_GIFs/*.gif          raw free-surface animation (Stage 1)
  outputs_3D/reconstruction_GIFs/*.gif         reconstruction comparison animations
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend required for 3D GIF rendering

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aux_fun_3D as aux

# =============================================================================
#  PARAMETERS
# =============================================================================

DATA_DIR = "3D_data/"
NPZ_PATH = "outputs_3D/sloshing_3d_data.npz"

T_START = 0.1  # [s] discard data before this time (transient particle relaxation)

r = 4  # number of POD modes to keep
p = 8  # number of sensors to place (must be >= r)

noise_std = 2e-3  # [m] standard deviation of noise added to sensor readings

# experimental grid: the discrete (x, y) positions where sensors can physically be placed
x_exp = np.arange(-0.22, 0.22 + 1e-3, 0.02)  # 23 points in x
y_exp = np.arange(-0.14, 0.14 + 1e-3, 0.02)  # 15 points in y

# which case to reconstruct in Stage 3
case_key = "X1.2Hz_Y1.6Hz"

# =============================================================================
#  STAGE 1 - LOAD AND PREPROCESS ALL CASES
# =============================================================================

all_files = sorted(Path(DATA_DIR).glob("*.csv"))
print(f"Found {len(all_files)} CSV files in {DATA_DIR}")

data_dict = {}
case_keys = []

for filepath in all_files:
    case_num, label, key = aux.parse_case_info(filepath.stem)
    case_keys.append(key)

    print(f" Loading {label} ...", end=" ", flush=True)

    px, py, time_raw, elev_raw = aux.load_case(str(filepath))
    t, fluct, x_ret, y_ret = aux.preprocess(time_raw, elev_raw, px, py, T_START)

    n_dropped = len(px) - len(x_ret)
    print(
        f"n_time={len(t)}, n_space={fluct.shape[1]}  (dropped {n_dropped} border pts)"
    )

    data_dict[key] = fluct  # shape: (n_time, n_space)

# store spatial grid and time once - identical across all cases
data_dict["x"] = x_ret
data_dict["y"] = y_ret
data_dict["time"] = t

Path("outputs_3D").mkdir(exist_ok=True)
np.savez_compressed(NPZ_PATH, **data_dict)
print(f"\nSaved preprocessed data -> {NPZ_PATH}\n")

# --- common spatial grid and time ---
x = data_dict["x"]
y = data_dict["y"]
t = data_dict["time"]

unique_x = np.unique(x)
unique_y = np.unique(y)
nx, ny = len(unique_x), len(unique_y)

print(f"Spatial grid (interior): {nx} x × {ny} y = {nx*ny} points")
print(
    f"Experimental grid      : {len(x_exp)} x × {len(y_exp)} y = {len(x_exp)*len(y_exp)} points"
)
print()

# --- plots for Stage 1 ---

fig_dir = "outputs_3D/figures"
os.makedirs(fig_dir, exist_ok=True)

example_key = case_keys[3]
fluct_example = data_dict[example_key]

# plot 1: free-surface visualisation GIF for the example case
VIS_GIF_DIR = Path("outputs_3D/visualisation_GIFs")
VIS_GIF_DIR.mkdir(parents=True, exist_ok=True)
gif_vis_path = VIS_GIF_DIR / f"{example_key}_3d.gif"

print(f"Generating visualisation GIF for {example_key} ...")
n_vis_frames = aux.make_visualisation_gif(
    x,
    y,
    t,
    fluct_example,
    gif_vis_path,
    GIF_FPS=25,
    GIF_MAX_FRAMES=200,
)
print(f"Visualisation GIF saved -> {gif_vis_path}  ({n_vis_frames} frames)\n")

# plot 2: time series at one spatial location
x_pt, y_pt = 0.0, 0.0
xy_sim = np.column_stack([x, y])
idx_space = np.argmin(((xy_sim - np.array([x_pt, y_pt])) ** 2).sum(axis=1))
closest_x = x[idx_space]
closest_y = y[idx_space]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, fluct_example[:, idx_space])
ax.set_xlabel("t [s]")
ax.set_ylabel("Elevation fluctuation [m]")
ax.set_title(
    f"Case {example_key} - time series at ({closest_x:.3f}, {closest_y:.3f}) m"
)
ax.grid(True)
fig.tight_layout()
fig.savefig(
    f"{fig_dir}/case_{example_key}_timeseries_x{closest_x:.3f}_y{closest_y:.3f}.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()


# =============================================================================
#  STAGE 2 - POD AND SENSOR PLACEMENT
# =============================================================================

X = aux.build_snapshot_matrix(data_dict, case_keys)  # (n_space, n_total)

n_space, n_total = X.shape
print(
    f"Snapshot matrix X : {X.shape}  ({len(case_keys)} cases, {n_total} snapshots total)"
)

U, s, energy = aux.compute_pod(X, r)

print(f"Energy captured by r={r} modes: {energy[r-1]*100:.3f} %")
print()

# plot 1: singular value decay and cumulative energy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(np.arange(1, len(s) + 1), s, "o-", ms=4, color="steelblue")
ax1.axvline(r, color="red", ls="--", lw=1.2, label=f"r = {r}")
ax1.set_xlabel("Mode index")
ax1.set_ylabel("Singular value")
ax1.set_title("Singular value decay")
ax1.set_xlim(0, 20)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(np.arange(1, len(s) + 1), energy * 100, "o-", ms=4, color="darkorange")
ax2.axvline(
    r, color="red", ls="--", lw=1.2, label=f"r = {r}  ({energy[r-1]*100:.2f} %)"
)
ax2.set_xlabel("Mode index")
ax2.set_ylabel("Cumulative energy [%]")
ax2.set_title("Cumulative energy")
ax2.set_ylim(0, 102)
ax2.set_xlim(0, 20)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("POD singular values")
fig.tight_layout()
fig.savefig(
    f"{fig_dir}/POD_singular_values_and_energy.png", dpi=300, bbox_inches="tight"
)
# plt.show()
plt.close()

# plot 2: first n_modes_vis POD modes as pcolormesh
n_modes_vis = min(12, U.shape[1])
ncols_m, nrows_m = 4, (n_modes_vis + 3) // 4

fig, axes = plt.subplots(nrows_m, ncols_m, figsize=(ncols_m * 3.8, nrows_m * 3.2))
axes = np.atleast_1d(axes).ravel()

for k in range(n_modes_vis):
    mode_2d = U[:, k].reshape(nx, ny)
    vmax_k = np.max(np.abs(mode_2d))
    im = axes[k].pcolormesh(
        unique_x,
        unique_y,
        mode_2d.T,
        cmap="RdBu_r",
        vmin=-vmax_k,
        vmax=vmax_k,
        shading="auto",
    )
    axes[k].set_title(f"Mode {k + 1}", fontsize=9)
    axes[k].set_xlabel("x [m]", fontsize=7)
    axes[k].set_ylabel("y [m]", fontsize=7)
    axes[k].set_aspect("equal")
    axes[k].tick_params(labelsize=6)
    plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

for k in range(n_modes_vis, len(axes)):
    axes[k].set_visible(False)

fig.suptitle(f"First {n_modes_vis} POD spatial modes - 3D sloshing", fontsize=12)
fig.tight_layout()
fig.savefig(
    f"{fig_dir}/POD_first_{n_modes_vis}_modes.png", dpi=300, bbox_inches="tight"
)
# plt.show()
plt.close()

# keep first r modes and place sensors
Phi_r = U[:, :r]

gamma, xy_sensors = aux.find_sensors_qr(Phi_r, x, y, x_exp, y_exp, p)
x_sensors = xy_sensors[:, 0]
y_sensors = xy_sensors[:, 1]

print("Sensor positions (x, y) [m]:")
for j, (xi, yi) in enumerate(zip(x_sensors, y_sensors)):
    print(f"  s{j+1}: ({xi:.3f}, {yi:.3f})  ->  full-grid index {gamma[j]}")

R_rec, kappa = aux.build_reconstruction_operator(Phi_r, gamma)

print(f"R_rec shape          : {R_rec.shape}  (n_space x p)")
print(f"Condition number kappa : {kappa:.4e}")
print()

# plot 3: columns of R_rec as pcolormesh (each column = how one sensor contributes)
cmap_s = matplotlib.colormaps["tab10"]
ncols_r = min(p, 4)
nrows_r = (p + ncols_r - 1) // ncols_r

fig, axes = plt.subplots(nrows_r, ncols_r, figsize=(ncols_r * 3.8, nrows_r * 3.2))
axes = np.atleast_1d(axes).ravel()

for j in range(p):
    col_2d = R_rec[:, j].reshape(nx, ny)
    vmax_j = np.max(np.abs(col_2d))
    im = axes[j].pcolormesh(
        unique_x,
        unique_y,
        col_2d.T,
        cmap="RdBu_r",
        vmin=-vmax_j,
        vmax=vmax_j,
        shading="auto",
    )
    axes[j].scatter(
        x_sensors[j],
        y_sensors[j],
        color=cmap_s(j),
        s=80,
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )
    axes[j].set_title(f"column {j+1}  (sensor s{j+1})", fontsize=9)
    axes[j].set_xlabel("x [m]", fontsize=7)
    axes[j].set_ylabel("y [m]", fontsize=7)
    axes[j].set_aspect("equal")
    axes[j].tick_params(labelsize=6)
    plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

for j in range(p, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Columns of reconstruction operator R_rec", fontsize=11)
fig.tight_layout()
fig.savefig(f"{fig_dir}/Reconstruction_operator.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

# plot 4: sensor locations overlaid on each POD mode
_grid = {
    1: (1, 1),
    2: (1, 2),
    3: (2, 2),
    4: (2, 2),
    5: (2, 3),
    6: (2, 3),
}  # 'adaptive' number of subplots
nrows_s, ncols_s = _grid.get(r, (int(np.ceil(r**0.5)),) * 2)

fig, axes = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s * 4.2, nrows_s * 3.8))
axes = np.atleast_1d(axes).ravel()

sensor_colors = [cmap_s(j) for j in range(p)]

for k in range(r):
    ax = axes[k]
    mode_2d = Phi_r[:, k].reshape(nx, ny)
    vmax_k = np.max(np.abs(mode_2d))
    im = ax.pcolormesh(
        unique_x,
        unique_y,
        mode_2d.T,
        cmap="RdBu_r",
        vmin=-vmax_k,
        vmax=vmax_k,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j, (xi, yi) in enumerate(zip(x_sensors, y_sensors)):
        ax.scatter(
            xi,
            yi,
            color=sensor_colors[j],
            s=100,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            f" s{j+1}",
            (xi, yi),
            color=sensor_colors[j],
            fontsize=8,
            fontweight="bold",
            va="center",
        )
    ax.set_title(f"Mode {k + 1}", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)

for k in range(r, len(axes)):
    axes[k].set_visible(False)

fig.suptitle(f"POD modes with sensor placement ({p} sensors, r={r})", fontsize=11)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sensor_locations_on_modes.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()


# =============================================================================
#  STAGE 3 - RECONSTRUCT AND EVALUATE ERROR
# =============================================================================

# make sure the chosen case exists
if case_key not in case_keys:
    raise ValueError(f"case_key '{case_key}' not found. Available: {case_keys}")

eta_true = data_dict[case_key].T  # (n_space, n_time)
t_case = data_dict["time"]

print(f"Reconstructing : {case_key}")

# clean reconstruction
eta_rec, eta_sens = aux.reconstruct(R_rec, gamma, eta_true, noise_std=0.0)
err_clean = aux.compute_error(eta_true, eta_rec)
print(f"Clean error  : {err_clean:.4f}  ({err_clean*100:.2f} %)")

# noisy reconstruction
eta_rec_noisy, eta_sens_noisy = aux.reconstruct(R_rec, gamma, eta_true, noise_std)
err_noisy = aux.compute_error(eta_true, eta_rec_noisy)
print(f"Noisy error  : {err_noisy:.4f}  ({err_noisy*100:.2f} %)")
print(f"(noise std = {noise_std:.1e} m)")
print()

# --- plots for Stage 3 ---

# plot 1: true vs clean reconstruction at one time step (pcolormesh)
t_snapshot = 1.00
idx_t = np.argmin(np.abs(t_case - t_snapshot))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

vmax_snap = np.max(np.abs(eta_true[:, idx_t]))

for ax, field, title in zip(
    [ax1, ax2],
    [eta_true[:, idx_t], eta_rec[:, idx_t]],
    ["True", "Clean rec"],
):
    im = ax.pcolormesh(
        unique_x,
        unique_y,
        field.reshape(nx, ny).T,
        cmap="RdBu_r",
        vmin=-vmax_snap,
        vmax=vmax_snap,
        shading="auto",
    )
    ax.scatter(x_sensors, y_sensors, c="black", s=60, zorder=5, marker="v")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Fluctuation [m]")

fig.suptitle(
    f"Reconstruction at t = {t_case[idx_t]:.3f} s  |  r={r}, p={p}", fontsize=11
)
fig.tight_layout()
fig.savefig(
    f"{fig_dir}/reconstruction_at_snapshot_t_{t_case[idx_t]:.2f}s.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# plot 2: time series at one spatial location
x_pt, y_pt = 0.0, 0.0
idx_space = np.argmin(((xy_sim - np.array([x_pt, y_pt])) ** 2).sum(axis=1))
closest_x = x[idx_space]
closest_y = y[idx_space]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_case, eta_true[idx_space, :], color="steelblue", lw=1.8, label="true")
ax.plot(t_case, eta_rec[idx_space, :], color="darkorange", lw=1.5, label="clean rec")
ax.plot(
    t_case,
    eta_rec_noisy[idx_space, :],
    color="crimson",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label="noisy rec",
)
ax.set_xlabel("t [s]")
ax.set_ylabel("Elevation fluctuation [m]")
ax.set_title(f"Time series at ({closest_x:.3f}, {closest_y:.3f}) m")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(
    f"{fig_dir}/reconstruction_at_x{closest_x:.3f}_y{closest_y:.3f}.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# plot 3 + GIFs: sweep over (r, p) pairs
rp_pairs = [(2, 2), (2, 4), (4, 4), (4, 6), (6, 6), (6, 8)]
n_pairs = len(rp_pairs)

err_clean_list = []
err_noisy_list = []
labels = []

make_gifs = True
OUT_GIFS = Path("outputs_3D/reconstruction_GIFs")
OUT_GIFS.mkdir(parents=True, exist_ok=True)

for k, (r_i, p_i) in enumerate(rp_pairs, start=1):

    Phi_r_i = U[:, :r_i]
    gamma_i, xy_sensors_i = aux.find_sensors_qr(Phi_r_i, x, y, x_exp, y_exp, p_i)
    R_rec_i, kappa_i = aux.build_reconstruction_operator(Phi_r_i, gamma_i)

    eta_rec_i, _ = aux.reconstruct(R_rec_i, gamma_i, eta_true, noise_std=0.0)
    eta_rec_noisy_i, _ = aux.reconstruct(R_rec_i, gamma_i, eta_true, noise_std)

    err_clean_list.append(aux.compute_error(eta_true, eta_rec_i))
    err_noisy_list.append(aux.compute_error(eta_true, eta_rec_noisy_i))
    labels.append(f"r={r_i}, p={p_i}")

    if make_gifs:
        gif_path = OUT_GIFS / f"recon_r{r_i}_p{p_i}_{case_key}.gif"
        print(f"Preparing GIF {k}/{n_pairs}: r={r_i}, p={p_i}")
        n_frames = aux.make_reconstruction_gif(
            x,
            y,
            t_case,
            eta_true,
            eta_rec_i,
            eta_rec_noisy_i,
            xy_sensors_i,
            gif_path,
            r_i,
            p_i,
            GIF_FPS=25,
            GIF_MAX_FRAMES=200,
        )

# plot 3: error summary
x_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x_pos, err_clean_list, "o-", color="steelblue", lw=1.8, ms=7, label="clean")
ax.plot(
    x_pos,
    err_noisy_list,
    "s--",
    color="crimson",
    lw=1.8,
    ms=7,
    label=f"noisy  (sigma = {noise_std:.0e} m)",
)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_xlabel("(r, p) pair")
ax.set_ylabel("Relative error (Frobenius norm)")
ax.set_title(f"Reconstruction error  -  {case_key}")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{fig_dir}/reconstruction_error.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()
