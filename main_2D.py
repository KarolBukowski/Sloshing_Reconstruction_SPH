"""
main_2D.py
----------
Sparse reconstruction of the 2D sloshing free-surface from a few wave gauges.

Context
-------
SPH simulations of a rectangular sloshing tank (0.45 m long) are run at several
forcing frequencies. Each case gives the free-surface elevation along the tank
centreline at many spatial points and time steps. The idea is to learn the
dominant spatial shapes from all cases together, then show that just a handful
of well-placed gauges is enough to recover the full surface at any time.

Input
-----
  2D_data/*.csv  - one CSV file per forcing frequency.
  File format:
    Row 1   - x positions [m] of each column, separated by ";"
    Row 2-3 - metadata (skipped)
    Row 4   - column headers (Part ; Time [s] ; Elevation_0 [m] ; ...)
    Row 5+  - data rows

Pipeline
--------
STAGE 1 - Load and preprocess
    - Load all CSV files, drop N_CROP points from each spatial end because
      SPH results near the wall are unreliable.
    - Remove the mean elevation and cut the initial transient (t < T_START).
    - Save the cleaned fluctuation arrays to the NPZ file.
    - Plot: free-surface profile at one instant, time series at one location.

STAGE 2 - POD and sensor placement
    - Combine all cases into one big data matrix and decompose it to find the
      r dominant spatial shapes (POD modes).
    - From those shapes, automatically select p gauge locations out of 23
      allowed positions on the experimental grid (QR pivoting).
    - Compute the operator that reconstructs the full profile from p readings.
    - Plot: energy content of the modes, mode shapes, reconstruction operator.

STAGE 3 - Reconstruction and error evaluation
    - For the chosen case, simulate reading only the p gauge values.
    - Reconstruct the full free surface at every time step, with and without
      added noise on the gauge readings.
    - Repeat for several (r, p) combinations and compare the errors.
    - Plot: snapshot and time-series comparison, error vs (r, p).
    - Save GIFs comparing the true field with clean and noisy reconstructions.

Output files
------------
  outputs_2D/sloshing_data.npz          preprocessed data
  outputs_2D/figures/*.png              all static figures
  outputs_2D/reconstruction_GIFs/*.gif  animated reconstructions
"""

import numpy as np
import matplotlib.pyplot as plt
import aux_fun_2D as aux
import re
import os
from pathlib import Path

# =============================================================================
#  PARAMETERS
# =============================================================================

DATA_DIR = "2D_data/"  # folder with the raw CSV files
NPZ_PATH = "outputs_2D/sloshing_data.npz"  # where the preprocessed data is saved
FIG_DIR = "outputs_2D/figures"
os.makedirs(FIG_DIR, exist_ok=True)

T_START = 1.0  # [s] discard data before this time (transient particle relaxation)
N_CROP = 5  # points dropped from each spatial end (SPH near-wall boundary error)

r = 4  # number of POD modes to keep
p = 6  # number of sensors to place (must be >= r)

noise_std = 1e-3  # [m] standard deviation of noise added to sensor readings

# experimental grid: the discrete positions where sensors can physically be placed
x_exp = np.arange(-0.22, 0.22 + 1e-3, 0.02)  # 23 evenly spaced positions [m]

# which case to reconstruct in Stage 3
case_key = "1.5Hz"


# =============================================================================
#  STAGE 1 - LOAD AND PREPROCESS ALL CASES
# =============================================================================

# find all CSV files and sort them by frequency
all_files = sorted(Path(DATA_DIR).glob("*.csv"))
print(f"Found {len(all_files)} CSV files in {DATA_DIR}")


# load, preprocess, and save each case
data_dict = {}  # will collect all arrays to save into the .npz
case_keys = []

for filepath in all_files:

    # extract forcing frequency from the filename, e.g. "1p5Hz" -> 1.5
    m = re.search(r"(\d+)p(\d+)Hz", filepath.stem)
    freq = float(m.group(1) + "." + m.group(2))
    key = f"{freq:.1f}" + "Hz"
    case_keys.append(key)

    print(f" Loading {key} ...")

    x, time_raw, elev_raw = aux.load_case(str(filepath))
    t, fluct, x = aux.preprocess(time_raw, elev_raw, T_START, x, N_CROP)

    # store in data_dict
    data_dict[key] = fluct  # shape: (n_time, n_space)

# store in data_dict once
data_dict["x"] = x
data_dict["time"] = t

# save everything to one compressed file
Path("outputs_2D").mkdir(exist_ok=True)
np.savez_compressed(NPZ_PATH, **data_dict)
print(f"\nSaved preprocessed data -> {NPZ_PATH}\n")

# --- plots for Stage 1 -------------

example_key = case_keys[3]
fluct_example = data_dict[example_key]

t_example = 1.00
x_example = 0.15

idx_t = np.argmin(np.abs(t - t_example))
closest_time = t[idx_t]

# plot the fluctuation profile at one time step
plt.figure()
plt.plot(x, fluct_example[idx_t, :])
plt.xlabel("x [m]")
plt.ylabel("Elevation fluctuation [m]")
plt.title(f"Case {example_key} - snapshot at t = {closest_time:.2f} s")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"{FIG_DIR}/case_{example_key}_snapshot_t_{closest_time:.2f}s.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()


idx_x = np.argmin(np.abs(x - x_example))
closest_x = x[idx_x]

# plot the time series at one spatial position
plt.figure()
plt.plot(t, fluct_example[:, idx_x])
plt.xlabel("t [s]")
plt.ylabel("Elevation fluctuation [m]")
plt.title(f"Case {example_key} - time series at x = {closest_x:.3f} m")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"{FIG_DIR}/case_{example_key}_timeseries_x_{closest_x:.3f}m.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# =============================================================================
#  STAGE 2 - POD AND SENSOR PLACEMENT
# =============================================================================

# build snapshot matrix
X = aux.build_snapshot_matrix(data_dict, case_keys)  # shape: (n_space, n_total)

n_space, n_total = X.shape
print(
    f"Snapshot matrix X : {X.shape}  ({len(case_keys)} cases, {n_total} snapshots total)"
)

# compute POD
U, s, energy = aux.compute_pod(X, r)

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
plt.savefig(
    f"{FIG_DIR}/POD_singular_values_and_energy.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# plot 2: first plot_r POD modes

plot_r = 6

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(plot_r):
    ax.plot(x, U[:, i], lw=1.5, label=f"mode {i+1}")

ax.axhline(0, color="gray", lw=0.7, ls=":")
ax.set_xlabel("x [m]")
ax.set_ylabel("Mode amplitude")
ax.set_title(f"First {plot_r} POD modes")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(
    f"{FIG_DIR}/POD_first_{plot_r}_modes.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()


# keep first r modes
Phi_r = U[:, :r]

# place sensors on the experimental grid
gamma, x_sensors = aux.find_sensors_qr(Phi_r, x, x_exp, p, plot_it=False)

print(f"Sensor indices gamma : {gamma}")
print(f"Sensor positions [m] : {np.round(x_sensors, 3)}")

# build reconstruction operator

R_rec, kappa = aux.build_reconstruction_operator(Phi_r, gamma)

print(f"R_rec shape          : {R_rec.shape}  (n_space x p)")
print(f"Condition number kappa : {kappa:.4e}")


# plot 3: columns of R_rec (each column = how one sensor contributes to the full field)
fig, ax = plt.subplots(figsize=(10, 5))

for j in range(p):
    ax.plot(
        x,
        R_rec[:, j],
        lw=1.5,
        label=f"column {j+1}  (sensor at x={x_sensors[j]:.3f} m)",
    )
    ax.axvline(x_sensors[j], ls="--", lw=0.8, alpha=0.5)

ax.set_xlabel("x [m]")
ax.set_ylabel("Response [m / m]")
ax.set_title("Columns of reconstruction operator R_rec")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(
    f"{FIG_DIR}/Reconstruction_operator.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()


# =============================================================================
#  STAGE 3 - RECONSTRUCT AND EVALUATE ERROR
# =============================================================================


# eta_true must be (n_space, n_time) for the reconstruction: transpose data_dict case
eta_true = data_dict[case_key]
eta_true = eta_true.T  # shape: (n_space, n_time)

# clean reconstruction
eta_rec, eta_sens = aux.reconstruct(R_rec, gamma, eta_true, noise_std=0.0)
err_clean = aux.compute_error(eta_true, eta_rec)

print(f"Clean error  : {err_clean:.4f}  ({err_clean*100:.2f} %)")

# noisy reconstruction
eta_rec_noisy, eta_sens_noisy = aux.reconstruct(R_rec, gamma, eta_true, noise_std)
err_noisy = aux.compute_error(eta_true, eta_rec_noisy)

print(f"Noisy error  : {err_noisy:.4f}  ({err_noisy*100:.2f} %)")
print(f"(noise std = {noise_std:.1e} m)")


# plots for Stage 3 ---------------------------------------------------

# plot 1: true vs reconstructed at one time step
t_example = 1.00

idx_t = np.argmin(np.abs(t - t_example))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, eta_true[:, idx_t], color="steelblue", lw=2.0, label="true")
ax.plot(x, eta_rec[:, idx_t], color="darkorange", lw=1.5, label="clean rec")
ax.plot(
    x,
    eta_rec_noisy[:, idx_t],
    color="crimson",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label="noisy rec",
)

for xi in x_sensors:
    ax.axvline(xi, color="gray", ls=":", lw=0.8)
    ax.plot(xi, 0, "v", color="black", ms=6, zorder=5)

ax.set_xlabel("x [m]")
ax.set_ylabel("Elevation fluctuation [m]")
ax.set_title(f"Reconstruction at t = {t[idx_t]:.3f} s  |  r={r}, p={p}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(
    f"{FIG_DIR}/reconstruction_at_snapshot_t_{t[idx_t]:.2f}s.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# plot 2: time series at one spatial position
x_example = 0.15
idx_x = np.argmin(np.abs(x - x_example))
closest_x = x[idx_x]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, eta_true[idx_x, :], color="steelblue", lw=1.8, label="true")
ax.plot(t, eta_rec[idx_x, :], color="darkorange", lw=1.5, label="clean rec")
ax.plot(
    t,
    eta_rec_noisy[idx_x, :],
    color="crimson",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label="noisy rec",
)

ax.set_xlabel("t [s]")
ax.set_ylabel("Elevation fluctuation [m]")
ax.set_title(f"Time series at x = {closest_x:.3f} m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(
    f"{FIG_DIR}/reconstruction_at_x_{closest_x:.3f}m.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()

# plot 3: error summary across a range of (r, p) pairs
# --- compare different number of modes r and number of sensor p values -------------

rp_pairs = [(2, 2), (2, 4), (4, 4), (4, 6), (6, 6), (6, 8)]
n_pairs = len(rp_pairs)

err_clean_list = []
err_noisy_list = []
labels = []

make_gifs = True
OUT_GIFS = Path("outputs_2D/reconstruction_GIFs")
OUT_GIFS.mkdir(parents=True, exist_ok=True)
for k, (r_i, p_i) in enumerate(rp_pairs, start=1):

    Phi_r_i = U[:, :r_i]
    gamma_i, x_sensors_i = aux.find_sensors_qr(Phi_r_i, x, x_exp, p_i)
    R_rec_i, kappa_i = aux.build_reconstruction_operator(Phi_r_i, gamma_i)

    eta_rec_i, _ = aux.reconstruct(R_rec_i, gamma_i, eta_true, noise_std=0.0)
    eta_rec_noisy_i, _ = aux.reconstruct(R_rec_i, gamma_i, eta_true, noise_std)

    err_clean_list.append(aux.compute_error(eta_true, eta_rec_i))
    err_noisy_list.append(aux.compute_error(eta_true, eta_rec_noisy_i))
    labels.append(f"r={r_i}, p={p_i}")

    if make_gifs:
        gif_path = OUT_GIFS / f"recon_r{r_i}_p{p_i}.gif"
        print(f"Preparing GIF {k}/{n_pairs}: r={r_i}, p={p_i}")

        n_frames = aux.make_reconstruction_gif(
            x,
            t,
            eta_true,
            eta_rec_i,
            eta_rec_noisy_i,
            x_sensors_i,
            gif_path,
            r_i,
            p_i,
            GIF_FPS=25,
            GIF_MAX_FRAMES=500,
        )

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
plt.savefig(
    f"{FIG_DIR}/reconstruction_error.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()
