"""
aux_fun_2D.py
-------------
Helper functions for the 2D sloshing free-surface analysis.

Functions
---------
load_case(filepath)
    Read one CSV file -> x, time, elev (raw, no processing)

preprocess(time, elev, T_start, x, n_crop)
    Crop n_crop near-wall points from each end, subtract global mean, crop time -> t, fluct, x_ret

build_snapshot_matrix(data, case_keys)
    Stack fluctuation arrays from all cases column-wise -> X

compute_pod(X, r)
    SVD of X -> POD modes Phi_r, singular values s, cumulative energy

find_sensors_qr(Phi_r, x, x_exp, p)
    QR pivoting on experimental grid -> sensor indices gamma, sensor positions x_sensors

build_reconstruction_operator(Phi_r, gamma)
    Theta = Phi_r[gamma, :], R_rec = Phi_r @ pinv(Theta) -> R_rec, condition number kappa

reconstruct(R_rec, gamma, eta_true, noise_std)
    Extract sensor readings, optionally add noise, apply R_rec -> eta_rec

compute_error(eta_true, eta_rec)
    Relative Frobenius norm error between true and reconstructed field -> scalar

make_reconstruction_gif(...)
    Create and save a GIF comparing the true field with clean and noisy reconstructions over time
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd, qr, pinv
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import tqdm

# =============================================================================
#  STAGE 1 : LOAD AND PREPROCESS
# =============================================================================


def load_case(filepath):
    """
    Read one sloshing CSV file and return the raw data.

    The file format is:
      Row 1 : x spatial positions separated by ";"
      Rows 2-3 : skipped
      Row 4 : column headers
      Row 5+ : data (time, elevation values)

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    x    : numpy array, shape (n_space,)
        Spatial positions in metres.
    time : numpy array, shape (n_time,)
        Raw time vector in seconds.
    elev : numpy array, shape (n_time, n_space)
        Raw surface elevation in metres.
    """
    # read the first line to get the x positions
    with open(filepath) as f:
        x_line = f.readline()

    # split by ";" and skip the first two entries (labels), convert to float
    x_parts = x_line.strip().split(";")
    x = np.array([float(v) for v in x_parts[2:] if v.strip()])

    # read the rest of the file, skipping 3 header rows
    df = pd.read_csv(filepath, sep=";", skiprows=3, header=0)

    time = df["Time [s]"].values
    elev = df.iloc[:, 2:].values.astype(float)  # shape: (n_time, n_space)

    return x, time, elev


def preprocess(time, elev, T_start=0.0, x=None, n_crop=0):
    """
    Crop near-wall points, remove the mean elevation, and crop to the steady-state signal.

    Steps:
      1. Drop n_crop points from each spatial end.
         SPH near-wall results suffer from boundary errors; these points are excluded.
      2. Subtract the global mean of elev (one number for the whole array).
      3. Keep only rows where time >= T_start.
      4. Shift time so it starts at 0.

    Parameters
    ----------
    time    : numpy array, shape (n_time,)
    elev    : numpy array, shape (n_time, n_space)
    T_start : float   Time before which data is discarded. Default 0.0 s.
    x       : numpy array, shape (n_space,)   Spatial positions.
    n_crop  : int   Points to drop from each spatial end. Default 0.

    Returns
    -------
    t     : numpy array, shape (n_time_cropped,)
    fluct : numpy array, shape (n_time_cropped, n_space - 2*n_crop)
    x_ret : numpy array, shape (n_space - 2*n_crop,)
    """
    # step 1: drop near-wall points (SPH near-wall boundary error)
    if n_crop > 0:
        elev = elev[:, n_crop:-n_crop]
        if x is not None:
            x_ret = x[n_crop:-n_crop]

    # step 2: subtract global mean
    fluct = elev - elev.mean()

    # steps 3 and 4: crop and shift time
    mask = time >= T_start
    t = time[mask] - T_start
    fluct = fluct[mask]

    return t, fluct, x_ret


# =============================================================================
#  STAGE 2 : POD AND SENSOR PLACEMENT
# =============================================================================


def build_snapshot_matrix(data_dict, case_keys):
    """
    Stack all preprocessed fluctuation arrays into one big matrix X.

    Each case contributes a block of shape (n_space, n_time).
    They are stacked side by side: X has shape (n_space, n_time_total).

    Note: fluct arrays in the npz file are stored as (n_time, n_space),
    so we transpose each one before stacking.

    Parameters
    ----------
    data_dict      : dict-like (numpy npz file loaded with np.load)
        Contains arrays named like "0.5Hz", "0.6Hz", etc.
    case_keys : list of str
        Keys like ["0.5Hz", "0.6Hz", ...] in the order to stack.

    Returns
    -------
    X : numpy array, shape (n_space, n_total_snapshots)
        The snapshot matrix.
    """
    blocks = []
    for key in case_keys:
        fluct = data_dict[key]  # shape: (n_time, n_space)
        blocks.append(fluct.T)  # transpose to (n_space, n_time)

    X = np.hstack(blocks)  # stack side by side: (n_space, n_total)
    return X


def compute_pod(X, r):
    """
    Compute the POD modes of the snapshot matrix using SVD.

    The SVD gives:  X = U @ diag(s) @ Vt
    The columns of U are the spatial POD modes.

    Parameters
    ----------
    X : numpy array, shape (n_space, n_snapshots)
        Snapshot matrix.


    Returns
    -------
    U  : numpy array, shape (n_space, rank(X))
        The POD modes (columns).
    s      : numpy array, shape (n_snapshots(X),)
        All singular values in decreasing order.
    energy : numpy array, shape (n_snapshots,)
        Cumulative energy fraction (0 to 1) for each mode.

        energy[r-1] tells you how much energy r modes capture.
        rank(X)=min(n_space, n_snapshots)
    """
    # economy SVD: only compute as many columns of U as rank(X)
    U, s, _ = svd(X, full_matrices=False)

    # cumulative energy: how much of the total variance is captured
    energy = np.cumsum(s**2) / np.sum(s**2)

    return U, s, energy


def find_sensors_qr(Phi_r, x, x_exp, p, plot_it=False):
    """
    Select p sensor positions from the experimental grid using QR pivoting.

    Steps:
      1. For each point in x_exp, find the nearest point in x (simulation grid).
         This gives Phi_r_exp: the POD modes evaluated at experimental positions.
      2. Do QR with column pivoting on Phi_r_exp (or its outer product if p > r).
         The pivot order tells us which positions are most "important".
      3. Take the first p pivots as the chosen experimental positions.
      4. Map those back to indices in the full simulation grid x.

    Parameters
    ----------
    Phi_r : numpy array, shape (n_space, r)
        POD modes on the full simulation grid.
    x     : numpy array, shape (n_space,)
        Full simulation spatial grid in metres.
    x_exp : numpy array, shape (n_exp,)
        Discrete experimental grid (allowed sensor positions) in metres.
    p     : int
        Number of sensors to place. Must satisfy p >= r.
    plot_it : bool, optional
        If True, plot the POD modes on the experimental grid together with
        the original modes on the simulation grid. Default is False.

    Returns
    -------
    gamma     : numpy array, shape (p,)
        Indices into x (the full simulation grid) of the chosen sensor positions.
    x_sensors : numpy array, shape (p,)
        x-coordinates of the chosen sensors in metres.
    """
    r = Phi_r.shape[1]
    # step 1: find nearest simulation grid index for each experimental position
    exp_grid_idx = np.array([np.argmin(np.abs(x - xi)) for xi in x_exp])

    # Phi_r evaluated only at the experimental grid points
    Phi_r_exp = Phi_r[exp_grid_idx, :]  # shape: (n_exp, r)

    if plot_it:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i in range(r):
            color = colors[i % len(colors)]
            ax.plot(
                x_exp, Phi_r_exp[:, i], lw=1.5, color=color, label=f"Exp mode {i+1}"
            )
            ax.plot(x, Phi_r[:, i], "--", lw=1.5, color=color, label=f"Num mode {i+1}")
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Mode amplitude")
        ax.set_title(f"POD modes on experimental grid")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        # plt.savefig(
        #     "Test.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        plt.show()
        plt.close()

    # step 2: QR with column pivoting
    # when p == r: pivot on Phi_r_exp.T (shape r x n_exp)
    # when p > r:  pivot on the outer product (shape n_exp x n_exp)
    if p == r:
        _, _, piv = qr(Phi_r_exp.T, pivoting=True)
    else:
        _, _, piv = qr(Phi_r_exp @ Phi_r_exp.T, pivoting=True)

    # step 3: take first p pivots -> chosen experimental positions
    chosen_exp_positions = x_exp[piv[:p]]

    # step 4: map back to the full simulation grid
    gamma = np.array([np.argmin(np.abs(x - xi)) for xi in chosen_exp_positions])
    x_sensors = x[gamma]

    return gamma, x_sensors


def build_reconstruction_operator(Phi_r, gamma):
    """
    Build the linear reconstruction operator R_rec.

    The idea:
      - Theta = Phi_r[gamma, :] is the (p x r) matrix of modes at sensor positions.
      - If we know the sensor measurements y = Theta @ a (where a are POD coefficients),
        we can recover a = pinv(Theta) @ y, then reconstruct the full field as
        eta_rec = Phi_r @ a = Phi_r @ pinv(Theta) @ y = R_rec @ y.
      - R_rec has shape (n_space, p): it maps p sensor readings to the full field.

    Also computes the condition number of Theta (kappa).
    A high kappa means the sensor placement is poorly conditioned -> noisy reconstruction.

    Parameters
    ----------
    Phi_r : numpy array, shape (n_space, r)
        POD modes.
    gamma : numpy array, shape (p,)
        Sensor indices into the spatial grid.

    Returns
    -------
    R_rec : numpy array, shape (n_space, p)
        The reconstruction operator.
    kappa : float
        Condition number of Theta = Phi_r[gamma, :].
    """
    Theta = Phi_r[gamma, :]  # shape: (p, r)
    R_rec = Phi_r @ pinv(Theta)  # shape: (n_space, p)

    # condition number: ratio of largest to smallest singular value of Theta
    s_theta = svd(Theta, compute_uv=False)  # singular values only
    kappa = s_theta[0] / s_theta[-1]

    return R_rec, kappa


# =============================================================================
#  STAGE 3 : RECONSTRUCTION AND EVALUATION
# =============================================================================


def reconstruct(R_rec, gamma, eta_true, noise_std=0.0):
    """
    Reconstruct the full field from sparse sensor readings.

    Steps:
      1. Extract sensor readings from the true field at positions gamma.
      2. Optionally add Gaussian noise to the sensor readings.
      3. Multiply by R_rec to get the reconstructed field.

    Parameters
    ----------
    R_rec     : numpy array, shape (n_space, p)
        Reconstruction operator.
    gamma     : numpy array, shape (p,)
        Sensor indices.
    eta_true  : numpy array, shape (n_space, n_time)
        True fluctuation field.
    noise_std : float
        Standard deviation of Gaussian noise added to sensor readings.
        Use 0.0 for a clean (noise-free) reconstruction.

    Returns
    -------
    eta_rec     : numpy array, shape (n_space, n_time)
        Reconstructed field.
    eta_sensors : numpy array, shape (p, n_time)
        Sensor readings used for reconstruction (with noise if noise_std > 0).
    """
    # step 1: extract sensor readings
    eta_sensors = eta_true[gamma, :]  # shape: (p, n_time)

    # step 2: add noise if requested
    if noise_std > 0.0:
        noise = np.random.randn(*eta_sensors.shape) * noise_std
        eta_sensors = eta_sensors + noise

    # step 3: reconstruct
    eta_rec = R_rec @ eta_sensors  # shape: (n_space, n_time)

    return eta_rec, eta_sensors


def compute_error(eta_true, eta_rec):
    """
    Compute the relative reconstruction error using the Frobenius norm.

    Error = ||eta_true - eta_rec||_F  /  ||eta_true||_F

    A value of 0 means perfect reconstruction.
    A value of 1 means the error is as large as the signal itself.

    Parameters
    ----------
    eta_true : numpy array, shape (n_space, n_time)
    eta_rec  : numpy array, shape (n_space, n_time)

    Returns
    -------
    error : float
        Relative error (dimensionless).
    """
    error = np.linalg.norm(eta_true - eta_rec, "fro") / np.linalg.norm(eta_true, "fro")
    return error


def make_reconstruction_gif(
    x,
    t,
    eta_true,
    eta_rec,
    eta_rec_noisy,
    x_sensors,
    out_path,
    r,
    p,
    GIF_FPS,
    GIF_MAX_FRAMES,
):
    """
    Create a GIF showing the true field and two reconstructions over time.

    The animation compares:
      1. The true free-surface fluctuation
      2. The clean reconstruction
      3. The noisy reconstruction

    Sensor positions are marked on the x-axis. Only the first GIF_MAX_FRAMES time steps are used.

    Parameters
    ----------
    x : numpy array, shape (n_space,)
        Spatial grid in metres.
    t : numpy array, shape (n_time,)
        Time vector in seconds.
    eta_true : numpy array, shape (n_space, n_time)
        True fluctuation field.
    eta_rec : numpy array, shape (n_space, n_time)
        Clean reconstructed field.
    eta_rec_noisy : numpy array, shape (n_space, n_time)
        Reconstructed field using noisy sensor readings.
    x_sensors : numpy array, shape (p,)
        Sensor positions in metres.
    out_path : str or pathlib.Path
        Output path for the GIF file.
    r : int
        Number of POD modes used.
    p : int
        Number of sensors used.
    GIF_FPS : int
        Frames per second of the saved GIF.
    GIF_MAX_FRAMES : int
        Maximum number of frames to include.

    Returns
    -------
    n_frames : int
        Number of frames saved in the GIF.
    """

    n_frames = min(GIF_MAX_FRAMES, len(t))

    yabs = np.max(np.abs(eta_true)) * 1.15
    ylim = (-yabs, yabs)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ylim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Elevation fluctuation [m]")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    for xs in x_sensors:
        ax.plot(xs, 0, "v", color="black", ms=6, zorder=5)

    (line_true,) = ax.plot(x, eta_true[:, 0], color="black", lw=2.0, label="true")
    (line_clean,) = ax.plot(
        x, eta_rec[:, 0], color="green", lw=1.5, ls="--", label="clean rec"
    )
    (line_noisy,) = ax.plot(
        x, eta_rec_noisy[:, 0], color="crimson", lw=1.5, ls=":", label="noisy rec"
    )

    ax.legend(fontsize=9, loc="upper right")
    title = ax.set_title(f"r={r}, p={p}  |  t = {t[0]:.3f} s", fontsize=10)
    fig.tight_layout()

    frames = []
    for i in tqdm(range(n_frames), desc="Rendering GIF frames"):
        line_true.set_ydata(eta_true[:, i])
        line_clean.set_ydata(eta_rec[:, i])
        line_noisy.set_ydata(eta_rec_noisy[:, i])
        title.set_text(f"r={r}, p={p}  |  t = {t[i]:.3f} s")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimsave(str(out_path), frames, fps=GIF_FPS, loop=0)
    return len(frames)
