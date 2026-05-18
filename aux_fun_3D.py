"""
aux_fun_3D.py
-------------
Helper functions for the 3D sloshing free-surface analysis.

Functions
---------
parse_case_info(stem)
    Extract case number, descriptive label, and primary forcing frequency from a filename.

load_case(filepath)
    Read one 3D CSV file -> px, py, time, elev (raw, no processing)

preprocess(time, elev, px, py, T_start, x_borders, y_borders)
    Drop domain border columns, crop time, shift to zero, subtract global mean
    -> t, fluct, x_ret, y_ret

build_snapshot_matrix(data_dict, case_keys)
    Stack fluctuation arrays from all cases column-wise -> X

compute_pod(X, r)
    SVD of X -> POD modes U, singular values s, cumulative energy

find_sensors_qr(Phi_r, x, y, x_exp, y_exp, p)
    QR pivoting on 2D experimental grid -> sensor indices gamma, sensor positions xy_sensors

build_reconstruction_operator(Phi_r, gamma)
    Theta = Phi_r[gamma, :], R_rec = Phi_r @ pinv(Theta) -> R_rec, condition number kappa

reconstruct(R_rec, gamma, eta_true, noise_std)
    Extract sensor readings, optionally add noise, apply R_rec -> eta_rec, eta_sensors

compute_error(eta_true, eta_rec)
    Relative Frobenius norm error between true and reconstructed field -> scalar

make_visualisation_gif(...)
    Create and save a single-panel GIF of the free-surface fluctuation over time

make_reconstruction_gif(...)
    Create and save a 3-panel GIF (True | Clean Rec | Noisy Rec) over time
"""

import math
import re

import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3D projection
from scipy.linalg import pinv, qr, svd
from tqdm.auto import tqdm

# =============================================================================
#  STAGE 1 : LOAD AND PREPROCESS
# =============================================================================


def parse_case_info(stem):
    """
    Extract case number, descriptive label, and primary forcing frequency from a filename.

    Handles single-direction (X or Y) and combined (X + Y) cases.

    Parameters
    ----------
    stem : str
        Filename stem, e.g. "CaseSloshing_01_X_0p6Hz_40mm_out_...".

    Returns
    -------
    case_num     : int    Case number.
    label        : str    Readable label, e.g. "Case01 X:0.6Hz".
    """
    m = re.search(r"CaseSloshing_(\d+)", stem)
    case_num = int(m.group(1))

    pairs = re.findall(r"([XY])_(\d+(?:p\d+)?)Hz", stem)
    freq_parts = [(d, float(f.replace("p", "."))) for d, f in pairs]

    label = f"Case{case_num:02d} " + " ".join(f"{d}:{f}Hz" for d, f in freq_parts)
    key = "_".join(
        f"{d}{f:.1f}Hz" for d, f in freq_parts
    )  # e.g. "X0.6Hz" or "X1.2Hz_Y1.6Hz"

    return case_num, label, key


def load_case(filepath):
    """
    Read one 3D sloshing CSV file and return the raw data.

    The file format is:
      Row 1 : PosX [m] for each spatial column, separated by ";"
      Row 2 : PosY [m] for each spatial column, separated by ";"
      Row 3 : PosZ [m] (initial surface, ignored)
      Row 4 : column headers: Part ; Time [s] ; Elevation_0 [m] ; ...
      Row 5+ : data rows

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    px   : numpy array, shape (n_space,)   Raw x-coordinates.
    py   : numpy array, shape (n_space,)   Raw y-coordinates.
    time : numpy array, shape (n_time,)    Raw time vector in seconds.
    elev : numpy array, shape (n_time, n_space)  Raw surface elevation in metres.
    """
    with open(filepath) as f:
        x_line = f.readline()
        y_line = f.readline()

    px = np.array([float(v) for v in x_line.strip().split(";")[2:]])
    py = np.array([float(v) for v in y_line.strip().split(";")[2:]])

    df = pd.read_csv(filepath, sep=";", skiprows=3, header=0)
    time = df["Time [s]"].values
    elev = df.iloc[:, 2:].values.astype(float)  # (n_time, n_space)

    return px, py, time, elev


def preprocess(
    time, elev, px, py, T_start=0.1, x_borders=(-0.225, 0.225), y_borders=(-0.15, 0.15)
):
    """
    Drop domain border columns, crop to steady-state, and subtract the global mean.

    Steps:
      1. Drop columns whose x or y coordinate is on the domain boundary.
      2. Keep only rows where time >= T_start; shift time to start at 0.
      3. Subtract the global mean (one scalar for the whole cropped array, same as 2D).

    Parameters
    ----------
    time      : numpy array, shape (n_time,)
    elev      : numpy array, shape (n_time, n_space)
    px        : numpy array, shape (n_space,)   x-coordinate for each column.
    py        : numpy array, shape (n_space,)   y-coordinate for each column.
    T_start   : float   Crop rows with time < T_start (transient phase). Default 0.1 s.
    x_borders : tuple   (x_min, x_max) boundary x-values to drop. Default (-0.225, 0.225).
    y_borders : tuple   (y_min, y_max) boundary y-values to drop. Default (-0.15, 0.15).

    Returns
    -------
    t     : numpy array, shape (n_time_crop,)
    fluct : numpy array, shape (n_time_crop, n_kept)   Global mean removed.
    x_ret : numpy array, shape (n_kept,)
    y_ret : numpy array, shape (n_kept,)
    """
    is_x_border = np.isclose(px, x_borders[0]) | np.isclose(px, x_borders[1])
    is_y_border = np.isclose(py, y_borders[0]) | np.isclose(py, y_borders[1])
    keep = ~(is_x_border | is_y_border)

    elev_kept = elev[:, keep]
    x_ret = px[keep]
    y_ret = py[keep]

    mask = time >= T_start
    t = time[mask] - T_start
    elev_crop = elev_kept[mask]

    # global mean removal
    fluct = elev_crop - elev_crop.mean()

    return t, fluct, x_ret, y_ret


# =============================================================================
#  STAGE 2 : POD AND SENSOR PLACEMENT
# =============================================================================


def build_snapshot_matrix(data_dict, case_keys):
    """
    Stack all preprocessed fluctuation arrays into one big matrix X.

    Each case contributes a block of shape (n_space, n_time).
    They are stacked side by side: X has shape (n_space, n_time_total).

    Parameters
    ----------
    data_dict : dict-like (numpy npz or plain dict)
        Contains arrays named like "X0.6Hz", "X1.2Hz_Y1.6Hz", etc.
    case_keys : list of str
        Keys in the order to stack.

    Returns
    -------
    X : numpy array, shape (n_space, n_total_snapshots)
        The snapshot matrix.
    """
    blocks = []
    for key in case_keys:
        fluct = data_dict[key]  # (n_time, n_space)
        blocks.append(fluct.T)  # (n_space, n_time)

    X = np.hstack(blocks)
    return X


def compute_pod(X, r):
    """
    Compute the POD modes of the snapshot matrix using SVD.

    Identical to the 2D version.

    Parameters
    ----------
    X : numpy array, shape (n_space, n_snapshots)

    Returns
    -------
    U      : numpy array, shape (n_space, rank(X))   POD modes (columns).
    s      : numpy array, shape (rank(X),)            Singular values in decreasing order.
    energy : numpy array, shape (rank(X),)            Cumulative energy fraction (0 to 1).
    """
    U, s, _ = svd(X, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    return U, s, energy


def find_sensors_qr(Phi_r, x, y, x_exp, y_exp, p):
    """
    Select p sensor positions from a 2D experimental grid using QR pivoting.

    Steps:
      1. Build the Cartesian product of x_exp x y_exp to get all experimental positions.
      2. For each experimental point, find the nearest simulation grid point (2D distance).
         This gives Phi_r_exp: POD modes at the experimental grid.
      3. QR with column pivoting on Phi_r_exp (or its outer product if p > r).
      4. Map the first p pivots back to indices in the full simulation grid.

    Parameters
    ----------
    Phi_r : numpy array, shape (n_space, r)
    x     : numpy array, shape (n_space,)   Flattened x-coordinates of simulation grid.
    y     : numpy array, shape (n_space,)   Flattened y-coordinates of simulation grid.
    x_exp : numpy array, shape (n_xexp,)   Allowed x sensor positions.
    y_exp : numpy array, shape (n_yexp,)   Allowed y sensor positions.
    p     : int   Number of sensors. Must satisfy p >= r.

    Returns
    -------
    gamma      : numpy array, shape (p,)    Indices into simulation grid.
    xy_sensors : numpy array, shape (p, 2)  (x, y) coordinates of chosen sensors.
    """
    r_modes = Phi_r.shape[1]

    # Cartesian product of experimental grids
    xx_exp, yy_exp = np.meshgrid(x_exp, y_exp)
    xy_exp_pts = np.column_stack([xx_exp.ravel(), yy_exp.ravel()])  # (n_exp, 2)

    # nearest simulation grid point for each experimental position (Euclidean distance)
    xy_sim = np.column_stack([x, y])  # (n_space, 2)
    exp_grid_idx = np.array(
        [np.argmin(((xy_sim - pt) ** 2).sum(axis=1)) for pt in xy_exp_pts]
    )

    Phi_r_exp = Phi_r[exp_grid_idx, :]  # (n_exp, r)

    # QR with column pivoting - same logic as 2D
    if p == r_modes:
        _, _, piv = qr(Phi_r_exp.T, pivoting=True)
    else:
        _, _, piv = qr(Phi_r_exp @ Phi_r_exp.T, pivoting=True)

    exp_sensor_idx = piv[:p]
    xy_sensors = xy_exp_pts[exp_sensor_idx]  # (p, 2)

    gamma = np.array([np.argmin(((xy_sim - pt) ** 2).sum(axis=1)) for pt in xy_sensors])

    return gamma, xy_sensors


def build_reconstruction_operator(Phi_r, gamma):
    """
    Build the linear reconstruction operator R_rec.

    Identical to the 2D version.

    Parameters
    ----------
    Phi_r : numpy array, shape (n_space, r)
    gamma : numpy array, shape (p,)

    Returns
    -------
    R_rec : numpy array, shape (n_space, p)
    kappa : float   Condition number of Theta = Phi_r[gamma, :].
    """
    Theta = Phi_r[gamma, :]  # (p, r)
    R_rec = Phi_r @ pinv(Theta)  # (n_space, p)

    s_theta = svd(Theta, compute_uv=False)
    kappa = s_theta[0] / s_theta[-1]

    return R_rec, kappa


# =============================================================================
#  STAGE 3 : RECONSTRUCTION AND EVALUATION
# =============================================================================


def reconstruct(R_rec, gamma, eta_true, noise_std=0.0):
    """
    Reconstruct the full field from sparse sensor readings.

    Identical to the 2D version.

    Parameters
    ----------
    R_rec     : numpy array, shape (n_space, p)
    gamma     : numpy array, shape (p,)
    eta_true  : numpy array, shape (n_space, n_time)
    noise_std : float   Std of Gaussian noise added to sensor readings.

    Returns
    -------
    eta_rec     : numpy array, shape (n_space, n_time)
    eta_sensors : numpy array, shape (p, n_time)
    """
    eta_sensors = eta_true[gamma, :]  # (p, n_time)

    if noise_std > 0.0:
        noise = np.random.randn(*eta_sensors.shape) * noise_std
        eta_sensors = eta_sensors + noise

    eta_rec = R_rec @ eta_sensors  # (n_space, n_time)

    return eta_rec, eta_sensors


def compute_error(eta_true, eta_rec):
    """
    Relative Frobenius norm error.

    Identical to the 2D version.

    Parameters
    ----------
    eta_true : numpy array, shape (n_space, n_time)
    eta_rec  : numpy array, shape (n_space, n_time)

    Returns
    -------
    error : float
    """
    return np.linalg.norm(eta_true - eta_rec, "fro") / np.linalg.norm(eta_true, "fro")


def make_reconstruction_gif(
    x,
    y,
    t,
    eta_true,
    eta_rec,
    eta_rec_noisy,
    xy_sensors,
    out_path,
    r,
    p,
    GIF_FPS,
    GIF_MAX_FRAMES,
    z_lim=None,
    elev_cam=30,
    azim=-60,
    colormap="coolwarm",
):
    """
    Create a 3-panel GIF: True | Clean Rec | Noisy Rec over time.

    Sensor positions are marked on the Noisy Rec panel.

    Parameters
    ----------
    x, y         : numpy arrays, shape (n_space,)   Flat simulation grid coordinates.
    t            : numpy array,  shape (n_time,)    Time vector.
    eta_true     : numpy array,  shape (n_space, n_time)
    eta_rec      : numpy array,  shape (n_space, n_time)   Clean reconstruction.
    eta_rec_noisy: numpy array,  shape (n_space, n_time)   Noisy reconstruction.
    xy_sensors   : numpy array,  shape (p, 2)       Sensor (x, y) positions.
    out_path     : str or Path   Output GIF path.
    r, p         : int           Shown in title.
    GIF_FPS      : int           Frames per second.
    GIF_MAX_FRAMES : int         Maximum frames to render.
    z_lim        : float or None Auto-computed from eta_true if None.
    elev_cam     : float         3D view elevation angle [deg].
    azim         : float         3D view azimuth angle [deg].
    colormap     : str           Matplotlib colormap name.

    Returns
    -------
    n_frames : int
    """
    n_frames = min(GIF_MAX_FRAMES, len(t))

    if z_lim is None:
        _max = float(np.max(np.abs(eta_true)))
        _mag = math.floor(math.log10(_max))
        z_lim = math.ceil(_max / 10**_mag) * 10**_mag

    cmap_obj = matplotlib.colormaps.get_cmap(colormap)
    norm_obj = mcolors.Normalize(vmin=-z_lim, vmax=z_lim)

    x_s, y_s = xy_sensors[:, 0], xy_sensors[:, 1]
    n_sensors = len(x_s)

    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(top=0.85)
    ax_true = fig.add_subplot(131, projection="3d")
    ax_clean = fig.add_subplot(132, projection="3d")
    ax_noisy = fig.add_subplot(133, projection="3d")

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=[ax_true, ax_clean, ax_noisy],
        shrink=0.6,
        pad=0.05,
        label="Fluctuation [m]",
    )

    sup = fig.suptitle("", fontsize=10)

    frames = []
    for i in tqdm(range(n_frames), desc="Rendering GIF frames"):
        z_true = eta_true[:, i]
        z_clean = eta_rec[:, i]
        z_noisy = eta_rec_noisy[:, i]

        for ax, z_vals, subtitle in zip(
            [ax_true, ax_clean, ax_noisy],
            [z_true, z_clean, z_noisy],
            ["True", "Clean rec", "Noisy rec"],
        ):
            ax.cla()
            ax.scatter(
                x,
                y,
                z_vals,
                c=cmap_obj(norm_obj(z_vals)),
                s=15,
                depthshade=False,
                linewidths=0,
            )
            ax.set_box_aspect([3, 2, 1])
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_zlim(-z_lim, z_lim)
            ax.set_xlabel("x [m]", labelpad=4)
            ax.set_ylabel("y [m]", labelpad=4)
            ax.set_zlabel("elev [m]", labelpad=4)
            ax.set_title(subtitle, fontsize=9)
            ax.view_init(elev=elev_cam, azim=azim)

        # sensor markers on the noisy rec panel only
        ax_noisy.scatter(
            x_s, y_s, np.zeros(n_sensors), c="black", s=40, zorder=10, depthshade=False
        )
        for j in range(n_sensors):
            ax_noisy.text(x_s[j], y_s[j], 0, f" {j+1}", fontsize=7, color="black")

        sup.set_text(f"r={r}, p={p}  |  t = {t[i]:.3f} s")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimsave(str(out_path), frames, fps=GIF_FPS, loop=0)
    return len(frames)


def make_visualisation_gif(
    x,
    y,
    t,
    fluct,
    out_path,
    GIF_FPS,
    GIF_MAX_FRAMES,
    z_lim=None,
    elev_cam=30,
    azim=-60,
    colormap="coolwarm",
):
    """
    Create a single-panel GIF of the 3D free-surface fluctuation over time.

    Parameters
    ----------
    x, y   : numpy arrays, shape (n_space,)        Flat simulation grid coordinates.
    t      : numpy array,  shape (n_time,)          Time vector.
    fluct  : numpy array,  shape (n_time, n_space)  Fluctuation field.
    out_path : str or Path   Output GIF path.
    GIF_FPS        : int    Frames per second.
    GIF_MAX_FRAMES : int    Maximum frames to render.
    z_lim    : float or None  Auto-computed from fluct if None.
    elev_cam : float   3D view elevation angle [deg].
    azim     : float   3D view azimuth angle [deg].
    colormap : str     Matplotlib colormap name.

    Returns
    -------
    n_frames : int
    """
    n_frames = min(GIF_MAX_FRAMES, len(t))

    if z_lim is None:
        _max = float(np.max(np.abs(fluct)))
        _mag = math.floor(math.log10(_max))
        z_lim = math.ceil(_max / 10**_mag) * 10**_mag

    cmap_obj = matplotlib.colormaps.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=-z_lim, vmax=z_lim)

    fig = plt.figure(figsize=(9, 7))
    ax3d = fig.add_subplot(111, projection="3d")

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.12, label="Fluctuation [m]")

    frames = []
    for i in tqdm(range(n_frames), desc="Rendering visualisation GIF"):
        ax3d.cla()
        z_vals = fluct[i]
        ax3d.scatter(
            x,
            y,
            z_vals,
            c=cmap_obj(norm(z_vals)),
            s=15,
            depthshade=False,
            linewidths=0,
        )
        ax3d.set_box_aspect([3, 2, 1])
        ax3d.set_xlim(x.min(), x.max())
        ax3d.set_ylim(y.min(), y.max())
        ax3d.set_zlim(-z_lim, z_lim)
        ax3d.set_xlabel("x [m]", labelpad=4)
        ax3d.set_ylabel("y [m]", labelpad=4)
        ax3d.set_zlabel("elev [m]", labelpad=4)
        ax3d.set_title(f"t = {t[i]:.3f} s", fontsize=11)
        ax3d.view_init(elev=elev_cam, azim=azim)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimsave(str(out_path), frames, fps=GIF_FPS, loop=0)
    return len(frames)
