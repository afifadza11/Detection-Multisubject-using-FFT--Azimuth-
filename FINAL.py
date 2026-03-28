import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =========================================================
# FILE
# =========================================================
fileName = ("isi dengan file anda")
# =========================================================
# RADAR CONFIG — dari mmWave Studio
# =========================================================
numRX         = 4
numTX         = 3
numADCSamples = 200
numADCBits    = 16
isReal        = 0

FS_ADC = 4e6
SLOPE  = 66.662e12
C      = 3e8

chirps_per_frame = numTX   # 3 TX per TDM set
FFTSize          = 256

MIN_RANGE_M  = 0.40
MAX_RANGE_M  = 2.00
ANGLE_MAX    = 60
ANGLES       = np.arange(-ANGLE_MAX, ANGLE_MAX + 1, 1)

# =========================================================
# PARAMETER — hasil diagnostik menunjukkan 1e-4 terbaik
# =========================================================
DIAG_LOAD    = 1e-4
TOP_K        = 2
GUARD_R      = 8
GUARD_A      = 10
USE_MTI      = False
USE_HANNING  = False
D_OVER_LAMBDA = 0.5

RX_ORDERS = [
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [1, 3, 0, 2],
]
SIGN_CANDIDATES  = [+1, -1]
ENFORCE_OPPOSITE = True
AMBIGUOUS_DEG    = 15


# =========================================================
# FUNGSI — identik dengan kode asli
# =========================================================
def read_raw_safe(fileName, numADCSamples=200, numRX=4,
                  numADCBits=16, isReal=0, diag=True):
    if not os.path.exists(fileName):
        raise FileNotFoundError(f"File tidak ditemukan: {fileName}")
    adcData = np.fromfile(fileName, dtype=np.int16)
    if diag:
        print("File:", fileName)
        print("[Info] int16 count =", adcData.size)
    if numADCBits != 16:
        l_max = 2 ** (numADCBits - 1) - 1
        adcData = adcData.copy()
        adcData[adcData > l_max] -= 2 ** numADCBits
    if isReal:
        raise NotImplementedError
    raw  = np.empty(adcData.size // 2, dtype=np.complex64)
    cidx = 0
    for i in range(0, adcData.size - 3, 4):
        raw[cidx]     = adcData[i]     + 1j * adcData[i + 2]
        raw[cidx + 1] = adcData[i + 1] + 1j * adcData[i + 3]
        cidx += 2
    raw = raw[:cidx]
    if diag:
        print("[Info] complex count =", raw.size)
    denom   = numRX * numADCSamples
    nChirps = raw.size // denom
    rem     = raw.size - nChirps * denom
    if diag:
        print("[Info] nChirps =", nChirps, "| remainder =", rem)
    raw        = raw[:nChirps * denom]
    Radar_data = raw.reshape(nChirps, numRX, numADCSamples)
    if diag:
        print("Radar_data shape =", Radar_data.shape)
    return Radar_data


def range_fft(tx_frame_rx_adc, fft_size=256, use_hanning=False):
    x = np.transpose(tx_frame_rx_adc, (0, 2, 1))
    if use_hanning:
        win = np.hanning(x.shape[1]).astype(np.float32)
        x   = x * win[None, :, None]
    return np.fft.fft(x, n=fft_size, axis=1)


def mti_2pulse(X):
    Y = np.zeros_like(X);  Y[1:] = X[1:] - X[:-1];  return Y


def build_range_axis(numADCSamples, FS_ADC, SLOPE, C, fft_size):
    B   = (numADCSamples / FS_ADC) * SLOPE
    dR  = C / (2 * B)
    rng = np.arange(fft_size) * dR
    return dR, rng


def align_tx_to_ref(RP_ref, RP_to_align, rb_cal):
    cross      = RP_to_align[:, rb_cal, :] * np.conj(RP_ref[:, rb_cal, :])
    phi_rx     = np.angle(np.mean(cross, axis=0))
    corr       = np.exp(-1j * phi_rx)
    RP_aligned = RP_to_align * corr[None, None, :]
    cross_after = RP_aligned[:, rb_cal, :] * np.conj(RP_ref[:, rb_cal, :])
    phi_after   = np.angle(np.mean(cross_after, axis=0))
    return RP_aligned, phi_rx, phi_after


def build_virtual12(RP0, RP1_aligned, RP2_aligned, order):
    return np.concatenate(
        [RP0[:, :, order], RP1_aligned[:, :, order], RP2_aligned[:, :, order]],
        axis=2,
    )


def steering_ula(theta_deg, M, d_over_lambda=0.5, sign=+1):
    th = np.deg2rad(theta_deg)
    m  = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d_over_lambda * m * np.sin(sign * th))


def mvdr_spectrum_for_rb(V, rb, angles_deg, diag_load=1e-4,
                          sign=+1, d_over_lambda=0.5):
    X    = V[:, rb, :].T
    M    = X.shape[0]
    R    = (X @ X.conj().T) / max(X.shape[1], 1)
    R   += diag_load * np.trace(R) / M * np.eye(M)
    Rinv = np.linalg.pinv(R)
    P    = np.zeros(len(angles_deg), dtype=float)
    for i, ang in enumerate(angles_deg):
        a    = steering_ula(ang, M, d_over_lambda=d_over_lambda,
                             sign=sign).reshape(M, 1)
        den  = (a.conj().T @ Rinv @ a).item()
        P[i] = 1.0 / (np.abs(den) + 1e-12)
    return P


def mvdr_range_angle_map(V, r_bins, angles_deg, diag_load=1e-4,
                          sign=+1, d_over_lambda=0.5):
    M  = V.shape[2]
    RA = np.zeros((len(r_bins), len(angles_deg)), dtype=float)
    for ri, rb in enumerate(r_bins):
        X    = V[:, rb, :].T
        R    = (X @ X.conj().T) / max(X.shape[1], 1)
        R   += diag_load * np.trace(R) / M * np.eye(M)
        Rinv = np.linalg.pinv(R)
        for ai, ang in enumerate(angles_deg):
            a          = steering_ula(ang, M, d_over_lambda=d_over_lambda,
                                       sign=sign).reshape(M, 1)
            den        = (a.conj().T @ Rinv @ a).item()
            RA[ri, ai] = 1.0 / (np.abs(den) + 1e-12)
    return RA


def normalize_RA_per_range(RA):
    return RA / (np.mean(RA, axis=1, keepdims=True) + 1e-12)


def parabolic_peak(y_arr, peak_idx):
    """Sub-bin refinement via 3-point parabola."""
    if peak_idx <= 0 or peak_idx >= len(y_arr) - 1:
        return 0.0
    y0 = float(y_arr[peak_idx - 1])
    y1 = float(y_arr[peak_idx])
    y2 = float(y_arr[peak_idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-30:
        return 0.0
    return (y0 - y2) / denom


def pick_topk_2d(RA, rng_bins, angles_deg, k=2,
                 guard_r=8, guard_a=10,
                 enforce_opposite=True, ambiguous_deg=15):
    angles_deg = np.array(angles_deg)
    rng_arr    = np.array(rng_bins, dtype=float)
    RA2        = RA.copy().astype(float)
    targets    = []

    for step in range(k):
        ri, ai   = np.unravel_index(np.argmax(RA2), RA2.shape)
        r_m      = float(rng_arr[ri])
        ang      = float(angles_deg[ai])
        peak_val = float(RA[ri, ai])

        # Parabolic sub-bin refinement
        dR_loc  = float(rng_arr[1] - rng_arr[0]) if len(rng_arr) > 1 else 0.0
        dr      = parabolic_peak(RA[:, ai], ri)
        r_ref   = float(np.clip(r_m + dr * dR_loc, rng_arr[0], rng_arr[-1]))
        da      = parabolic_peak(RA[ri, :], ai)
        ang_ref = float(np.clip(ang + da, angles_deg[0], angles_deg[-1]))

        targets.append((ri, ai, r_ref, ang_ref, peak_val))

        # Guard zone
        r0 = max(0, ri - guard_r);  r1 = min(RA2.shape[0], ri + guard_r + 1)
        a0 = max(0, ai - guard_a);  a1 = min(RA2.shape[1], ai + guard_a + 1)
        RA2[r0:r1, a0:a1] = -np.inf

        # Constraint berseberangan
        if step == 0 and enforce_opposite and abs(ang) > ambiguous_deg:
            same = (np.sign(angles_deg) == np.sign(ang))
            same[angles_deg == 0] = False
            RA2[:, same] = -np.inf
            print(f"  [Constraint] T1={ang_ref:+.2f}° → "
                  f"T2 hanya di sisi {'negatif' if ang > 0 else 'positif'}")

    return targets


# =========================================================
# VISUALISASI
# =========================================================
def plot_range_azimuth(RA_n, rng_sel, angles_deg, targets):
    """Range-azimuth map halus — seperti yang dosen tunjukkan."""
    RA_s = gaussian_filter(RA_n, sigma=[0.8, 1.2])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        RA_s, origin="lower", aspect="auto",
        extent=[angles_deg[0], angles_deg[-1], rng_sel[0], rng_sel[-1]],
        cmap="viridis", interpolation="bilinear",
    )
    colors_m = ["orange", "white", "cyan", "lime"]
    for idx, (_, _, r_m, ang, _) in enumerate(targets, start=1):
        c = colors_m[(idx-1) % len(colors_m)]
        ax.scatter(ang, r_m, s=120, marker="x", color=c, linewidths=2.5, zorder=5)
        ax.text(ang + 1.5, r_m + 0.02,
                f"{r_m:.2f}m, {ang:+.1f}°",
                fontsize=9, color=c, fontweight="bold")
    plt.colorbar(im, ax=ax, label="MVDR power (norm)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Range-Azimuth Map (MVDR) | 3TX × 4RX = 12 virtual channel")
    plt.tight_layout();  plt.show()


def plot_sector(RA_n, rng_sel, angles_deg, targets, angle_max=60):
    """Sector polar view."""
    angles_deg = np.array(angles_deg)
    RA_s = gaussian_filter(RA_n, sigma=[0.8, 1.2])
    fig, ax = plt.subplots(figsize=(9, 8),
                            subplot_kw=dict(projection="polar"))
    az_rad = np.deg2rad(angles_deg)
    theta  = np.pi / 2 - az_rad
    T, R   = np.meshgrid(theta, rng_sel)
    pcm    = ax.pcolormesh(T, R, RA_s, cmap="viridis", shading="auto")
    plt.colorbar(pcm, ax=ax, label="MVDR Power (norm)", pad=0.1, shrink=0.7)
    ax.set_thetamin(90 - angle_max);  ax.set_thetamax(90 + angle_max)
    ax.set_theta_zero_location("N");  ax.set_theta_direction(-1)
    colors_m = ["orange", "white", "cyan", "lime"]
    for idx, (_, _, r_m, ang, _) in enumerate(targets, start=1):
        c    = colors_m[(idx-1) % len(colors_m)]
        th_t = np.pi / 2 - np.deg2rad(ang)
        ax.scatter(th_t, r_m, s=250, marker="*", color=c, zorder=10,
                   edgecolors="black", lw=1,
                   label=f"T{idx}: {r_m:.2f}m {ang:+.1f}°")
        ax.annotate(f"T{idx}\n{r_m:.2f}m\n{ang:+.1f}°",
                    xy=(th_t, r_m), xytext=(th_t + 0.1, r_m + 0.12),
                    fontsize=9, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=9)
    ax.set_title("Range-Azimuth Sector (MVDR)", pad=20)
    plt.tight_layout();  plt.show()


def plot_spectrum(V_best, rb, rng_val, angles, diag_load,
                  sign_best, d, label=""):
    """Spektrum azimuth raw — bentuk sinusoidal seperti yang dosen minta."""
    P = mvdr_spectrum_for_rb(V_best, rb, angles,
                              diag_load=diag_load,
                              sign=sign_best,
                              d_over_lambda=d)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(angles, P, linewidth=1.8, color="steelblue")
    ax.set_title(f"MVDR azimuth spectrum @ range {rng_val:.2f} m {label}")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("MVDR Power")
    ax.grid(True, alpha=0.4)
    plt.tight_layout();  plt.show()


# =========================================================
# MAIN
# =========================================================
print("=" * 60)
print("FMCW Radar — IWR6843 ISK ODS")
print("=" * 60)

Radar_data = read_raw_safe(
    fileName, numADCSamples=numADCSamples,
    numRX=numRX, numADCBits=numADCBits, isReal=isReal, diag=True,
)

total_chirps = Radar_data.shape[0]
n_frames     = total_chirps // chirps_per_frame
Radar_use    = Radar_data[: n_frames * chirps_per_frame]
X_raw        = Radar_use.reshape(n_frames, chirps_per_frame, numRX, numADCSamples)
tx0 = X_raw[:, 0, :, :]
tx1 = X_raw[:, 1, :, :]
tx2 = X_raw[:, 2, :, :]
print(f"n_frames = {n_frames}")

# ---------- Range FFT ----------
RP0 = range_fft(tx0, FFTSize, use_hanning=USE_HANNING)
RP1 = range_fft(tx1, FFTSize, use_hanning=USE_HANNING)
RP2 = range_fft(tx2, FFTSize, use_hanning=USE_HANNING)

dR, rng = build_range_axis(numADCSamples, FS_ADC, SLOPE, C, FFTSize)
min_bin = int(np.searchsorted(rng, MIN_RANGE_M))
max_bin = min(int(np.searchsorted(rng, MAX_RANGE_M)), FFTSize - 1)
print(f"dR={dR:.4f}m | bins {min_bin}({rng[min_bin]:.3f}m)..{max_bin}({rng[max_bin]:.3f}m)")

# ---------- Range energy ----------
E0 = np.sum(np.abs(RP0[:, min_bin:max_bin+1, :]), axis=(0, 2))
rb_cal = min_bin + int(np.argmax(E0))
print(f"rb_cal = {rb_cal} ({rng[rb_cal]:.3f}m)")

plt.figure(figsize=(10, 4))
plt.plot(rng[min_bin:max_bin+1], E0, lw=1.5)
plt.title("Range Energy (TX0)");  plt.xlabel("Range (m)");  plt.grid(True)
plt.tight_layout();  plt.show()

# ---------- Phase alignment ----------
RP1_aligned, phi1_before, phi1_after = align_tx_to_ref(RP0, RP1, rb_cal)
RP2_aligned, phi2_before, phi2_after = align_tx_to_ref(RP0, RP2, rb_cal)
print(f"phi1_before: {np.round(phi1_before, 3)}")
print(f"phi1_after : {np.round(phi1_after, 3)}")

if USE_MTI:
    RP0_use = mti_2pulse(RP0)
    RP1_use = mti_2pulse(RP1_aligned)
    RP2_use = mti_2pulse(RP2_aligned)
else:
    RP0_use, RP1_use, RP2_use = RP0, RP1_aligned, RP2_aligned

# ---------- Candidate RB ----------
cand_rb = [rb_cal]
for t in np.argsort(-E0)[:5]:
    cand_rb.append(min_bin + int(t))
cand_rb = sorted(list(set(
    [rb for rb in cand_rb if min_bin <= rb <= max_bin]
)))
print(f"cand_rb: {[f'{rng[rb]:.3f}m' for rb in cand_rb]}")

# ---------- Cari order + sign terbaik ----------
best = None
for order in RX_ORDERS:
    V = build_virtual12(RP0_use, RP1_use, RP2_use, order)
    for sign in SIGN_CANDIDATES:
        score = 0.0
        for rb in cand_rb:
            P        = mvdr_spectrum_for_rb(V, rb, ANGLES,
                                            diag_load=DIAG_LOAD,
                                            sign=sign,
                                            d_over_lambda=D_OVER_LAMBDA)
            peak_i   = int(np.argmax(P))
            score   += float(P[peak_i]) - 0.05 * abs(float(ANGLES[peak_i]))
        if (best is None) or (score > best["score"]):
            best = {"order": order, "sign": sign, "score": score}

print(f"Best: order={best['order']}  sign={best['sign']:+d}  score={best['score']:.3f}")

V_best    = build_virtual12(RP0_use, RP1_use, RP2_use, best["order"])
sign_best = best["sign"]

# ---------- MVDR range-angle map ----------
r_bins  = np.arange(min_bin, max_bin + 1)
rng_sel = rng[r_bins]

print("Menghitung MVDR range-angle map...")
RA   = mvdr_range_angle_map(V_best, r_bins, ANGLES,
                             diag_load=DIAG_LOAD,
                             sign=sign_best,
                             d_over_lambda=D_OVER_LAMBDA)
RA_n = normalize_RA_per_range(RA)

# ---------- Pick target ----------
print("\nPeak picking:")
targets = pick_topk_2d(
    RA_n, rng_sel, ANGLES,
    k=TOP_K, guard_r=GUARD_R, guard_a=GUARD_A,
    enforce_opposite=ENFORCE_OPPOSITE,
    ambiguous_deg=AMBIGUOUS_DEG,
)

print(f"\n{'='*60}")
print(f"Deteksi: {len(targets)} target")
print(f"{'='*60}")
for i, (_, _, r_m, ang, pwr) in enumerate(targets, start=1):
    x_m = r_m * np.sin(np.deg2rad(ang))
    y_m = r_m * np.cos(np.deg2rad(ang))
    print(f"Target {i}: range={r_m:.3f}m | azimuth={ang:+.2f}° | "
          f"x={x_m:+.3f}m | y={y_m:.3f}m | score={pwr:.4f}")

# ---------- Plot ----------
plot_range_azimuth(RA_n, rng_sel, ANGLES, targets)
plot_sector(RA_n, rng_sel, ANGLES, targets, angle_max=ANGLE_MAX)
for i, (ri, _, r_m, ang, _) in enumerate(targets, start=1):
    plot_spectrum(V_best, r_bins[ri], rng[r_bins[ri]], ANGLES,
                  DIAG_LOAD, sign_best, D_OVER_LAMBDA, label=f"(Target {i})")