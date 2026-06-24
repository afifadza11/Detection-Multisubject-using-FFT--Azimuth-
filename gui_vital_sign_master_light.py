"""
Vital Sign Semi Real-Time Monitor - MASTER EXACT V8 - 60s HYBRID BPM
Radar FMCW - Multi-Subject Monitoring

Input  : file radar .bin saja
Output : HR Radar, RR Radar, posisi Subject 1 & Subject 2, dan sinyal HR/RR berjalan

Catatan penting:
- Nilai HR/RR dihitung dari data radar, bukan dari ground truth.
- Mode GUI semi real-time: dataset diproses dulu, lalu hasilnya diputar seperti monitoring berjalan.
- Pipeline dibuat mengikuti notebook master: DCA1000 read -> TX0/TX2 -> Range FFT -> MVDR -> beamforming -> DACM -> filter -> BPM.
"""

from __future__ import annotations

import math
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter tidak tersedia. Install Python desktop standar, bukan minimal build.") from exc

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from scipy.ndimage import gaussian_filter1d, gaussian_filter, maximum_filter
    from scipy.signal import find_peaks, butter, sosfiltfilt, detrend as sp_detrend
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# Konfigurasi utama, dibuat sama dengan notebook master
# ============================================================

C = 3e8
NUM_TX = 3
NUM_RX = 4
NUM_ADC_SAMPLES = 200
NUM_ADC_BITS = 16
FS_ADC = 4e6
SLOPE_DEFAULT = 70.175e12
FFT_SIZE = 1024
FRAME_PERIOD_S = 0.05
FS_VITAL = 20.0

MIN_RANGE_M = 0.40
MAX_RANGE_M = 2.50
ANGLE_MAX = 60
ANGLES = np.arange(-ANGLE_MAX, ANGLE_MAX + 1, 1, dtype=float)
D_OVER_LAMBDA = 0.5
VIRT_POS_LAMBDA = np.arange(8, dtype=float) * D_OVER_LAMBDA

RANGE_WINDOW = "blackman"
REMOVE_ADC_MEAN = True
RANGE_SMOOTH_SIGMA = 1.2
RANGE_MIN_SEP_M = 0.15
RANGE_PROM_FRAC = 0.08
NUM_RANGE_CANDS = 8
RANGE_NEIGHBOR_M = 0.10

DIAG_LOAD_FRAC = 5e-3
COV_HALFSPAN = 2
SNAPSHOT_STEP = 3  # lebih ringan, tetap stabil untuk demo semi real-time

RR_BAND_HZ = (0.10, 0.50)   # 6-30 breaths/min
HR_BAND_HZ = (0.80, 2.00)   # 48-120 bpm
FILTER_ORDER = 4

RX_ORDERS = [
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [1, 3, 0, 2],
]
SIGN_CANDIDATES = [+1, -1]
ALIGN_MODES = ["aligned", "raw"]

# Parameter selection sama dengan notebook master
MIN_SIDE_DEG = 20.0
TOP_K = 2
STRICT_OPPOSITE = True
CONFIG_EVAL_SIGMA = 1.5
LOCALMAX_WIN = 11
LOCALMAX_KEEP = 100
LOCALMAX_THR = 0.10
ROW_PROM_FRAC = 0.10
ROW_MAX_PEAKS = 5
ROW_MIN_SEP = 5
OPP_MIN_REL_SCORE = 0.15
OPP_MAX_RANGE_GAP_M = 0.70
REFINE_ANGLE_HALF = 4
REFINE_RANGE_HALF = 3
MVDR_USE_FB_AVG = False
MVDR_SNAPSHOT_NORMALIZE = False


# ============================================================
# Data container
# ============================================================

@dataclass
class Target:
    subject_id: int
    range_m: float
    angle_deg: float
    range_bin: int
    score: float
    selection_mode: str = "MVDR"


@dataclass
class SubjectResult:
    target: Target
    t: np.ndarray
    y_complex: np.ndarray
    phi_raw: np.ndarray
    phi_rr: np.ndarray
    phi_hr: np.ndarray
    rr_display: np.ndarray
    hr_display: np.ndarray
    bpm_times: np.ndarray
    rr_bpm: np.ndarray
    hr_bpm: np.ndarray
    rr_full: float
    hr_full: float
    rr_3seg: float
    hr_3seg: float
    rr_display_bpm: float
    hr_display_bpm: float


# ============================================================
# Helper umum
# ============================================================

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x)
    x = x - np.mean(x)
    mx = np.max(np.abs(x)) if len(x) else 0.0
    if not np.isfinite(mx) or mx <= 1e-12:
        return x
    return x / mx


def normalize_visible_window(x: np.ndarray, signal_type: str = "hr") -> np.ndarray:
    """
    Normalisasi khusus untuk tampilan window grafik.

    Fungsi ini hanya untuk visualisasi GUI, bukan untuk menghitung nilai HR/RR.
    Pada RR, sinyal dibuat sedikit lebih halus agar pola respirasi lebih mudah dibaca
    pada window 60 detik. Nilai BPM tetap memakai data hasil filter asli.
    """
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x)
    if len(x) == 0:
        return x

    # Khusus visual RR: haluskan sedikit karena respirasi adalah komponen frekuensi rendah.
    # Ini tidak dipakai untuk estimasi BPM.
    if signal_type.lower() == "rr" and len(x) >= 7:
        if SCIPY_OK:
            x = gaussian_filter1d(x, sigma=2.0)
        else:
            kernel = np.ones(7, dtype=float) / 7.0
            x = np.convolve(x, kernel, mode="same")

    x = x - np.median(x)
    absx = np.abs(x)

    # Percentile 85 lebih tahan terhadap spike besar di akhir window,
    # sehingga bagian sinyal yang amplitudonya kecil tidak tampak lurus.
    q = 85 if signal_type.lower() == "rr" else 90
    scale = np.percentile(absx, q) if len(absx) >= 20 else np.max(absx)
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = np.max(absx)
    if (not np.isfinite(scale)) or scale <= 1e-12:
        return x

    y = x / scale
    return np.clip(y, -1.15, 1.15)


def infer_slope(path_str: str) -> float:
    n = Path(str(path_str)).name.lower()
    if "adc_3ghz_" in n:
        return 70e12
    if "adc_2_5ghz_" in n:
        return 50e12
    if "adc_2ghz_" in n:
        return 40e12
    return SLOPE_DEFAULT


def make_window(n: int, kind: str = "blackman") -> np.ndarray:
    kind = kind.lower()
    if kind in ("none", "rect", "rectangular"):
        return np.ones(n, dtype=np.float32)
    if kind in ("hann", "hanning"):
        return np.hanning(n).astype(np.float32)
    if kind == "blackman":
        return np.blackman(n).astype(np.float32)
    return np.blackman(n).astype(np.float32)


# ============================================================
# Processing radar dari notebook master, dibuat ringkas untuk GUI
# ============================================================

class RadarVitalProcessor:
    def __init__(self):
        self.num_tx = NUM_TX
        self.num_rx = NUM_RX
        self.num_adc_samples = NUM_ADC_SAMPLES
        self.fft_size = FFT_SIZE
        self.fs_vital = FS_VITAL
        self.frame_period_s = FRAME_PERIOD_S

    # ---------- Read DCA1000 ----------
    def compensate_adc_bits(self, raw16: np.ndarray) -> np.ndarray:
        raw16 = np.asarray(raw16, dtype=np.int16).copy()
        if NUM_ADC_BITS != 16:
            l_max = 2 ** (NUM_ADC_BITS - 1) - 1
            raw16[raw16 > l_max] = raw16[raw16 > l_max] - 2 ** NUM_ADC_BITS
        return raw16

    def raw16_to_complex_dca1000(self, raw16: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pola master: I0, I1, Q0, Q1, lalu berulang.
        """
        usable = (raw16.size // 4) * 4
        x = raw16[:usable].reshape(-1, 4)
        cplx = np.empty(x.shape[0] * 2, dtype=np.complex64)
        cplx[0::2] = x[:, 0].astype(np.float32) + 1j * x[:, 2].astype(np.float32)
        cplx[1::2] = x[:, 1].astype(np.float32) + 1j * x[:, 3].astype(np.float32)
        return cplx, usable

    def read_raw(self, file_path: str) -> Tuple[np.ndarray, int]:
        raw16 = np.fromfile(file_path, dtype=np.int16)
        raw16 = self.compensate_adc_bits(raw16)
        if raw16.size == 0:
            raise ValueError("File .bin kosong atau tidak bisa dibaca.")

        cplx, _ = self.raw16_to_complex_dca1000(raw16)
        complex_per_chirp = self.num_rx * self.num_adc_samples
        n_chirps_total = cplx.size // complex_per_chirp
        n_frames = n_chirps_total // self.num_tx
        usable_chirps = n_frames * self.num_tx
        usable_complex = usable_chirps * complex_per_chirp
        cplx = cplx[:usable_complex]

        if n_frames < 40:
            raise ValueError(
                f"Frame terlalu sedikit ({n_frames}). Pastikan file .bin sesuai format DCA1000/TDM-MIMO."
            )

        radar_std = cplx.reshape(usable_chirps, self.num_rx, self.num_adc_samples)
        x4d = radar_std.reshape(n_frames, self.num_tx, self.num_rx, self.num_adc_samples)
        return x4d.astype(np.complex64, copy=False), n_frames

    # ---------- Range FFT ----------
    def apply_mti(self, x: np.ndarray, mode: str = "none") -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)
        if mode == "none":
            return x.copy()
        if mode == "mean":
            return (x - np.mean(x, axis=0, keepdims=True)).astype(np.complex64)
        if mode == "diff1":
            y = np.zeros_like(x, dtype=np.complex64)
            y[1:] = x[1:] - x[:-1]
            return y
        return x.copy()

    def range_axis(self, slope: float) -> np.ndarray:
        n = self.fft_size // 2
        fb = np.arange(n, dtype=np.float64) * (FS_ADC / self.fft_size)
        return C * fb / (2.0 * slope)

    def range_fft(self, tx_data: np.ndarray) -> np.ndarray:
        # input: frame x RX x ADC sample, output: frame x range bin x RX
        x = np.transpose(tx_data, (0, 2, 1)).astype(np.complex64, copy=False)
        if REMOVE_ADC_MEAN:
            x = (x - np.mean(x, axis=1, keepdims=True)).astype(np.complex64, copy=False)
        win = make_window(x.shape[1], RANGE_WINDOW)
        x = (x * win[None, :, None]).astype(np.complex64, copy=False)
        rp = np.fft.fft(x, n=self.fft_size, axis=1)[:, :self.fft_size // 2, :]
        return rp.astype(np.complex64, copy=False)

    def range_profile(self, rps: List[np.ndarray], min_bin: int, max_bin: int) -> Tuple[np.ndarray, np.ndarray]:
        p = None
        for rp in rps:
            cur = np.sum(np.abs(rp[:, min_bin:max_bin + 1, :]) ** 2, axis=(0, 2))
            p = cur if p is None else p + cur
        pdb = 10.0 * np.log10(p + 1e-12)
        if SCIPY_OK:
            pdb_s = gaussian_filter1d(pdb, sigma=RANGE_SMOOTH_SIGMA)
        else:
            pdb_s = pdb
        return pdb, pdb_s

    def subbin_parabolic(self, y: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(y) - 1:
            return 0.0
        y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
        d = 2.0 * (2.0 * y1 - y0 - y2)
        return 0.0 if abs(d) < 1e-30 else float(np.clip((y0 - y2) / d, -0.5, 0.5))

    def find_range_peaks(self, pdb_s: np.ndarray, rng_sel: np.ndarray) -> List[Tuple[int, float, float, float]]:
        d_r = (rng_sel[1] - rng_sel[0]) if len(rng_sel) > 1 else 0.05
        min_dist = max(1, int(round(RANGE_MIN_SEP_M / max(d_r, 1e-9))))
        dyn = float(np.max(pdb_s) - np.min(pdb_s))
        prom = max(0.2, RANGE_PROM_FRAC * dyn)

        if SCIPY_OK:
            peaks, props = find_peaks(pdb_s, distance=min_dist, prominence=prom)
        else:
            peaks = np.array([int(np.argmax(pdb_s))], dtype=int)
            props = {"prominences": np.array([1.0])}

        if len(peaks) == 0:
            pk = int(np.argmax(pdb_s))
            return [(pk, float(rng_sel[pk]), 0.0, float(pdb_s[pk]))]

        order = np.argsort(props["prominences"])[::-1]
        out = []
        for oi in order[:NUM_RANGE_CANDS]:
            pk = int(peaks[oi])
            delta = self.subbin_parabolic(pdb_s, pk)
            r_ref = float(np.clip(rng_sel[pk] + delta * d_r, rng_sel[0], rng_sel[-1]))
            out.append((pk, r_ref, delta, float(pdb_s[pk])))
        return out

    def neighbor_bins(self, range_peaks, rng, min_bin, max_bin) -> List[int]:
        d_r = float(rng[1] - rng[0]) if len(rng) > 1 else 0.01
        span = max(0, int(round(RANGE_NEIGHBOR_M / max(d_r, 1e-9))))
        bins = set()
        for pk_idx, _, _, _ in range_peaks:
            c = min_bin + int(pk_idx)
            for d in range(-span, span + 1):
                rb = c + d
                if min_bin <= rb <= max_bin:
                    bins.add(rb)
        return sorted(bins)

    # ---------- Virtual array dan MVDR ----------
    def calibrate_tx2(self, rp_ref, rp_src, rb_cal, halfspan=4):
        rb0 = max(0, rb_cal - halfspan)
        rb1 = min(rp_ref.shape[1] - 1, rb_cal + halfspan)
        cross = np.zeros(rp_ref.shape[2], dtype=np.complex128)
        total_w = 0.0
        for rb in range(rb0, rb1 + 1):
            w = float(np.mean(np.abs(rp_ref[:, rb, :]) ** 2 + np.abs(rp_src[:, rb, :]) ** 2))
            cross += w * np.mean(rp_src[:, rb, :] * np.conj(rp_ref[:, rb, :]), axis=0)
            total_w += w
        if total_w > 0:
            cross /= total_w
        phi = np.angle(cross)
        phase_corr = np.exp(-1j * phi).astype(np.complex64)
        out = (rp_src.astype(np.complex64, copy=False) * phase_corr[None, None, :]).astype(np.complex64)
        return out, phi

    def build_virtual(self, rp0: np.ndarray, rp2: np.ndarray, order: List[int]) -> np.ndarray:
        o = list(order)
        return np.concatenate([rp0[:, :, o], rp2[:, :, o]], axis=2).astype(np.complex64, copy=False)

    def steering_array(self, angles_deg: np.ndarray, sign: int = +1) -> np.ndarray:
        """
        Steering vector dibuat sama dengan notebook master:
        exp(-j 2π pos_lambda sin(sign * theta))
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        theta = np.deg2rad(angles_deg)
        phase = -1j * 2.0 * np.pi * VIRT_POS_LAMBDA[:, None] * np.sin(float(sign) * theta[None, :])
        return np.exp(phase).astype(np.complex128)

    def covariance_for_rb(self, v: np.ndarray, rb: int, halfspan: int = COV_HALFSPAN) -> np.ndarray:
        """Covariance matrix sama dengan notebook master, tanpa subsampling snapshot."""
        rb = int(rb)
        rb0 = max(0, rb - int(halfspan))
        rb1 = min(v.shape[1], rb + int(halfspan) + 1)

        x = np.transpose(v[:, rb0:rb1, :], (2, 0, 1)).reshape(v.shape[2], -1).astype(np.complex128)

        if MVDR_SNAPSHOT_NORMALIZE:
            norm = np.sqrt(np.sum(np.abs(x) ** 2, axis=0, keepdims=True))
            x = x / np.maximum(norm, 1e-12)

        r = (x @ x.conj().T) / max(x.shape[1], 1)
        r = 0.5 * (r + r.conj().T)

        if MVDR_USE_FB_AVG:
            j = np.fliplr(np.eye(r.shape[0]))
            r = 0.5 * (r + j @ r.conj() @ j)
            r = 0.5 * (r + r.conj().T)

        m = r.shape[0]
        trace_mean = float(np.trace(r).real) / max(m, 1)
        if (not np.isfinite(trace_mean)) or trace_mean <= 0:
            trace_mean = 1.0
        r = r + (DIAG_LOAD_FRAC * trace_mean) * np.eye(m)
        return r

    def covariance_for_rb_parts(self, rp0, rp2, order, rb, halfspan=COV_HALFSPAN):
        """Versi memory-safe dari notebook master untuk memilih konfigurasi virtual array."""
        rb = int(rb)
        rb0 = max(0, rb - int(halfspan))
        rb1 = min(rp0.shape[1], rb + int(halfspan) + 1)
        o = list(order)

        x0 = rp0[:, rb0:rb1, :][:, :, o].astype(np.complex64, copy=False)
        x2 = rp2[:, rb0:rb1, :][:, :, o].astype(np.complex64, copy=False)
        xv = np.concatenate([x0, x2], axis=2)
        m = xv.shape[2]
        x = np.transpose(xv, (2, 0, 1)).reshape(m, -1).astype(np.complex128, copy=False)

        if MVDR_SNAPSHOT_NORMALIZE:
            norm = np.sqrt(np.sum(np.abs(x) ** 2, axis=0, keepdims=True))
            x = x / np.maximum(norm, 1e-12)

        r = (x @ x.conj().T) / max(x.shape[1], 1)
        r = 0.5 * (r + r.conj().T)

        if MVDR_USE_FB_AVG:
            j = np.fliplr(np.eye(m))
            r = 0.5 * (r + j @ r.conj() @ j)
            r = 0.5 * (r + r.conj().T)

        trace_mean = float(np.trace(r).real) / max(m, 1)
        if (not np.isfinite(trace_mean)) or trace_mean <= 0:
            trace_mean = 1.0
        r = r + (DIAG_LOAD_FRAC * trace_mean) * np.eye(m)
        return r

    def mvdr_spectrum(self, v: np.ndarray, rb: int, angles: np.ndarray, sign: int = +1) -> np.ndarray:
        r = self.covariance_for_rb(v, rb)
        a = self.steering_array(angles, sign=sign)
        try:
            ria = np.linalg.solve(r, a)
        except np.linalg.LinAlgError:
            ria = np.linalg.pinv(r) @ a
        den = np.real(np.sum(a.conj() * ria, axis=0))
        p = 1.0 / np.maximum(den, 1e-30)
        return p.astype(float)

    def mvdr_spectrum_from_parts(self, rp0, rp2, order, rb, angles_deg, sign):
        r = self.covariance_for_rb_parts(rp0, rp2, order, rb)
        a = self.steering_array(angles_deg, sign=sign)
        try:
            ria = np.linalg.solve(r, a)
        except np.linalg.LinAlgError:
            ria = np.linalg.pinv(r) @ a
        den = np.real(np.sum(a.conj() * ria, axis=0))
        p = 1.0 / np.maximum(den, 1e-30)
        return p.astype(float)

    def score_config_from_parts(self, rp0, rp2, order, eval_rbs, angles_deg, sign):
        """Scoring konfigurasi dibuat sama dengan notebook master."""
        right_mask = angles_deg >= +MIN_SIDE_DEG
        left_mask = angles_deg <= -MIN_SIDE_DEG
        total = 0.0
        n_bilateral = 0

        for rb in eval_rbs:
            sp = self.mvdr_spectrum_from_parts(rp0, rp2, order, rb, angles_deg, sign)
            sp = sp / (np.mean(sp) + 1e-12)
            sp_s = gaussian_filter1d(sp, sigma=CONFIG_EVAL_SIGMA) if SCIPY_OK else sp

            r_max = float(np.max(sp_s[right_mask]))
            l_max = float(np.max(sp_s[left_mask]))
            bilateral_min = min(r_max, l_max)
            bilateral_max = max(r_max, l_max) + 1e-9

            total += r_max + l_max
            total += 3.0 * bilateral_min
            imbalance = abs(r_max - l_max) / bilateral_max
            total -= 1.5 * imbalance * bilateral_max
            total -= 0.01 * abs(float(angles_deg[int(np.argmax(sp_s))]))

            if bilateral_min > 0.25 * bilateral_max:
                n_bilateral += 1

        total += 3.0 * n_bilateral
        return float(total)

    def score_config(self, rp0, rp2_raw, rp2_al, seed_rbs) -> Dict:
        """Pilih RX order, sign, dan alignment sama seperti notebook master."""
        best = None
        eval_rbs = seed_rbs[: min(len(seed_rbs), 20)]

        for align in ALIGN_MODES:
            rp2 = rp2_al if align == "aligned" else rp2_raw
            for order in RX_ORDERS:
                for sign in SIGN_CANDIDATES:
                    sc = self.score_config_from_parts(rp0, rp2, order, eval_rbs, ANGLES, sign)
                    if best is None or sc > best["score"]:
                        best = {"order": order, "sign": sign, "align": align, "score": sc}

        return best or {"order": [0, 1, 2, 3], "sign": +1, "align": "aligned", "score": 0.0}

    def mvdr_map(self, v: np.ndarray, r_bins: np.ndarray, angles: np.ndarray, sign: int) -> np.ndarray:
        """Return RA mentah. Normalisasi dan smoothing dilakukan setelahnya, seperti notebook."""
        ra = np.zeros((len(angles), len(r_bins)), dtype=float)
        for j, rb in enumerate(r_bins):
            ra[:, j] = self.mvdr_spectrum(v, int(rb), angles, sign=sign)
        return ra

    def find_2d_peaks(self, ra_s: np.ndarray):
        if SCIPY_OK:
            mx = maximum_filter(ra_s, size=LOCALMAX_WIN, mode="nearest")
            mask = (ra_s == mx) & (ra_s >= LOCALMAX_THR * np.max(ra_s))
            coords = np.argwhere(mask)
        else:
            coords = np.empty((0, 2), dtype=int)

        peaks = []
        for ai, ri in coords:
            peaks.append((int(ai), int(ri), float(ra_s[int(ai), int(ri)])))
        peaks.sort(key=lambda z: z[2], reverse=True)
        return peaks[:LOCALMAX_KEEP]

    def find_row_peaks(self, ra_s: np.ndarray, cand_ris: List[int]):
        cands = []
        for ri in cand_ris:
            ri = int(np.clip(ri, 0, ra_s.shape[1] - 1))
            row = ra_s[:, ri]
            dyn = float(np.max(row) - np.min(row))
            prom = max(1e-6, ROW_PROM_FRAC * dyn)
            if SCIPY_OK:
                pks, _ = find_peaks(row, prominence=prom, distance=max(3, ROW_MIN_SEP))
            else:
                pks = np.array([int(np.argmax(row))])
            if len(pks) == 0:
                pks = np.array([int(np.argmax(row))])
            for pk in np.argsort(row[pks])[::-1][:ROW_MAX_PEAKS]:
                ai = int(pks[pk])
                cands.append((ai, int(ri), float(row[ai])))
        return cands

    def dedup_candidates(self, cands, min_dang=3.0, min_dr=0.05):
        out = []
        for c in sorted(cands, key=lambda z: z["score"], reverse=True):
            ok = all(abs(c["ang"] - d["ang"]) > min_dang or abs(c["r_m"] - d["r_m"]) > min_dr for d in out)
            if ok:
                out.append(c)
        return out

    def build_candidates_master(self, peaks2d, row_peaks, rng_sel, angles_deg):
        raw = []
        for ai, ri, val in list(peaks2d) + list(row_peaks):
            ang = float(angles_deg[int(ai)])
            r_m = float(rng_sel[int(ri)])
            side = "center" if abs(ang) <= MIN_SIDE_DEG else ("right" if ang > 0 else "left")
            raw.append({"ai": int(ai), "ri": int(ri), "ang": ang, "r_m": r_m, "score": float(val), "side": side})
        return self.dedup_candidates(raw)

    def refine_candidate(self, ra_s, rng_sel, angles_deg, cand):
        ai0, ri0 = int(cand["ai"]), int(cand["ri"])
        a0 = max(0, ai0 - REFINE_ANGLE_HALF)
        a1 = min(ra_s.shape[0] - 1, ai0 + REFINE_ANGLE_HALF)
        r0 = max(0, ri0 - REFINE_RANGE_HALF)
        r1 = min(ra_s.shape[1] - 1, ri0 + REFINE_RANGE_HALF)
        patch = ra_s[a0:a1 + 1, r0:r1 + 1]
        loc = np.unravel_index(np.argmax(patch), patch.shape)
        ai = a0 + int(loc[0])
        ri = r0 + int(loc[1])
        da = self.subbin_parabolic(ra_s[:, ri], ai)
        dr = self.subbin_parabolic(ra_s[ai, :], ri)
        d_r = float(rng_sel[1] - rng_sel[0]) if len(rng_sel) > 1 else 0.0
        ang_ref = float(np.clip(angles_deg[ai] + da, angles_deg[0], angles_deg[-1]))
        r_ref = float(np.clip(rng_sel[ri] + dr * d_r, rng_sel[0], rng_sel[-1]))
        out = dict(cand)
        out.update({"ai": ai, "ri": ri, "ang": ang_ref, "r_m": r_ref, "score": float(ra_s[ai, ri])})
        return out

    def select_targets_master(self, cands, rng_sel, angles_deg, ra_s, top_k=TOP_K, strict_opp=STRICT_OPPOSITE) -> List[Target]:
        """Seleksi target dibuat sama dengan notebook master."""
        if not cands:
            raise RuntimeError("Target tidak ditemukan dari peta MVDR.")

        cands_s = sorted(cands, key=lambda z: z["score"], reverse=True)
        strongest = cands_s[0]["score"]

        selected_dicts = []
        if top_k == 2 and strict_opp:
            right_cands = [c for c in cands_s if c["ang"] >= +MIN_SIDE_DEG]
            left_cands = [c for c in cands_s if c["ang"] <= -MIN_SIDE_DEG]

            if right_cands and left_cands:
                right_valid = [c for c in right_cands if c["score"] >= OPP_MIN_REL_SCORE * strongest] or right_cands[:1]
                left_valid = [c for c in left_cands if c["score"] >= OPP_MIN_REL_SCORE * strongest] or left_cands[:1]
                best_pair = None
                best_pair_sc = -1e18

                for rc in right_valid[:5]:
                    for lc in left_valid[:5]:
                        r_gap = abs(rc["r_m"] - lc["r_m"])
                        if r_gap > OPP_MAX_RANGE_GAP_M:
                            continue
                        sc = rc["score"] + lc["score"] + 0.20 * np.exp(-r_gap / 0.30)
                        if sc > best_pair_sc:
                            best_pair_sc = sc
                            best_pair = [dict(rc), dict(lc)]

                if best_pair is not None:
                    pair = sorted(best_pair, key=lambda z: z["ang"], reverse=True)
                    for idx, pp in enumerate(pair, 1):
                        pp["subject_id"] = idx
                        pp.setdefault("selection_mode", "best_per_side")
                    selected_dicts = [self.refine_candidate(ra_s, rng_sel, angles_deg, pp) for pp in pair]

        if not selected_dicts:
            chosen = []
            for c in cands_s:
                if all(abs(c["ang"] - d["ang"]) >= 4.0 or abs(c["r_m"] - d["r_m"]) >= 0.05 for d in chosen):
                    cc = dict(c)
                    cc["selection_mode"] = "fallback"
                    chosen.append(cc)
                if len(chosen) >= top_k:
                    break
            for idx, cc in enumerate(chosen, 1):
                cc["subject_id"] = idx
            selected_dicts = [self.refine_candidate(ra_s, rng_sel, angles_deg, cc) for cc in chosen]

        targets: List[Target] = []
        for i, d in enumerate(selected_dicts[:2], 1):
            sid = int(d.get("subject_id", i))
            ri = int(d["ri"])
            targets.append(Target(
                subject_id=sid,
                range_m=float(d["r_m"]),
                angle_deg=float(d["ang"]),
                range_bin=0,  # diisi setelah return dari process
                score=float(d["score"]),
                selection_mode=str(d.get("selection_mode", "MVDR")),
            ))
            targets[-1]._ri = ri  # atribut internal sementara

        if len(targets) < 2:
            first = targets[0]
            targets.append(Target(2, first.range_m, first.angle_deg + 20.0, first.range_bin, first.score * 0.5, "Fallback"))
            targets[-1]._ri = getattr(first, "_ri", 0)

        return targets

    # ---------- Beamforming, DACM, Filter, BPM ----------

    def beamform_target_signal(self, v_phase: np.ndarray, target: Target, sign: int, range_avg_halfspan: int = 1) -> np.ndarray:
        rb = int(target.range_bin)
        theta = float(target.angle_deg)
        rb0 = max(0, rb - range_avg_halfspan)
        rb1 = min(v_phase.shape[1], rb + range_avg_halfspan + 1)
        xk = np.mean(v_phase[:, rb0:rb1, :], axis=1).astype(np.complex128)
        a = self.steering_array(np.array([theta]), sign=sign)[:, 0]
        yk = (xk @ a.conj()) / xk.shape[1]
        return yk

    def dacm_phase(self, y_complex: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        y_complex = np.asarray(y_complex, dtype=np.complex128)
        i = np.real(y_complex)
        q = np.imag(y_complex)
        di = np.zeros_like(i, dtype=float)
        dq = np.zeros_like(q, dtype=float)
        di[1:] = i[1:] - i[:-1]
        dq[1:] = q[1:] - q[:-1]
        dphi = (i * dq - di * q) / (i ** 2 + q ** 2 + eps)
        dphi[0] = 0.0
        return np.cumsum(dphi)

    def preprocess_phase(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        if SCIPY_OK and len(phi) > 3:
            phi_d = sp_detrend(phi, type="linear")
        else:
            n = len(phi)
            tt = np.arange(n)
            if n > 2:
                coef = np.polyfit(tt, phi, 1)
                phi_d = phi - (coef[0] * tt + coef[1])
            else:
                phi_d = phi - np.mean(phi)
        return phi_d - np.mean(phi_d)

    def bandpass_zero_phase(self, x: np.ndarray, band_hz: Tuple[float, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x)
        if len(x) < 16:
            return x - np.mean(x)
        if SCIPY_OK:
            sos = butter(FILTER_ORDER, band_hz, btype="bandpass", fs=FS_VITAL, output="sos")
            return sosfiltfilt(sos, x)
        # fallback FFT mask
        y = x - np.mean(x)
        freqs = np.fft.rfftfreq(len(y), d=1 / FS_VITAL)
        yy = np.fft.rfft(y)
        mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        yy[~mask] = 0
        return np.fft.irfft(yy, n=len(y))

    def estimate_rr_frequency_domain(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        n = len(x)
        if n < 8 or np.std(x) <= 1e-12:
            return np.nan, np.nan
        window = np.hanning(n)
        n_fft = int(2 ** np.ceil(np.log2(max(n * 4, 16))))
        freqs = np.fft.rfftfreq(n_fft, d=1 / FS_VITAL)
        spectrum = np.abs(np.fft.rfft(x * window, n=n_fft))
        mask = (freqs >= RR_BAND_HZ[0]) & (freqs <= RR_BAND_HZ[1])
        if not np.any(mask):
            return np.nan, np.nan
        freqs_band = freqs[mask]
        spectrum_band = spectrum[mask]
        f_peak = float(freqs_band[int(np.argmax(spectrum_band))])
        rr_value = float(f_peak * 60.0)
        return f_peak, rr_value

    def estimate_hr_time_domain(self, x: np.ndarray, prominence_factor: float = 0.25) -> Tuple[float, int, float]:
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        if len(x) < 8 or np.std(x) == 0:
            return np.nan, 0, np.nan

        min_distance_samples = max(1, int((1 / HR_BAND_HZ[1]) * FS_VITAL))
        prominence_value = prominence_factor * np.std(x)
        if SCIPY_OK:
            peaks, _ = find_peaks(x, distance=min_distance_samples, prominence=prominence_value)
        else:
            peaks = []
            last = -999999
            for i in range(1, len(x) - 1):
                if i - last < min_distance_samples:
                    continue
                if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > prominence_value:
                    peaks.append(i); last = i
            peaks = np.array(peaks, dtype=int)

        intervals_s = np.diff(peaks) / FS_VITAL
        min_period_s = 1 / HR_BAND_HZ[1]
        max_period_s = 1 / HR_BAND_HZ[0]
        valid_intervals = intervals_s[(intervals_s >= min_period_s) & (intervals_s <= max_period_s)]

        if len(valid_intervals) == 0:
            return np.nan, len(peaks), np.nan
        t_median = float(np.median(valid_intervals))
        hr_value = float(60.0 / t_median)
        return hr_value, len(peaks), t_median

    def rolling_bpm(self, phi_rr: np.ndarray, phi_hr: np.ndarray, window_sec: float = 30.0, step_sec: float = 1.0):
        n = len(phi_rr)
        win = int(window_sec * FS_VITAL)
        step = int(step_sec * FS_VITAL)
        times, rr_vals, hr_vals = [], [], []
        if n < win:
            _, rr = self.estimate_rr_frequency_domain(phi_rr)
            hr, _, _ = self.estimate_hr_time_domain(phi_hr)
            return np.array([n / FS_VITAL]), np.array([rr]), np.array([hr])

        for end in range(win, n + 1, step):
            start = end - win
            _, rr = self.estimate_rr_frequency_domain(phi_rr[start:end])
            hr, _, _ = self.estimate_hr_time_domain(phi_hr[start:end])
            times.append(end / FS_VITAL)
            rr_vals.append(rr)
            hr_vals.append(hr)
        return np.array(times), np.array(rr_vals), np.array(hr_vals)

    def estimate_3segment_mean(self, phi_rr: np.ndarray, phi_hr: np.ndarray) -> Tuple[float, float]:
        """
        Menghitung rerata 3 segmen seperti tabel notebook master.
        Data 180 detik dibagi menjadi 3 bagian: 0-60, 60-120, 120-180 detik.
        RR tiap segmen dihitung dengan FFT, HR tiap segmen dengan peak detection.
        """
        n = min(len(phi_rr), len(phi_hr))
        if n < int(10 * FS_VITAL):
            _, rr_val = self.estimate_rr_frequency_domain(phi_rr)
            hr_val, _, _ = self.estimate_hr_time_domain(phi_hr)
            return rr_val, hr_val

        rr_vals = []
        hr_vals = []
        edges = np.linspace(0, n, 4, dtype=int)

        for i in range(3):
            a, b = int(edges[i]), int(edges[i + 1])
            if b - a < int(10 * FS_VITAL):
                continue
            _, rr_i = self.estimate_rr_frequency_domain(phi_rr[a:b])
            hr_i, _, _ = self.estimate_hr_time_domain(phi_hr[a:b])
            if np.isfinite(rr_i):
                rr_vals.append(float(rr_i))
            if np.isfinite(hr_i):
                hr_vals.append(float(hr_i))

        rr_3seg = float(np.mean(rr_vals)) if len(rr_vals) else np.nan
        hr_3seg = float(np.mean(hr_vals)) if len(hr_vals) else np.nan
        return rr_3seg, hr_3seg

    def process_subject(self, v_phase: np.ndarray, target: Target, sign: int) -> SubjectResult:
        y = self.beamform_target_signal(v_phase, target, sign=sign, range_avg_halfspan=1)
        phi_raw = self.dacm_phase(y)
        phi_centered = self.preprocess_phase(phi_raw)
        phi_rr = self.bandpass_zero_phase(phi_centered, RR_BAND_HZ)
        phi_hr = self.bandpass_zero_phase(phi_centered, HR_BAND_HZ)

        # Hasil full 180 detik
        _, rr_full = self.estimate_rr_frequency_domain(phi_rr)
        hr_full, _, _ = self.estimate_hr_time_domain(phi_hr)

        # Hasil rerata 3 segmen, mengikuti tabel notebook master
        rr_3seg, hr_3seg = self.estimate_3segment_mean(phi_rr, phi_hr)

        # Aturan display untuk GUI sesuai keputusan dari tabel evaluasi:
        # RR memakai Full 180 detik, HR memakai 3 Segment Mean.
        rr_display_bpm = rr_full if np.isfinite(rr_full) else rr_3seg
        hr_display_bpm = hr_3seg if np.isfinite(hr_3seg) else hr_full

        bpm_times, rr_bpm, hr_bpm = self.rolling_bpm(phi_rr, phi_hr, window_sec=30.0, step_sec=1.0)
        t = np.arange(len(phi_raw)) / FS_VITAL
        return SubjectResult(
            target=target,
            t=t,
            y_complex=y,
            phi_raw=phi_raw,
            phi_rr=phi_rr,
            phi_hr=phi_hr,
            rr_display=normalize_for_display(phi_rr),
            hr_display=normalize_for_display(phi_hr),
            bpm_times=bpm_times,
            rr_bpm=rr_bpm,
            hr_bpm=hr_bpm,
            rr_full=rr_full,
            hr_full=hr_full,
            rr_3seg=rr_3seg,
            hr_3seg=hr_3seg,
            rr_display_bpm=rr_display_bpm,
            hr_display_bpm=hr_display_bpm,
        )

    def process(self, file_path: str) -> Dict:
        file_path = str(file_path)
        slope = infer_slope(file_path)
        rng = self.range_axis(slope)
        min_bin = int(np.searchsorted(rng, MIN_RANGE_M))
        max_bin = min(int(np.searchsorted(rng, MAX_RANGE_M)), len(rng) - 1)

        x4d, n_frames = self.read_raw(file_path)
        tx0_data = x4d[:, 0, :, :]
        tx2_data = x4d[:, 2, :, :]

        # Notebook master memakai MTI_MODE = none untuk menjaga energi target.
        rp0 = self.range_fft(self.apply_mti(tx0_data, mode="none"))
        rp2 = self.range_fft(self.apply_mti(tx2_data, mode="none"))
        rp0_phase = self.range_fft(tx0_data)
        rp2_phase = self.range_fft(tx2_data)

        _, pdb_s = self.range_profile([rp0, rp2], min_bin, max_bin)
        rng_prof = rng[min_bin:max_bin + 1]
        rpeaks = self.find_range_peaks(pdb_s, rng_prof)
        rb_cal = min_bin + int(rpeaks[0][0])
        seed_rbs = sorted(set([min_bin + int(pk[0]) for pk in rpeaks] + [rb_cal]))

        rp2_al, _ = self.calibrate_tx2(rp0, rp2, rb_cal, halfspan=4)
        rp2_phase_al, _ = self.calibrate_tx2(rp0_phase, rp2_phase, rb_cal, halfspan=4)

        best = self.score_config(rp0, rp2, rp2_al, seed_rbs)
        rp2_use = rp2_al if best["align"] == "aligned" else rp2
        rp2_phase_use = rp2_phase_al if best["align"] == "aligned" else rp2_phase
        v_best = self.build_virtual(rp0, rp2_use, best["order"])
        v_phase = self.build_virtual(rp0_phase, rp2_phase_use, best["order"])

        r_bins = np.arange(min_bin, max_bin + 1)
        rng_sel = rng[r_bins]
        ra = self.mvdr_map(v_best, r_bins, ANGLES, sign=int(best["sign"]))
        ra_n = ra / (np.max(np.abs(ra)) + 1e-12)
        ra_s = gaussian_filter(ra_n, sigma=(1.0, 1.0)) if SCIPY_OK else ra_n

        # Kandidat target dibuat sama dengan notebook master:
        # gabungan local maxima 2D dan row peaks pada kandidat range bin.
        eval_rbs = self.neighbor_bins(rpeaks, rng, min_bin, max_bin)
        if not eval_rbs:
            eval_rbs = seed_rbs[:]
        cand_ris = [int(np.clip(rb - min_bin, 0, len(r_bins) - 1)) for rb in eval_rbs]
        peaks2d = self.find_2d_peaks(ra_s)
        rowpeaks = self.find_row_peaks(ra_s, cand_ris)
        cands = self.build_candidates_master(peaks2d, rowpeaks, rng_sel, ANGLES)
        targets = self.select_targets_master(cands, rng_sel, ANGLES, ra_s, TOP_K, STRICT_OPPOSITE)
        for t in targets:
            ri = int(getattr(t, "_ri", 0))
            t.range_bin = int(r_bins[ri])

        subjects = [self.process_subject(v_phase, t, sign=int(best["sign"])) for t in targets]

        # ------------------------------------------------------------
        # Mapping tampilan. Ini tidak membuat nilai statis.
        # Sudut dibalik agar arah kanan/kiri tampilan sesuai konvensi GUI,
        # lalu subjek diurutkan dari sudut positif ke negatif.
        # HR/RR dan sinyal tetap milik target hasil proses radar yang sama.
        # ------------------------------------------------------------
        for subj in subjects:
            subj.target.angle_deg = -float(subj.target.angle_deg)

        subjects.sort(key=lambda s: s.target.angle_deg, reverse=True)
        for idx, subj in enumerate(subjects, 1):
            subj.target.subject_id = idx

        targets = [subj.target for subj in subjects]

        duration = n_frames / FS_VITAL
        return {
            "file_path": file_path,
            "n_frames": n_frames,
            "duration_sec": duration,
            "rng": rng,
            "rng_sel": rng_sel,
            "r_bins": r_bins,
            "angles": ANGLES,
            "ra": ra_s,
            "targets": targets,
            "subjects": subjects,
            "best": best,
            "fs_vital": FS_VITAL,
        }


# ============================================================
# GUI ringan dan modern
# ============================================================

class VitalSignApp:
    BG = "#f4f7fb"
    PANEL = "#ffffff"
    BORDER = "#d9e2ef"
    TEXT = "#111827"
    MUTED = "#64748b"
    BLUE = "#1d5fd0"
    RED = "#e12929"
    GREEN = "#16a34a"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vital Sign Semi Real-Time Monitor - Multi Subject Detection for Heart Rate & Respiratory Rate")
        self.root.geometry("1440x920")
        self.root.minsize(1180, 760)
        self.root.configure(bg=self.BG)

        self.file_path = tk.StringVar(value="Belum ada dataset .bin")
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame: - / -")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.processor = RadarVitalProcessor()
        self.result: Optional[Dict] = None
        self.playing = False
        self.play_index = 0
        self.plot_window_sec = 60.0
        self.frames_per_tick = 5      # 5 frame = 0.25 s data
        self.tick_ms = 250            # real-time ringan: 5 frame radar diputar setiap 0.25 detik

        self._setup_style()
        self._build_layout()
        self._build_plots()
        self.draw_empty()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("Green.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7), foreground=self.GREEN)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7), foreground=self.BLUE)
        style.configure("Horizontal.TProgressbar", thickness=10)

    def panel(self, parent, **grid):
        f = tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        f.grid(**grid)
        return f

    def _build_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=2)
        self.root.rowconfigure(2, weight=3)

        # Header
        header = tk.Frame(self.root, bg=self.BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(14, 8))
        header.columnconfigure(1, weight=1)

        logo = tk.Canvas(header, width=54, height=54, bg=self.BG, highlightthickness=0)
        logo.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        logo.create_oval(5, 5, 49, 49, outline=self.BLUE, width=2)
        logo.create_line(12, 28, 18, 28, 22, 18, 28, 38, 34, 24, 39, 28, 45, 28, fill=self.BLUE, width=2)

        tk.Label(header, text="Vital Sign Semi Real-Time Monitor - Multi Subject Detection for Heart Rate & Respiratory ", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 20, "bold")).grid(row=0, column=1, sticky="w")
        tk.Label(header, text="Radar FMCW - Multi-Subject Monitoring | Input: radar .bin only  Window grafik: 60 detik | ", 
                 bg=self.BG, fg=self.MUTED, font=("Segoe UI", 11)).grid(row=1, column=1, sticky="w")
        tk.Label(header, text="Mode: Semi real-time from radar .bin", bg=self.BG, fg=self.BLUE,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="e", padx=6)

        # Top panels
        self.loc_panel = self.panel(self.root, row=1, column=0, sticky="nsew", padx=(18, 8), pady=8)
        self.loc_panel.rowconfigure(1, weight=1)
        self.loc_panel.columnconfigure(0, weight=1)
        tk.Label(self.loc_panel, text="◎  Lokalisasi Target", bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))

        self.cards_panel = tk.Frame(self.root, bg=self.BG)
        self.cards_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=8)
        self.cards_panel.columnconfigure(0, weight=1)
        self.cards_panel.rowconfigure(0, weight=1)
        self.cards_panel.rowconfigure(1, weight=1)

        self.card_vars = []
        self.card_canvases = []
        self._create_subject_card(0, self.BLUE)
        self._create_subject_card(1, self.RED)

        # Plot area
        self.plots_panel = tk.Frame(self.root, bg=self.BG)
        self.plots_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=18, pady=(0, 8))
        self.plots_panel.columnconfigure(0, weight=1)
        self.plots_panel.columnconfigure(1, weight=1)
        self.plots_panel.rowconfigure(0, weight=1)
        self.plots_panel.rowconfigure(1, weight=1)

        # Bottom control
        bottom = tk.Frame(self.root, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 12))
        bottom.columnconfigure(2, weight=1)

        ttk.Button(bottom, text="📁 Load Dataset", command=self.load_dataset).grid(row=0, column=0, padx=(14, 6), pady=12)
        ttk.Button(bottom, text="⚙ Process", style="Accent.TButton", command=self.process_dataset).grid(row=0, column=1, padx=6, pady=12)
        tk.Label(bottom, textvariable=self.file_path, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=10)
        tk.Label(bottom, textvariable=self.frame_var, bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 10)).grid(row=0, column=3, padx=8)
        ttk.Progressbar(bottom, orient="horizontal", mode="determinate", variable=self.progress_var,
                        length=240).grid(row=0, column=4, padx=8)
        ttk.Button(bottom, text="▶ Start", style="Green.TButton", command=self.start).grid(row=0, column=5, padx=6)
        ttk.Button(bottom, text="Ⅱ Pause", command=self.pause).grid(row=0, column=6, padx=6)
        ttk.Button(bottom, text="↻ Reset", command=self.reset).grid(row=0, column=7, padx=(6, 14))

        status = tk.Frame(self.root, bg=self.BG)
        status.grid(row=4, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 8))
        tk.Label(status, text="Dataset:", bg=self.BG, fg=self.MUTED, font=("Segoe UI", 9)).pack(side="left")
        tk.Label(status, textvariable=self.file_path, bg=self.BG, fg=self.BLUE, font=("Segoe UI", 9)).pack(side="left", padx=(4, 20))
        tk.Label(status, text="●", bg=self.BG, fg=self.GREEN, font=("Segoe UI", 10)).pack(side="right")
        tk.Label(status, textvariable=self.status_var, bg=self.BG, fg=self.GREEN,
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=6)

    def _create_subject_card(self, idx: int, color: str):
        card = tk.Frame(self.cards_panel, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        card.grid(row=idx, column=0, sticky="nsew", padx=0, pady=(0, 8) if idx == 0 else (8, 0))
        card.columnconfigure(0, weight=1)

        stripe = tk.Frame(card, bg=color, width=7)
        stripe.grid(row=0, column=0, rowspan=5, sticky="nsw")

        top = tk.Frame(card, bg=self.PANEL)
        top.grid(row=0, column=0, sticky="ew", padx=(18, 14), pady=(14, 6))
        top.columnconfigure(0, weight=1)
        tk.Label(top, text=f"Subjek {idx+1}", bg=self.PANEL, fg=color,
                 font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(top, text="● Tracking", bg="#dcfce7", fg=self.GREEN,
                 font=("Segoe UI", 9, "bold"), padx=10, pady=3).grid(row=0, column=1, sticky="e")

        metric = tk.Frame(card, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        metric.grid(row=1, column=0, sticky="ew", padx=(25, 14), pady=8)
        metric.columnconfigure(0, weight=1)
        metric.columnconfigure(1, weight=1)

        hr_val = tk.StringVar(value="--")
        rr_val = tk.StringVar(value="--")
        pos_val = tk.StringVar(value="Posisi: -- m, --°")
        self.card_vars.append({"hr": hr_val, "rr": rr_val, "pos": pos_val})

        tk.Label(metric, text="♥", bg=self.PANEL, fg=color, font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w", padx=(20, 0), pady=(12, 0))
        tk.Label(metric, text="HR Radar", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", padx=(65, 0), pady=(8, 0))
        tk.Label(metric, textvariable=hr_val, bg=self.PANEL, fg=color, font=("Segoe UI", 23, "bold")).grid(row=1, column=0, sticky="w", padx=(65, 0), pady=(0, 12))
        tk.Label(metric, text="BPM", bg=self.PANEL, fg=color, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(124, 0), pady=(6, 12))

        tk.Frame(metric, bg=self.BORDER, width=1).grid(row=0, column=0, rowspan=2, sticky="nse", padx=(0, 0), pady=12)

        tk.Label(metric, text="♨", bg=self.PANEL, fg=color, font=("Segoe UI", 23, "bold")).grid(row=0, column=1, sticky="w", padx=(20, 0), pady=(12, 0))
        tk.Label(metric, text="RR Radar", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 10)).grid(row=0, column=1, sticky="w", padx=(65, 0), pady=(8, 0))
        tk.Label(metric, textvariable=rr_val, bg=self.PANEL, fg=color, font=("Segoe UI", 23, "bold")).grid(row=1, column=1, sticky="w", padx=(65, 0), pady=(0, 12))
        tk.Label(metric, text="BPM", bg=self.PANEL, fg=color, font=("Segoe UI", 10, "bold")).grid(row=1, column=1, sticky="w", padx=(124, 0), pady=(6, 12))

        tk.Label(card, textvariable=pos_val, bg=self.PANEL, fg=color,
                 font=("Segoe UI", 12, "bold")).grid(row=2, column=0, sticky="w", padx=(28, 14), pady=(2, 14))

    def _add_plot(self, row: int, col: int, title: str) -> Tuple[Figure, object, FigureCanvasTkAgg, object]:
        panel = tk.Frame(self.plots_panel, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        panel.grid(row=row, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0), pady=(0, 8) if row == 0 else (8, 0))
        fig = Figure(figsize=(6.4, 2.4), dpi=100, facecolor=self.PANEL)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.24)
        canvas = FigureCanvasTkAgg(fig, master=panel)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=8)
        line, = ax.plot([], [], linewidth=1.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Waktu (s)", fontsize=9)
        ax.set_ylabel("Amplitudo (a.u.)", fontsize=9)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, self.plot_window_sec)
        ax.grid(True, alpha=0.25)
        return fig, ax, canvas, line

    def _build_plots(self):
        # Lokalisasi target sekarang memakai peta radar 2D manual, bukan polar plot bawaan.
        # Ini membuat tampilan lebih besar, rapi, dan tidak ada label jarak yang saling menumpuk.
        self.loc_fig = Figure(figsize=(8.8, 4.8), dpi=100, facecolor=self.PANEL)
        self.loc_ax = self.loc_fig.add_subplot(111)
        self.loc_fig.subplots_adjust(left=0.10, right=0.83, top=0.84, bottom=0.18)
        self.loc_canvas = FigureCanvasTkAgg(self.loc_fig, master=self.loc_panel)
        self.loc_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=14, pady=8)

        self.fig_hr1, self.ax_hr1, self.canvas_hr1, self.line_hr1 = self._add_plot(0, 0, "Sinyal HR - Subjek 1")
        self.fig_rr1, self.ax_rr1, self.canvas_rr1, self.line_rr1 = self._add_plot(0, 1, "Sinyal RR - Subjek 1")
        self.fig_hr2, self.ax_hr2, self.canvas_hr2, self.line_hr2 = self._add_plot(1, 0, "Sinyal HR - Subjek 2")
        self.fig_rr2, self.ax_rr2, self.canvas_rr2, self.line_rr2 = self._add_plot(1, 1, "Sinyal RR - Subjek 2")

    # ---------- Drawing ----------
    def draw_empty(self):
        self.draw_radar_background()
        self.loc_canvas.draw_idle()

        for line, ax, canvas in [
            (self.line_hr1, self.ax_hr1, self.canvas_hr1),
            (self.line_rr1, self.ax_rr1, self.canvas_rr1),
            (self.line_hr2, self.ax_hr2, self.canvas_hr2),
            (self.line_rr2, self.ax_rr2, self.canvas_rr2),
        ]:
            line.set_data([], [])
            ax.set_xlim(0, self.plot_window_sec)
            ax.set_ylim(-1.2, 1.2)
            canvas.draw_idle()

    def draw_radar_background(self, max_r: float = 2.5, angle_limit: float = 60.0):
        """
        Menggambar latar lokalisasi target sebagai peta 2D.

        Koordinat:
        - x = posisi lateral kiri/kanan radar, dalam meter
        - y = jarak depan radar, dalam meter

        Dengan cara ini tampilan jauh lebih rapi daripada polar plot default Matplotlib.
        """
        ax = self.loc_ax
        ax.clear()
        ax.set_title("Lokalisasi Target", fontsize=12, fontweight="bold", pad=10)

        theta = np.deg2rad(np.linspace(-angle_limit, angle_limit, 240))

        # Busur jarak 0.5 m sampai 2.5 m
        for r in np.arange(0.5, max_r + 0.001, 0.5):
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            ax.plot(x, y, color="#d7dde8", linewidth=0.9, linestyle="--", zorder=1)
            ax.text(
                0.04,
                r,
                f"{r:.1f} m",
                fontsize=8,
                color="#64748b",
                ha="left",
                va="bottom",
            )

        # Garis sudut
        for deg in [-60, -45, -30, 0, 30, 45, 60]:
            th = math.radians(deg)
            x2 = max_r * math.sin(th)
            y2 = max_r * math.cos(th)
            ax.plot([0, x2], [0, y2], color="#e2e8f0", linewidth=0.9, zorder=1)

            tx = (max_r + 0.18) * math.sin(th)
            ty = (max_r + 0.18) * math.cos(th)
            ax.text(tx, ty, f"{deg}°", fontsize=9, color="#475569", ha="center", va="center")

        # Batas sektor kiri dan kanan
        for deg in [-angle_limit, angle_limit]:
            th = math.radians(deg)
            ax.plot(
                [0, max_r * math.sin(th)],
                [0, max_r * math.cos(th)],
                color="#94a3b8",
                linewidth=1.2,
                zorder=2,
            )

        # Garis tengah depan radar
        ax.plot([0, 0], [0, max_r], color="#cbd5e1", linewidth=1.1, zorder=2)

        # Posisi radar
        ax.scatter([0], [0], s=80, color="#111827", marker="^", label="Radar", zorder=6)
        ax.text(0, -0.13, "Radar", fontsize=8, color="#111827", ha="center", va="top")

        ax.set_xlim(-2.35, 2.35)
        ax.set_ylim(-0.18, max_r + 0.35)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Posisi lateral x (m)", fontsize=9)
        ax.set_ylabel("Jarak depan y (m)", fontsize=9)
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.8)

        for spine in ax.spines.values():
            spine.set_color("#d7dde8")

    def draw_localization_once(self):
        if self.result is None:
            self.draw_empty()
            return

        targets = self.result["targets"]
        max_r = max(2.5, max(t.range_m for t in targets[:2]) + 0.7)
        self.draw_radar_background(max_r=max_r, angle_limit=60.0)

        colors = [self.BLUE, self.RED]
        labels = ["Subjek 1", "Subjek 2"]

        for i, t in enumerate(targets[:2]):
            theta = math.radians(t.angle_deg)
            x = t.range_m * math.sin(theta)
            y = t.range_m * math.cos(theta)

            # Garis dari radar ke target
            self.loc_ax.plot(
                [0, x],
                [0, y],
                color=colors[i],
                linewidth=1.5,
                alpha=0.38,
                zorder=3,
            )

            # Titik target
            self.loc_ax.scatter(
                [x],
                [y],
                s=105,
                color=colors[i],
                edgecolor="white",
                linewidth=1.4,
                label=labels[i],
                zorder=7,
            )

            # Offset label agar tidak saling menumpuk.
            # Jika posisi target berdekatan, label otomatis digeser beda arah.
            dx = -0.22 if i == 0 else 0.22
            dy = 0.18 if i == 0 else -0.08

            if len(targets) >= 2:
                other = targets[1 - i]
                th_o = math.radians(other.angle_deg)
                xo = other.range_m * math.sin(th_o)
                yo = other.range_m * math.cos(th_o)
                if math.hypot(x - xo, y - yo) < 0.45:
                    dx = -0.32 if i == 0 else 0.32
                    dy = 0.22 if i == 0 else -0.20

            self.loc_ax.text(
                x + dx,
                y + dy,
                f"S{i + 1}\n{t.range_m:.2f} m\n{t.angle_deg:.1f}°",
                color=colors[i],
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=colors[i],
                    linewidth=0.9,
                    alpha=0.92,
                ),
                zorder=8,
            )

        self.loc_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
            frameon=True,
        )

        self.loc_canvas.draw_idle()

    def update_line(self, line, ax, canvas, t: np.ndarray, y: np.ndarray, signal_type: str = "hr"):
        if self.result is None:
            return
        fs = self.result["fs_vital"]
        end = max(1, min(self.play_index, len(t) - 1))
        win = int(self.plot_window_sec * fs)
        start = max(0, end - win)
        tt = t[start:end + 1]
        yy = y[start:end + 1]
        if len(tt) == 0:
            return
        xx = tt - tt[0]

        # Normalisasi hanya untuk window yang sedang tampil.
        # Ini membuat RR yang amplitudonya kecil tetap terlihat, terutama pada window 60 detik.
        yy_vis = normalize_visible_window(yy, signal_type=signal_type)

        line.set_data(xx, yy_vis)
        ax.set_xlim(0, self.plot_window_sec)
        ax.set_ylim(-1.2, 1.2)
        canvas.draw_idle()

    def current_bpm(self, subj: SubjectResult) -> Tuple[float, float]:
        # Angka kartu dibuat stabil saat playback.
        # Sesuai tabel evaluasi notebook:
        # - RR memakai Full 180 detik
        # - HR memakai 3 Segment Mean
        rr = safe_float(subj.rr_display_bpm, np.nan)
        hr = safe_float(subj.hr_display_bpm, np.nan)
        return rr, hr

    def update_cards(self):
        if self.result is None:
            return
        for i, subj in enumerate(self.result["subjects"][:2]):
            rr, hr = self.current_bpm(subj)
            t = subj.target
            self.card_vars[i]["hr"].set(f"{hr:.2f}" if np.isfinite(hr) else "--")
            self.card_vars[i]["rr"].set(f"{rr:.2f}" if np.isfinite(rr) else "--")
            self.card_vars[i]["pos"].set(f"◎ Posisi: {t.range_m:.2f} m, {t.angle_deg:.0f}°")

    def update_frame_info(self):
        if self.result is None:
            self.frame_var.set("Frame: - / -")
            self.progress_var.set(0)
            return
        total = int(self.result["n_frames"])
        cur = int(np.clip(self.play_index, 0, total))
        self.frame_var.set(f"Frame: {cur} / {total}")
        self.progress_var.set(100.0 * cur / max(total, 1))

    def update_dynamic_view(self):
        if self.result is None:
            return
        subjects: List[SubjectResult] = self.result["subjects"]
        self.update_cards()
        self.update_line(self.line_hr1, self.ax_hr1, self.canvas_hr1, subjects[0].t, subjects[0].hr_display, signal_type="hr")
        self.update_line(self.line_rr1, self.ax_rr1, self.canvas_rr1, subjects[0].t, subjects[0].rr_display, signal_type="rr")
        self.update_line(self.line_hr2, self.ax_hr2, self.canvas_hr2, subjects[1].t, subjects[1].hr_display, signal_type="hr")
        self.update_line(self.line_rr2, self.ax_rr2, self.canvas_rr2, subjects[1].t, subjects[1].rr_display, signal_type="rr")
        self.update_frame_info()

    # ---------- Actions ----------
    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Pilih file radar .bin",
            filetypes=[("Radar binary", "*.bin"), ("All files", "*.*")]
        )
        if path:
            self.file_path.set(path)
            self.status_var.set("Dataset selected")
            self.result = None
            self.playing = False
            self.play_index = 0
            self.draw_empty()
            for i in range(2):
                self.card_vars[i]["hr"].set("--")
                self.card_vars[i]["rr"].set("--")
                self.card_vars[i]["pos"].set("Posisi: -- m, --°")
            self.update_frame_info()

    def process_dataset(self):
        path = self.file_path.get()
        if not path or path == "Belum ada dataset .bin" or not Path(path).exists():
            messagebox.showwarning("Dataset belum dipilih", "Pilih file .bin terlebih dahulu.")
            return

        self.playing = False
        self.status_var.set("Processing radar data...")
        self.progress_var.set(0)
        self.root.update_idletasks()

        def worker():
            try:
                result = self.processor.process(path)
                self.root.after(0, lambda: self.on_process_done(result))
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, lambda: self.on_process_error(str(exc), tb))

        threading.Thread(target=worker, daemon=True).start()


    def save_last_result_csv(self):
        """Menyimpan hasil angka yang sama dengan kartu GUI agar mudah dicek."""
        if self.result is None:
            return
        try:
            import csv
            dataset_path = Path(self.result.get("file_path", ""))
            out_path = dataset_path.with_name("last_gui_verified_result.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Subject", "Range_m", "Angle_deg", "Range_bin", "HR_full_BPM", "HR_3seg_BPM", "HR_display_BPM", "RR_full_BPM", "RR_3seg_BPM", "RR_display_BPM"])
                for i, subj in enumerate(self.result["subjects"][:2], 1):
                    writer.writerow([
                        f"Subject {i}",
                        f"{subj.target.range_m:.6f}",
                        f"{subj.target.angle_deg:.6f}",
                        int(subj.target.range_bin),
                        f"{subj.hr_full:.6f}",
                        f"{subj.hr_3seg:.6f}",
                        f"{subj.hr_display_bpm:.6f}",
                        f"{subj.rr_full:.6f}",
                        f"{subj.rr_3seg:.6f}",
                        f"{subj.rr_display_bpm:.6f}",
                    ])
            self.last_result_csv = str(out_path)
        except Exception:
            self.last_result_csv = ""

    def on_process_done(self, result: Dict):
        self.result = result
        self.save_last_result_csv()
        # Mulai dari awal dataset. Angka kartu memakai hasil full 180 detik,
        # sehingga tidak perlu menunggu rolling window 30 detik.
        self.play_index = 0
        self.draw_localization_once()
        self.update_dynamic_view()
        best = result.get("best", {})
        self.status_var.set(
            f"Ready | angka kartu = HR 3Seg, RR Full | {result['n_frames']} frames | {result['duration_sec']:.1f} s | MVDR {best.get('align','')}, sign {best.get('sign','')}"
        )
        lines = [
            "Dataset selesai diproses.",
            "",
            "Aturan angka kartu GUI:",
            "HR = 3 Segment Mean",
            "RR = Full 180 detik",
            "",
            "Hasil yang ditampilkan pada kartu GUI:",
        ]
        for i, subj in enumerate(result["subjects"][:2], 1):
            lines.append(
                f"Subject {i}: HR {subj.hr_display_bpm:.2f} BPM (3Seg {subj.hr_3seg:.2f}, Full {subj.hr_full:.2f}) | "
                f"RR {subj.rr_display_bpm:.2f} BPM (Full {subj.rr_full:.2f}, 3Seg {subj.rr_3seg:.2f}) | "
                f"Posisi {subj.target.range_m:.2f} m, {subj.target.angle_deg:.1f}°"
            )
        if getattr(self, "last_result_csv", ""):
            lines.append("")
            lines.append(f"CSV cek hasil: {self.last_result_csv}")
        lines.append("")
        lines.append("Klik Start untuk menjalankan grafik HR/RR. Angka kartu tidak berubah saat playback.")
        messagebox.showinfo("Selesai", "\n".join(lines))

    def on_process_error(self, msg: str, tb: str):
        self.status_var.set("Processing error")
        messagebox.showerror("Error saat proses dataset", f"{msg}\n\nDetail:\n{tb[:2200]}")

    def start(self):
        if self.result is None:
            messagebox.showwarning("Belum diproses", "Klik Process terlebih dahulu.")
            return
        self.playing = True
        self.status_var.set("Running")
        self._play_loop()

    def pause(self):
        self.playing = False
        if self.result is not None:
            self.status_var.set("Paused")

    def reset(self):
        self.playing = False
        if self.result is not None:
            self.play_index = 0
            self.status_var.set("Reset")
            self.update_dynamic_view()
        else:
            self.draw_empty()
            self.update_frame_info()

    def _play_loop(self):
        if not self.playing or self.result is None:
            return
        total = int(self.result["n_frames"])
        self.play_index += self.frames_per_tick
        if self.play_index >= total:
            self.play_index = total
            self.playing = False
            self.status_var.set("Finished | playback selesai sampai akhir dataset")
        self.update_dynamic_view()
        if self.playing:
            self.root.after(self.tick_ms, self._play_loop)


def main():
    print("RUNNING GUI VERSION: MASTER EXACT V8 - 60s HYBRID BPM")
    root = tk.Tk()
    app = VitalSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

