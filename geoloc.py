#!/usr/bin/env python3
"""Locate where a recording was made from the Doppler curves of aircraft flyovers.

Pipeline (after the method described in https://youtu.be/UytYrhYLOKI):
  1. Track the propeller tone of each flyover in the spectrogram and fit a flyby
     model -> speed, closest-approach (CPA) slant distance, arrival time.
  2. From ADS-B tracks, list every (candidate grid point, aircraft pass) whose
     speed and CPA slant distance match an observed flyover.
  3. Keep candidates whose passes also match in time: either the known recording
     time, or the time offset between two flyovers (no clock needed).
  4. Model the exact Doppler curve each surviving candidate would hear
     (emission-time radial velocity, sound propagation delay) and score it
     against the observed curve with one shared clock shift for all flyovers.
"""
import argparse, glob, gzip, json, os, pickle, re, subprocess
from multiprocessing import Pool
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
from scipy import optimize, signal
from scipy.interpolate import RegularGridInterpolator

C_SOUND = 343.0
A_WGS, E2_WGS = 6378137.0, 6.69437999014e-3


def lla_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2_WGS * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2_WGS) + h) * np.sin(lat)], axis=-1)


# ----------------------------------------------------------------- ADS-B data
@dataclass
class Track:
    icao: str
    t: np.ndarray          # unix seconds, increasing
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray        # metres (geometric if available)
    actype: str = ""
    xyz: np.ndarray = field(init=False)

    def __post_init__(self):
        o = np.argsort(self.t)
        self.t, self.lat, self.lon, self.alt = (np.asarray(a, float)[o] for a in
                                                (self.t, self.lat, self.lon, self.alt))
        self.xyz = lla_to_ecef(self.lat, self.lon, self.alt)

    def pos(self, t):
        t = np.asarray(t, float)
        return np.stack([np.interp(t.ravel(), self.t, self.xyz[:, k]) for k in range(3)],
                        -1).reshape(t.shape + (3,))


def load_readsb_trace(path):
    """readsb / tar1090 'trace_full_<icao>.json' (the adsb.lol / ADSBx history format)."""
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    j = json.loads(raw)
    t, lat, lon, alt = [], [], [], []
    for p in j.get("trace", []):
        a = p[10] if len(p) > 10 and isinstance(p[10], (int, float)) else p[3]
        if not isinstance(a, (int, float)):
            continue                       # on the ground
        t.append(j["timestamp"] + p[0]); lat.append(p[1]); lon.append(p[2]); alt.append(a * 0.3048)
    return Track(j.get("icao", os.path.basename(path)), t, lat, lon, alt, j.get("t", "")) if len(t) > 2 else None


def load_csv(path):
    """CSV with header: icao,time,lat,lon,alt_m[,type]  (time = unix seconds)."""
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    out = []
    for icao in np.unique(rows["icao"]):
        r = rows[rows["icao"] == icao]
        ty = str(r["type"][0]) if "type" in rows.dtype.names else ""
        if len(r) > 2:
            out.append(Track(str(icao), r["time"], r["lat"], r["lon"], r["alt_m"], ty))
    return out


JET = re.compile(r"^(A[123]\d\w|A1[0-9]\w?|B7[0-9]\w|B3[789]M|B46\d|BCS\d|CRJ\w|E1[3479]\d|E[1279]\d\w|E[35]\dP|E545|E550|"
                 r"C25\w|C5[0-9]\w|C6[58]\w|C7[05]\w|C17|C5M|CL\d\d|GL\w\w|GLF\d|G[1-7]\d0|GA\dC|GALX|ASTR|F2TH|F900|FA\w\w|"
                 r"LJ\d\d|H25\w|HA4T|BE40|PC24|MD\d\d|DC\d\w|HDJT|SF50|PRM1|EA50|K35\w|F1[5-8]\w?|F22|F35|T38|B52|B1|B2)$")


def load_pickle(path, drop_jets=True):
    """Compact day file made by extract.py: {icao: {type, reg, a=[t,lat,lon,alt_baro_ft,alt_geom_ft,gs]}}."""
    out = []
    for icao, r in pickle.load(open(path, "rb")).items():
        if drop_jets and JET.match(r["type"] or ""):
            continue
        a = r["a"]
        off = np.nanmedian(a[:, 4] - a[:, 3]) if np.isfinite(a[:, 4] - a[:, 3]).any() else 0.0
        alt = np.where(np.isfinite(a[:, 4]), a[:, 4], a[:, 3] + off) * 0.3048   # geometric if we have it
        ok = np.isfinite(alt)
        if ok.sum() > 2:
            out.append(Track(icao, a[ok, 0], a[ok, 1], a[ok, 2], alt[ok], f"{r['type']} {r['reg']}".strip()))
    return out


def load_tracks(path, types=None):
    tracks = []
    if path.endswith(".pkl"):
        return load_pickle(path)
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "**", "*.json*"), recursive=True):
            tr = load_readsb_trace(f)
            if tr:
                tracks.append(tr)
        for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
            tracks += load_csv(f)
    elif path.endswith(".csv"):
        tracks = load_csv(path)
    else:
        tracks = [t for t in [load_readsb_trace(path)] if t]
    if types:                              # e.g. propeller ICAO type codes: C172,P28A,...
        tracks = [t for t in tracks if t.actype in types]
    return tracks


# ------------------------------------------------------- audio -> flyby events
@dataclass
class Event:
    t: np.ndarray      # seconds from recording start (arrival time of each sample)
    f: np.ndarray      # observed frequency, Hz
    t_arr: float       # arrival time of the CPA sound
    v: float           # aircraft speed, m/s
    d: float           # CPA slant distance, m
    f0: float          # emitted frequency, Hz


def load_audio(path, fs=8000):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(fs), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32), fs


def extract_curve(x, fs, t0, t1, fmin, fmax, win=0.5, hop=0.1, max_slope=40.0, snr_db=8.0):
    """Follow the strongest smoothly-moving tone in [fmin,fmax] between t0 and t1 (Viterbi ridge)."""
    seg = x[int(t0 * fs):int(t1 * fs)]
    q = max(1, int(fs // (4 * fmax)))
    if q > 1:
        seg, fs = signal.resample_poly(seg, 1, q), fs / q
    nper = int(win * fs)
    fr, tt, S = signal.spectrogram(seg, fs, nperseg=nper, noverlap=nper - int(hop * fs), nfft=8 * nper)
    band = (fr >= fmin) & (fr <= fmax)
    fr, S = fr[band], 10 * np.log10(S[band] + 1e-20)
    S = S - np.median(S, axis=1, keepdims=True)        # remove steady hum (mains, fans)
    df = fr[1] - fr[0]
    jump = max(1, int(np.ceil(max_slope * hop / df)))  # max bins the tone may move per frame
    nb, nt = S.shape
    cost, back = S[:, 0].copy(), np.zeros((nb, nt), int)
    offs = np.arange(-jump, jump + 1)
    for k in range(1, nt):
        pad = np.pad(cost, jump, constant_values=-1e9)
        cand = np.stack([pad[jump + o:jump + o + nb] for o in offs]) - 0.5 * np.abs(offs)[:, None] / jump
        best = cand.argmax(0)
        back[:, k] = np.arange(nb) + offs[best]
        cost = cand[best, np.arange(nb)] + S[:, k]
    path = np.zeros(nt, int); path[-1] = cost.argmax()
    for k in range(nt - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    pk = S[path, np.arange(nt)]
    f = fr[path].astype(float)
    i = np.clip(path, 1, nb - 2)                        # parabolic peak refinement
    a, b, c = S[i - 1, np.arange(nt)], S[i, np.arange(nt)], S[i + 1, np.arange(nt)]
    den = a - 2 * b + c
    f += np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0).clip(-1, 1) * df
    ok = pk > snr_db
    return t0 + tt[ok], f[ok]


def flyby_model(p, t, c=C_SOUND):
    """Received frequency at arrival times t for a straight constant-speed pass."""
    f0, v, d, tc = p
    te = np.linspace(t[0] - tc - 60, t[-1] - tc + 5, 4000)
    r = np.hypot(d, v * te)
    return np.interp(t, tc + te + r / c, f0 / (1 + v * v * te / r / c))


def fit_flyby(t, f, c=C_SOUND, clip=True):
    n = max(3, len(f) // 10)
    hi, lo = np.median(f[:n]), np.median(f[-n:])
    f0 = 0.5 * (hi + lo)
    v = max(10.0, c * (hi - lo) / (hi + lo))
    tc = t[np.argmin(np.abs(f - f0))]
    slope = np.abs(np.gradient(signal.medfilt(f, 5), t)).max() + 1e-6
    d = np.clip(f0 * v * v / (c * slope), 50, 20000)
    best = None
    for d0 in (d, d * 2, d * 4, d / 2):
        r = optimize.least_squares(lambda p: flyby_model(p, t, c) - f, [f0, v, d0, tc], loss="soft_l1",
                                   f_scale=0.5, bounds=([1, 5, 10, t[0] - 120], [1e4, 300, 5e4, t[-1] + 120]))
        if best is None or r.cost < best.cost:
            best = r
    keep = np.abs(best.fun) < max(1.5, 3 * np.median(np.abs(best.fun)))   # drop stray tones, refit
    if clip and keep.sum() > 20 and not keep.all():
        return fit_flyby(t[keep], f[keep], c, clip=False)
    f0, v, d, tc = best.x
    return Event(t, f, tc + d / c, v, d, f0)


# ------------------------------------------------------- candidate generation
class Grid:
    def __init__(self, lat0, lon0, lat1, lon1, step):
        self.lat0, self.lon0 = min(lat0, lat1), min(lon0, lon1)
        self.dlat = step / 111320.0
        self.dlon = step / (111320.0 * np.cos(np.radians(0.5 * (lat0 + lat1))))
        self.ni = int(abs(lat1 - lat0) / self.dlat) + 1
        self.nj = int(abs(lon1 - lon0) / self.dlon) + 1

    def latlon(self, key):
        return self.lat0 + (key // self.nj) * self.dlat, self.lon0 + (key % self.nj) * self.dlon


def find_passes(tracks, grid, events, t_lo, t_hi, elev, tol_v, tol_d, node=10.0, chunk=6, max_gap=60.0):
    """All (grid point, aircraft pass) pairs whose CPA speed and slant distance fit some event.

    Each track is cut into straight 10 s legs; a grid point's closest approach to a leg is found by
    projecting the point onto it, so the cost is one dot product per (point, leg).
    """
    d_max = max(e.d for e in events) * (1 + tol_d)
    mlat = d_max / 111320.0
    out = []
    for ti, tr in enumerate(tracks):
        ts = np.arange(max(tr.t[0], t_lo), min(tr.t[-1], t_hi), node)
        if len(ts) < 2:
            continue
        P = tr.pos(ts)
        la, lo = np.interp(ts, tr.t, tr.lat), np.interp(ts, tr.t, tr.lon)
        dP = P[1:] - P[:-1]
        L = np.linalg.norm(dP, axis=1)
        v = L / node
        j = np.clip(np.searchsorted(tr.t, ts[:-1], side="right"), 1, len(tr.t) - 1)
        ok = (tr.t[j] - tr.t[j - 1] <= max_gap) & (L > 1)                 # no made-up legs across data gaps
        ok &= np.any([np.abs(v - e.v) <= tol_v * e.v for e in events], axis=0)
        mlon = mlat / np.cos(np.radians(la[0]))
        ok &= (la[:-1] > grid.lat0 - mlat) & (la[:-1] < grid.lat0 + grid.ni * grid.dlat + mlat)
        ok &= (lo[:-1] > grid.lon0 - mlon) & (lo[:-1] < grid.lon0 + grid.nj * grid.dlon + mlon)
        for s in range(0, len(L), chunk):
            e = min(s + chunk, len(L))
            if not ok[s:e].any():
                continue
            i0 = max(0, int(np.ceil((la[s:e + 1].min() - mlat - grid.lat0) / grid.dlat)))
            i1 = min(grid.ni - 1, int((la[s:e + 1].max() + mlat - grid.lat0) / grid.dlat))
            j0 = max(0, int(np.ceil((lo[s:e + 1].min() - mlon - grid.lon0) / grid.dlon)))
            j1 = min(grid.nj - 1, int((lo[s:e + 1].max() + mlon - grid.lon0) / grid.dlon))
            if i0 > i1 or j0 > j1:
                continue
            ii, jj = (g.ravel() for g in np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij"))
            clat, clon = grid.lat0 + ii * grid.dlat, grid.lon0 + jj * grid.dlon
            X = lla_to_ecef(clat, clon, elev(clat, clon))
            for k in range(s, e):
                if not ok[k]:
                    continue
                w = X - P[k]
                a = w @ (dP[k] / L[k])
                m = (a >= 0) & (a < L[k])
                if not m.any():
                    continue
                d = np.sqrt(np.maximum((w[m] ** 2).sum(1) - a[m] ** 2, 0))
                fit = np.zeros(len(d), bool)
                for ev in events:
                    if abs(v[k] - ev.v) <= tol_v * ev.v:
                        fit |= np.abs(d - ev.d) <= tol_d * ev.d
                if fit.any():
                    idx = np.flatnonzero(m)[fit]
                    out.append((ii[idx] * grid.nj + jj[idx], np.full(len(idx), ti),
                                ts[k] + a[idx] / v[k] + d[fit] / C_SOUND, d[fit], np.full(len(idx), v[k])))
    if not out:
        return [np.zeros(0)] * 5
    return [np.concatenate(c) for c in zip(*out)]      # key, track, t_arrival, d, v


# -------------------------------------------------------------------- scoring
def doppler_factor(tr, x, ta, c=C_SOUND):
    """f_received / f_emitted at arrival times ta for a listener at ECEF x."""
    te = ta - np.linalg.norm(tr.pos(ta) - x, axis=-1) / c
    for _ in range(4):
        te = ta - np.linalg.norm(tr.pos(te) - x, axis=-1) / c
    vr = (np.linalg.norm(tr.pos(te + 0.5) - x, axis=-1) - np.linalg.norm(tr.pos(te - 0.5) - x, axis=-1))
    return 1 / (1 + vr / c)


def score(x, trs, events, T0, shift_tol, c=C_SOUND):
    """RMS Hz misfit with one clock shift shared by all flyovers and a free tone per flyover."""
    def rms(delta):
        se = n = 0
        for tr, ev in zip(trs, events):
            g = doppler_factor(tr, x, T0 + delta + ev.t, c)
            r = ev.f - (ev.f * g).sum() / (g * g).sum() * g
            se, n = se + (r * r).sum(), n + len(r)
        return np.sqrt(se / n)
    grid = np.linspace(-shift_tol, shift_tol, 13)
    d0 = grid[np.argmin([rms(d) for d in grid])]
    h = grid[1] - grid[0]
    res = optimize.minimize_scalar(rms, bounds=(max(-shift_tol, d0 - h), min(shift_tol, d0 + h)), method="bounded")
    return res.fun, res.x


def _tile(args):
    """Match + score every candidate inside one tile. Uses module globals set by locate() (fork)."""
    bbox = args
    tracks, events, o = _G["tracks"], _G["events"], _G["opt"]
    m = 0.15
    sub = [k for k, tr in enumerate(tracks) if tr.lat.max() > bbox[0] - m and tr.lat.min() < bbox[2] + m
           and tr.lon.max() > bbox[1] - m and tr.lon.min() < bbox[3] + m]
    if not sub:
        return []
    grid = Grid(*bbox, o["step"])
    key, trk, tarr, d, v = find_passes([tracks[k] for k in sub], grid, events, o["t_lo"], o["t_hi"], o["elev"], o["tol_v"], o["tol_d"])
    if not len(key):
        return []
    trk = np.asarray(sub)[trk.astype(int)]
    key = key.astype(np.int64)
    fits = [np.flatnonzero((np.abs(v - e.v) <= o["tol_v"] * e.v) & (np.abs(d - e.d) <= o["tol_d"] * e.d)) for e in events]
    if o["start"] is not None:
        fits = [i[np.abs(tarr[i] - o["start"] - e.t_arr) <= o["time_tol"]] for i, e in zip(fits, events)]
    # chain events on (same grid point, arrival offset equal to the observed offset)
    SPAN = 4e5
    combos = fits[0][:, None]
    for e, idx in zip(events[1:], fits[1:]):
        comp = key[idx] * SPAN + (tarr[idx] - o["t_lo"])
        srt = np.argsort(comp); comp, idx = comp[srt], idx[srt]
        want = key[combos[:, 0]] * SPAN + (tarr[combos[:, 0]] - o["t_lo"]) + (e.t_arr - events[0].t_arr)
        lo, hi = np.searchsorted(comp, want - o["tol_dt"]), np.searchsorted(comp, want + o["tol_dt"])
        rep = hi - lo
        rows = np.repeat(np.arange(len(combos)), rep)
        pick = np.concatenate([np.arange(a, b) for a, b in zip(lo[rep > 0], hi[rep > 0])]) if rep.any() else np.zeros(0, int)
        combos = np.column_stack([combos[rows], idx[pick]])
        if not len(combos):
            return []
    out = []
    for combo in combos:
        if len(set(trk[combo])) < len(combo):
            continue
        la, ln = grid.latlon(int(key[combo[0]]))
        T0 = float(np.mean([tarr[i] - e.t_arr for i, e in zip(combo, events)]))
        x = lla_to_ecef(la, ln, float(o["elev"](la, ln)))
        rms, delta = score(x, [tracks[trk[i]] for i in combo], _G["thin"], T0, o["shift_tol"])
        out.append((rms, la, ln, T0 + delta, tuple(int(trk[i]) for i in combo)))
    if o.get("verbose"):
        print(f"tile {bbox[0]:.2f},{bbox[1]:.2f}: {len(key)} passes, {len(combos)} time-matched, best {min(out)[0] if out else float('nan'):.2f} Hz", flush=True)
    return out


_G = {}


def _zero_elevation(lat, lon):
    return np.zeros_like(np.asarray(lat, float))


def _init_worker(state):
    _G.update(state)


def locate(tracks, events, bbox, step=200.0, start=None, time_tol=30.0, window=None, elev=None,
           tol_v=0.25, tol_d=0.5, tol_dt=5.0, shift_tol=3.0, duration=None, silence_radius=None,
           top=10, refine=True, tile_deg=0.5, workers=4, verbose=False):
    """Rank grid points in bbox=(lat0,lon0,lat1,lon1).

    start  : recording start (unix s) if known, else None -> needs >=2 events and `window`.
    window : (t_lo, t_hi) unix-second range of ADS-B data to search when start is unknown.
    Returns the best point for each distinct set of aircraft, best first.
    """
    elev = elev or _zero_elevation
    if start is not None:
        t_lo, t_hi = start - time_tol - 300, start + max(e.t[-1] for e in events) + time_tol + 300
    elif window and len(events) >= 2:
        t_lo, t_hi = window
    else:
        raise ValueError("need either a start time, or two events plus a search window")
    thin = [Event(e.t[::max(1, len(e.t) // 60)], e.f[::max(1, len(e.t) // 60)], e.t_arr, e.v, e.d, e.f0) for e in events]
    _G.update(tracks=tracks, events=events, thin=thin, opt=dict(step=step, start=start, time_tol=time_tol, t_lo=t_lo,
              t_hi=t_hi, elev=elev, tol_v=tol_v, tol_d=tol_d, tol_dt=tol_dt, shift_tol=shift_tol, verbose=verbose))
    la0, lo0, la1, lo1 = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
    tiles = [(a, b, min(a + tile_deg, la1), min(b + tile_deg, lo1))
             for a in np.arange(la0, la1, tile_deg) for b in np.arange(lo0, lo1, tile_deg)]
    if workers > 1 and len(tiles) > 1:
        with Pool(workers, initializer=_init_worker, initargs=(_G,)) as pool:
            parts = list(pool.imap_unordered(_tile, tiles, chunksize=1))
    else:
        parts = [_tile(t) for t in tiles]
    allc = sorted(c for p in parts for c in p)
    locate.n_candidates = len(allc)
    best = {}
    for rms, la, ln, T0, ids in allc:                      # keep the best point per set of aircraft
        best.setdefault(ids, dict(lat=la, lon=ln, rms_hz=rms, start_unix=T0, trs=[tracks[i] for i in ids]))
    results = list(best.values())[:max(top, 30)]
    if refine:                                             # polish the leaders off-grid (finer than 100 m)
        dlat, dlon = step / 111320.0, step / 111320.0 / np.cos(np.radians(la0))
        for b in results:
            def cost(p):
                return score(lla_to_ecef(p[0], p[1], float(elev(p[0], p[1]))), b["trs"], events, b["start_unix"], shift_tol)[0]
            res = optimize.minimize(cost, [b["lat"], b["lon"]], method="Nelder-Mead",
                                    options=dict(initial_simplex=[[b["lat"], b["lon"]], [b["lat"] + dlat, b["lon"]],
                                                                  [b["lat"], b["lon"] + dlon]], xatol=1e-6, fatol=1e-5))
            b.update(grid_lat=b["lat"], grid_lon=b["lon"], lat=float(res.x[0]), lon=float(res.x[1]), rms_hz=float(res.fun))
        results.sort(key=lambda r: r["rms_hz"])
    if silence_radius and duration:                        # nothing else may fly close unheard
        for r in results:
            x = lla_to_ecef(r["lat"], r["lon"], float(elev(r["lat"], r["lon"])))
            ts = np.arange(r["start_unix"], r["start_unix"] + duration, 2.0)
            r["unheard"] = sum(1 for tr in tracks if tr not in r["trs"] and tr.t[0] < ts[-1] and tr.t[-1] > ts[0]
                               and np.linalg.norm(tr.pos(ts[(ts >= tr.t[0]) & (ts <= tr.t[-1])]) - x, axis=-1).min(initial=1e9) < silence_radius)
    for r in results:
        r["aircraft"] = [f"{t.icao} {t.actype}".strip() for t in r.pop("trs")]
        r["start_utc"] = datetime.fromtimestamp(r["start_unix"], timezone.utc).isoformat()
    return results[:top]


def confirm(x, fs, tracks, cand, events, elev=0.0, radius=9000.0, zmin=10.0, c=C_SOUND):
    """Independent check of one candidate: do OTHER planes that passed near it show up in the recording?

    With the candidate's position and start time fixed, every other nearby pass predicts a Doppler curve
    shape. Its tone is unknown, so slide the tone from 60 to 230 Hz and see whether the spectrogram is
    clearly brighter along the curve at one tone than at the rest (z-score). Returns the strong matches.
    """
    q = max(1, int(fs // 2000))
    y, f2 = (signal.resample_poly(x, 1, q), fs / q) if q > 1 else (x, fs)
    fr, tt, S = signal.spectrogram(y, f2, nperseg=int(f2), noverlap=int(f2 * 0.8), nfft=int(4 * f2))
    S = 10 * np.log10(S + 1e-20)
    S = np.clip(S - np.median(S, axis=1, keepdims=True), 0, 25)
    img = RegularGridInterpolator((fr, tt), S, bounds_error=False, fill_value=0)
    tones = np.arange(60, 230, 0.25)
    X, T0, dur = lla_to_ecef(cand["lat"], cand["lon"], elev), cand["start_unix"], len(x) / fs
    used = {a.split()[0] for a in cand["aircraft"]}
    hits = []
    for tr in tracks:
        if tr.icao in used or tr.t[-1] < T0 or tr.t[0] > T0 + dur:
            continue
        ts = np.arange(max(T0, tr.t[0]), min(T0 + dur, tr.t[-1]), 2.0)
        if len(ts) < 10:
            continue
        d = np.linalg.norm(tr.pos(ts) - X, axis=-1)
        k = d.argmin()
        ta = ts[k] + d[k] / c - T0
        if d[k] > radius or k < 5 or k > len(ts) - 6 or any(abs(ta - e.t_arr) < 90 for e in events):
            continue
        t = np.arange(ta - 45, ta + 45, 0.25)
        t = t[(t > 5) & (t < dur - 5)]
        gg = doppler_factor(tr, X, T0 + t, c)
        if len(t) < 100 or gg.max() / gg.min() < 1.08:       # too little pitch change to tell from a hum
            continue
        b = np.array([np.mean(img(np.c_[f * gg, t])) for f in tones])
        z = (b.max() - np.median(b)) / (1.4826 * np.median(np.abs(b - np.median(b))) + 1e-9)
        if z >= zmin:
            hits.append(dict(aircraft=f"{tr.icao} {tr.actype}".strip(), heard_at_s=round(float(ta)), slant_m=round(float(d[k])),
                             tone_hz=float(tones[b.argmax()]), z=round(float(z), 1)))
    return sorted(hits, key=lambda h: -h["z"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--event", action="append", required=True, metavar="T0:T1:FMIN:FMAX",
                    help="time window (s) and frequency band (Hz) of one flyover tone; repeat per flyover")
    ap.add_argument("--adsb", required=True, help="folder of readsb trace JSON files, or a CSV")
    ap.add_argument("--bbox", required=True, help="lat0,lon0,lat1,lon1 search area")
    ap.add_argument("--step", type=float, default=200.0)
    ap.add_argument("--start-utc", help="recording start, ISO 8601, if known")
    ap.add_argument("--time-tol", type=float, default=30.0)
    ap.add_argument("--window-utc", help="A,B ISO times bounding the recording when start is unknown")
    ap.add_argument("--types", help="comma list of ICAO type codes to keep (e.g. propeller types)")
    ap.add_argument("--elev", type=float, default=0.0, help="ground elevation, m (constant)")
    ap.add_argument("--silence-radius", type=float)
    ap.add_argument("--out", default="result.json")
    a = ap.parse_args()
    iso = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    x, fs = load_audio(a.audio)
    events = []
    for spec in a.event:
        t0, t1, f0, f1 = map(float, spec.split(":"))
        ev = fit_flyby(*extract_curve(x, fs, t0, t1, f0, f1))
        print(f"flyover {spec}: speed {ev.v:.0f} m/s, CPA slant {ev.d:.0f} m, tone {ev.f0:.1f} Hz, arrives {ev.t_arr:.1f} s")
        events.append(ev)
    tracks = load_tracks(a.adsb, set(a.types.split(",")) if a.types else None)
    print(f"{len(tracks)} aircraft tracks loaded")
    res = locate(tracks, events, tuple(map(float, a.bbox.split(","))), a.step,
                 start=iso(a.start_utc) if a.start_utc else None, time_tol=a.time_tol,
                 window=tuple(map(iso, a.window_utc.split(","))) if a.window_utc else None,
                 elev=lambda la, lo: np.full_like(np.asarray(la, float), a.elev),
                 duration=len(x) / fs, silence_radius=a.silence_radius)
    for i, r in enumerate(res, 1):
        print(f"{i:2d}. {r['lat']:.5f},{r['lon']:.5f}  rms {r['rms_hz']:.3f} Hz  start {r['start_utc']}  {r['aircraft']}")
    json.dump(res, open(a.out, "w"), indent=1)
    if res:
        print(f"best: https://www.google.com/maps?q={res[0]['lat']:.5f},{res[0]['lon']:.5f}")


if __name__ == "__main__":
    main()
