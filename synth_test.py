#!/usr/bin/env python3
"""End-to-end check on simulated data: fake aircraft, synthesised audio, blind location search."""
import sys, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import wavfile
import geoloc as g

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
TRUE = (39.8000 + rng.uniform(-.03, .03), -105.1000 + rng.uniform(-.03, .03))
T0, DUR, FS = 1_750_000_000.0, 1100.0, 4000
X = g.lla_to_ecef(TRUE[0], TRUE[1], 0.0)
M_LAT, M_LON = 111320.0, 111320.0 * np.cos(np.radians(TRUE[0]))


def straight(icao, t_cpa, offset, alt, speed, heading, span=900):
    """Level straight track passing `offset` m (horizontal) from TRUE at t_cpa, ADS-B every 5 s."""
    t = np.arange(t_cpa - span, t_cpa + span, 5.0)
    h = np.radians(heading)
    e = offset * np.cos(h) + speed * (t - t_cpa) * np.sin(h) + rng.normal(0, 5, len(t))
    n = -offset * np.sin(h) + speed * (t - t_cpa) * np.cos(h) + rng.normal(0, 5, len(t))
    return g.Track(icao, t, TRUE[0] + n / M_LAT, TRUE[1] + e / M_LON, np.full(len(t), alt))


heard = [(straight("aaa001", T0 + 140, 800, 900, 55, 70), 85.0),
         (straight("aaa002", T0 + 140 + 920, -1200, 1200, 65, 200), 78.0)]
decoys = [straight(f"d{i:05d}", T0 + rng.uniform(-3000, 4000), rng.uniform(-25e3, 25e3),
                   rng.uniform(500, 2500), rng.uniform(40, 80), rng.uniform(0, 360)) for i in range(80)]
tracks = [h[0] for h in heard] + decoys

# synthesise what a microphone at TRUE records: delayed, Doppler-shifted propeller tones + noise
ta = T0 + np.arange(int(DUR * FS)) / FS
coarse = T0 + np.arange(0, DUR + 0.1, 0.05)
audio = rng.normal(0, 0.3, len(ta)) + 0.2 * np.sin(2 * np.pi * 60 * (ta - T0))
for tr, f0 in heard:
    te = coarse.copy()
    for _ in range(8):
        te = coarse - np.linalg.norm(tr.pos(te) - X, axis=-1) / g.C_SOUND
    delay = CubicSpline(coarse, coarse - te)(ta)
    amp = 1500.0 / (delay * g.C_SOUND)
    for h, w in ((1, 1.0), (2, 0.5), (3, 0.3)):
        audio += amp * w * np.sin(2 * np.pi * f0 * h * (ta - T0 - delay))
wavfile.write("synth.wav", FS, (audio / np.abs(audio).max() * 30000).astype(np.int16))

x, fs = g.load_audio("synth.wav")
events = []
for (t0, t1, fa, fb), (tr, f0) in zip([(60, 230, 65, 105), (980, 1100, 60, 100)], heard):
    ev = g.fit_flyby(*g.extract_curve(x, fs, t0, t1, fa, fb))
    k = np.argmin(np.linalg.norm(tr.xyz - X, axis=1))
    print(f"fit: v={ev.v:5.1f} m/s d={ev.d:6.0f} m f0={ev.f0:.2f} Hz arr={ev.t_arr:7.1f} s ({len(ev.t)} pts) | "
          f"truth d={np.linalg.norm(tr.xyz[k] - X):.0f} m f0={f0}")
    events.append(ev)

bbox = (TRUE[0] - 0.18, TRUE[1] - 0.23, TRUE[0] + 0.18, TRUE[1] + 0.23)      # ~40 x 40 km
for label, kw in (("unknown time, two flyovers", dict(window=(T0 - 3600, T0 + 5400))),
                  ("known time +-30 s", dict(start=T0 + 12.0))):
    t = time.time()
    res = g.locate(tracks, events, bbox, 200.0, **kw)
    print(f"\n{label}: {len(res)} ranked candidates in {time.time() - t:.0f} s")
    for i, r in enumerate(res[:5], 1):
        err = np.hypot((r["lat"] - TRUE[0]) * M_LAT, (r["lon"] - TRUE[1]) * M_LON)
        print(f" {i}. {r['lat']:.5f},{r['lon']:.5f} rms {r['rms_hz']:.3f} Hz, {err:6.0f} m from truth, "
              f"clock err {r['start_unix'] - T0:+.1f} s, {r['aircraft']}")
