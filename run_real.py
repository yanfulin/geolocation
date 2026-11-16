#!/usr/bin/env python3
"""Run the search on the reference recording (youtu.be/4CxVxuMVkMc) with the Sept 8 2026 flight data.

  search : python run_real.py search  AUDIO ADSB.pkl OUT.json lat0,lon0,lat1,lon1
  confirm: python run_real.py confirm AUDIO ADSB.pkl FINAL.json OUT1.json [OUT2.json ...]
"""
import sys, json, time
import numpy as np, geoloc as g

def main():
    mode, audio, adsb, out = sys.argv[1:5]
    WINDOWS = [(2092, 2215, 82, 185), (3040, 3095, 108, 182)]     # the two flyovers, read off the spectrogram
    DAY = 1788825600.0                                            # 2026-09-08 00:00 UTC
    x, fs = g.load_audio(audio)
    events = [g.fit_flyby(*g.extract_curve(x, fs, *w)) for w in WINDOWS]
    for ev in events:
        print(f"flyover: {ev.v:.0f} m/s, CPA slant {ev.d:.0f} m, tone {ev.f0:.1f} Hz, heard at {ev.t_arr:.1f} s", flush=True)
    tracks = g.load_tracks(adsb)

    if mode == "search":
        bbox = tuple(map(float, sys.argv[5].split(",")))
        tracks = [t for t in tracks if t.lat.max() > bbox[0] - .2 and t.lat.min() < bbox[2] + .2
                  and t.lon.max() > bbox[1] - .2 and t.lon.min() < bbox[3] + .2]
        print(len(tracks), "non-jet tracks touch the search area", flush=True)
        t = time.time()
        res = g.locate(tracks, events, bbox, 200.0, window=(DAY, DAY + 86400), top=30, verbose=True)
        print(f"{g.locate.n_candidates} candidate (point, plane pair) matches scored in {time.time() - t:.0f} s")
    else:
        res = sorted((c for f in sys.argv[5:] for c in json.load(open(f))), key=lambda c: c["rms_hz"])[:12]
        for c in res:                                             # independent check with other planes
            c["confirmed_by"] = g.confirm(x, fs, tracks, c, events)
        res.sort(key=lambda c: (-len(c["confirmed_by"]), c["rms_hz"]))
    for i, r in enumerate(res[:12], 1):
        extra = f"  other planes also heard: {len(r['confirmed_by'])}" if "confirmed_by" in r else ""
        print(f"{i:2d}. {r['lat']:.5f},{r['lon']:.5f}  misfit {r['rms_hz']:.2f} Hz  start {r['start_utc'][:19]}Z  {r['aircraft']}{extra}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
