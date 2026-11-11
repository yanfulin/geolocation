# Audio geolocation from aircraft Doppler curves

Finds where a recording was made from the pitch curves of propeller planes flying past,
following the method in https://youtu.be/UytYrhYLOKI.

## Files
- `geoloc.py` - the method: curve tracking, flyby fit, candidate search, Doppler modelling, scoring.
- `extract_adsb.py` - turns one day of the public adsb.lol history archive into a compact file of continental-US tracks.
- `run_real.py` - runs the search on the author's reference recording (September 8, 2026).
- `synth_test.py` - end-to-end check on simulated planes and synthesised audio.

## How it works
1. **Measure** each flyover: follow the tone in the spectrogram, fit speed and closest-approach distance.
2. **Match**: on a 200 m grid, list every plane pass with a similar speed and distance, counting the time the sound needs to arrive.
3. **Time filter**: keep grid points where two such passes are separated by the same gap as in the recording (or that fit a known recording time).
4. **Score**: model the exact pitch curve each point would hear from the real flight path and compare it with the measured curve. One shared clock shift is allowed; the gap between flyovers stays locked.
5. **Report** the best point per pair of aircraft, polished off-grid.
6. **Confirm** (an addition to the video's method): for each leading point, check whether other planes that
   passed nearby in the same hour also show up in the spectrogram along the curve that point predicts.

## Result on the reference recording
`result_northeast.json`: best point 40.8758, -73.5245 (Oyster Bay, New York), recording start 20:17:29 UTC,
aircraft N22NY (PC-12) then N93CQ (King Air 350). The video's author reports 40.87, -73.5.
On curve misfit alone this point ranked second of about 72,000 matches (0.59 Hz against 0.52 Hz for a point
in Connecticut); the confirm step moves it to first, with two further planes found in the audio.

## Run it
```
pip install numpy scipy            # ffmpeg must be on PATH
python synth_test.py               # self-test, no downloads

# flight data for one day (about 4 GB download, keeps a 2.6 GB file)
B=https://github.com/adsblol/globe_history_2026/releases/download/v2026.09.08-planes-readsb-prod-0/v2026.09.08-planes-readsb-prod-0.tar
curl -sSL $B.aa $B.ab | python extract_adsb.py        # writes conus_0908.pkl

python run_real.py search  ref_audio.webm conus_0908.pkl ne.json 38.5,-77.5,43.5,-69.5
python run_real.py confirm ref_audio.webm conus_0908.pkl final.json ne.json
```
The search step takes about 6 minutes on 4 cores for the US Northeast. It can be run on several
smaller boxes and the outputs merged in the confirm step.
For another recording use `python geoloc.py --help` (give each flyover as `start:end:fmin:fmax`).

## Limits
- Ground elevation is taken as a constant (sea level by default); there is no terrain model yet.
- Wind and temperature are ignored (speed of sound fixed at 343 m/s).
- Flyover windows and frequency bands are picked by eye from the spectrogram.
- The video's "nothing else flew close" filter is replaced by the confirm step above.
- Only the US Northeast has been searched, not the whole continental US.
- Jets are removed with a built-in list of type codes, which is not complete.
