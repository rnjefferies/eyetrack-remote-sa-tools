# EyeTrack Remote-SA Tools

Software developed for the remote-operation situation-awareness (SA) studies reported in
the thesis *Measuring Situation Awareness in the Remote Operation of a Live Vehicle*.
The tools deliver SA queries during live remote driving and build a
synchronised, multi-stream dataset from gaze, audio (spoken responses), and 1000 Hz
vehicle-control telemetry.

> **No data is included in this repository.** All participant recordings, telemetry, and
> derived datasets are excluded (see `.gitignore`). Scripts contain hardcoded local paths
> in their `CONFIGURATION` sections that must be edited to point at your own data.

## Tools

### `recapp/` — query delivery and response capture
**Recapp** is a GUI application that plays the spoken SA query prompts, records the
operator's spoken response, and flags the end of each question and answer (timestamped to
CSV). **Wavstomp** (`scripts/wavstomp.py`) detects speech segments in the recordings and
derives question/answer timing (response latency).

### `event_logging/` — speech-recognition event logger
Whisper-based tools that transcribe the recorded audio, match each spoken response to the
expected answer for its query, timestamp it, and locate a spoken "mark" anchor to compute
the offset that synchronises the eye-tracking video with the audio recording.
`AI_Clapper.py` is a single-file prototype; `Batch_Synch_Script.py` batches across
operators.

### `gaze/` — gaze-to-scene mapping
`AI_GazeV5.py` maps gaze onto the scene video, defines areas of interest by object
detection (YOLO), supports per-operator gaze recalibration and manual AOI correction, and
computes area-of-interest dwell over event windows.

### `sync/` — manual stream synchronisation
`Manual_Sync_Tool.py` is a GUI to align the scene/gaze video with the audio recording on a
common timeline and to mark and label events, preserving events detected by the tools
above.

### `telemetry/` — event-marker to telemetry alignment
`i_Drive_Master_tele_sync_plot.py` aligns event markers with the 1000 Hz workstation
driving telemetry, plots them together per route, and allows manual adjustment of the
alignment. `auto_sync_tele.py` is the automated, non-plotting counterpart that writes
synced telemetry to CSV.

### `dashboard/` — operator SA-state monitor (applied prototype)
`sa_dashboard_app_v3.py` is a Dash/Plotly web app presenting the layered SA-state
indicator: an objective diagnosis layer (error and latency dials with a triggered manual
confidence read-out) and a per-operator triage layer with a recent-state (EWMA) view,
trend arrow, and persistence badge. It is a deployment-framed demonstration built on
probe-resolution data. `_build_dashboard_data.py` builds its inputs (`SA_Dashboard_*.csv`)
from the study dataset; those CSVs are derived data and are not included here, so the app
requires them to be built first.

## Requirements
Python 3. Depending on the tool, key dependencies include OpenCV, ultralytics (YOLO),
openai-whisper, moviepy, PyQt5, pandas, numpy, scipy, matplotlib, and Dash/Plotly (the
dashboard). See `recapp/requirements.txt` for the query application; install the others
per tool as needed.

## Licence
MIT. See `LICENSE`.
