# Data formats

Input and output schemas for each tool, extracted from the code. These document the
**contracts** a user must satisfy to run each tool on their own data. No data is included
in the repository; these are format specifications, not files.

Notes that recur below:
- Timestamps written by the tools are in **seconds** (`*_sec` / `... (s)`); Pupil Labs
  exports use **nanoseconds** in a `timestamp [ns]` column.
- Some values are **study-specific** and must be replaced for reuse; these are flagged
  **(study-specific)**.

---

## recapp — query delivery and response recording

**Inputs**
- `labels.csv` — the query set. Columns: `Label`, `Audio1`, `Audio2`, `Audio3`
  (a query label and up to three prompt-audio filenames to play).
- Prompt-audio `.wav` files referenced by `labels.csv` (in `audio/`).
- Entered in the GUI: `Participant ID` (format `XX_OL` / `XX_RT`) and
  `Condition` (one of `A_R, A_L, B_R, B_L, C_R, C_L`). **(study-specific)** — the ID and
  condition vocabularies are validated in code and must be edited for other designs.

**Outputs**
- `flagged_events.csv` — columns: `Participant ID`, `Condition`, `Event ID`,
  `Question Timestamp (s)`, `Answer Timestamp (s)`, `Time Difference (s)`,
  `Answer Accuracy`, `Confidence`.
- `collisions.csv` — columns: `Participant ID`, `Condition`, `Collision Timestamp (s)`,
  `Event ID`.
- One response `.wav` per session.

## wavstomp — speech-segment detection and Q/A timing

**Inputs**
- Session `.wav` files.
- `flagged_events.csv` (uses `Participant ID`, `Condition`, `Question Timestamp (s)`).

**Outputs**
- Segment CSV — columns: `Participant ID`, `Condition`, `Event ID`,
  `Question Timestamp (s)`, `Answer Timestamp (s)`, `Time Difference (s)`.
- Annotated waveform plots.

---

## event_logging — Whisper-based response logging and synchronisation

**Inputs**
- Per-participant folders under a data root (default `Data_Sorted/`), each containing a
  scene video (`*.mp4`) and a response recording (`*.wav`, excluding `*_temp`/`*_boosted`).
- `flagged_events.csv` — uses `Participant ID`, `Route`, and question timestamps.
- `collisions.csv` — uses `Participant ID`, `Route`, `Collision Timestamp (s)`.
- Spoken **anchor word** for synchronisation: `"mark"` (configurable via `ANCHOR_WORD`).
- Expected-answer keyword map **(study-specific)**: e.g. `speed0→zero … speed5→five`,
  `frogs→frog`, `duck→duck`. Replace with your own query/answer vocabulary.

**Outputs**
- `<folder>_custom_events.csv` (the synchronised per-participant event log; consumed
  downstream as the events file). Columns: `name` (e.g. `Q<n>_Q`, `Q<n>_A`, `collision`),
  `timestamp_sec`.

**Requires** `openai-whisper` and `ffmpeg` (via `moviepy`).

---

## gaze (AI_GazeV5) — gaze-to-scene mapping and AOI dwell

**Inputs** — a single JSON **config** with these keys:
- `input_files`: `video_file`, `fixations_file`, `events_file`, `scene_camera_file`,
  `saccades_file` (default `saccades.csv`), `blinks_file` (default `blinks.csv`),
  `gaze_file` (default `gaze.csv`).
- `settings`: `ai_model_path` (a YOLO `.pt` model **trained on your own scene objects** —
  **study-specific**, not shippable), `use_existing_tracking_data`, `recalibrate_coordinates_id`.
- `event_analysis`: `{enabled, query_codes}`.
- `advanced_metrics`: `{enabled, pre_event_window_s (default 5.0), post_event_window_s (default 5.0)}`.
- `manual_overrides`, `headless_overrides`, `metadata`.

Referenced data files. These are the export files produced by **Pupil Labs** software (from the wearable eye tracker used in the study, the Pupil Labs Invisible); the column names below follow the Pupil Labs export format:
- `fixations.csv` — includes `timestamp_sec`, `fixation id`.
- `events.csv` — `name`, `timestamp [ns]`; must contain the `recording.begin` marker.
- `saccades.csv` — `timestamp_sec`, `amplitude [px]`, `mean velocity [px/s]`.
- `blinks.csv` — `timestamp_sec`.
- `gaze.csv` — raw gaze samples.
- `scene_camera.json` — `camera_matrix` and distortion coefficients.
- scene video (`.mp4`), and the trained YOLO model (`.pt`).

**Outputs** (per video): `*_incident_metrics.csv`, `*_aoi_overall_metrics.csv`,
`*_aoi_question_interactions.csv`, a timeline CSV, a glances-timeline CSV, an ML-ready
feature CSV, and a tracking-data JSON cache.

**Requires** `opencv-python`, `ultralytics` (YOLO).

---

## sync (Manual_Sync_Tool) — manual video/audio sync and event labelling

**Inputs** — scene video (`.mp4`), audio recording (`.wav`), and an event CSV
(`events.csv` / `<folder>_custom_events.csv` / `flagged_events.csv`).

**Outputs** — an updated event CSV with synchronised, hand-labelled markers (AI-detected
events preserved).

**Requires** `PyQt5`.

---

## telemetry (auto_sync_tele, i_Drive_Master_tele_sync_plot)

**Inputs**
- Marker/event CSV (Pupil `events.csv`): `name`, `timestamp [ns]`; markers used:
  `recording.begin`, `wheel_in`, `pedal_in`, `video_in`.
- Joystick/ROS telemetry CSV: **two columns, no header** — `ros_time`, `message`; each
  `message` contains a ROS `Joy` string with `axes: [steer, throttle, ...]`
  (`axes[0]`=steering, `axes[1]`=throttle). Logged at ~1000 Hz. **(study-specific)** to the
  drive-by-wire rig's ROS output format.
- `i_Drive` additionally uses per-participant `TURN_OVERRIDES` and a `TARGET_ROUTE`
  setting **(study-specific)**.

**Outputs** — `synced_joystick_telemetry.csv` (auto_sync); per-route/participant
synchronisation plots (i_Drive).

---

## dashboard (_build_dashboard_data → sa_dashboard_app_v3)

**Input to the builder** — `route<i>_ml_ready.csv`, the model-ready per-probe dataset
(one file per route). Required columns:
- Identifiers/outcomes: `Participant_ID`, `Route`, `Event`, `Question_Type`
  (e.g. `Sign`, `Animal`), `Target_Accuracy`, `Target_Latency`, `Target_Confidence`,
  `Before_Hazard_Encountered`.
- Behavioural features (the 11-feature "Gaze and Control" set):
  `Before_Saccade_Rate_Hz`, `Before_Dwell_Proportion_Target_Object`,
  `Before_Mean_Saccadic_Velocity`, `Before_Road_Gaze_Pct`, `Before_Scanpath_Rate_px_s`,
  `Before_Steer_Variance`, `Before_Speed_Variance`, `Before_Major_SRR`, `Before_Fine_SRR`,
  `Before_TRR`, `Before_Zero_Throttle_Pct`.

> `route<i>_ml_ready.csv` is produced by the study's feature-extraction / analysis
> pipeline (which merges the gaze, telemetry, and event streams). That pipeline is **not
> part of this repository**; a reuser would supply an equivalent per-probe table with the
> columns above.

**Outputs of the builder** (consumed by the app):
- `SA_Dashboard_Data.csv` — per-probe: `Participant_ID`, `Route`, `evn`, `probe_label`,
  `seq`, `Question_Type`, `Target_Accuracy/Latency/Confidence`, `target_dwell`, the three
  calibrated dials (`error_dial`, `latency_dial`, `unsure_dial`) with `_recent`/`_prev`
  variants, `attn_risk*`, `recent_flag/sustained/level`, `conf_prompt/confirmed`,
  `overconf_flag`, and the outcome flags `is_error`, `is_delayed`, `is_unsure`,
  `any_fail`, `lapse`.
- `SA_Dashboard_Ops.csv` — operator-level dial means, base-rate lift, triage score/rank.
- `SA_Dashboard_Meta.csv` — calibrated thresholds (`amber_thr`, `red_thr`, …).

**Requires** `pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost` (builder); `dash`, `plotly` (app).
