# Example: run the dashboard end-to-end on synthetic data

This is a self-contained demo of the dashboard pipeline that needs **no real data**.
`make_synthetic_data.py` generates fake `route<i>_ml_ready.csv` files with the exact schema
the builder expects (see [`../docs/DATA_FORMATS.md`](../docs/DATA_FORMATS.md)), so you can
watch the whole chain run before wiring in your own data.

## Steps

From this `examples/` directory:

```bash
# 1. install the dashboard dependencies (ideally in a virtual environment)
pip install -r ../dashboard/requirements.txt

# 2. generate synthetic per-probe data (writes route1_ml_ready.csv, route2_ml_ready.csv)
python make_synthetic_data.py

# 3. build the dashboard's cached tables from that data
python ../dashboard/_build_dashboard_data.py
#    -> writes SA_Dashboard_Data.csv, SA_Dashboard_Ops.csv, SA_Dashboard_Meta.csv

# 4. launch the dashboard (open the printed http://127.0.0.1:8050)
python ../dashboard/sa_dashboard_app_v3.py
```

The generated `*.csv` files are synthetic and are ignored by git (the repository never
stores data). Delete them any time; step 2 regenerates them.

## What this shows

Steps 2–3 exercise the real modelling pipeline — grouped cross-validated dials for error,
latency, and confidence, isotonic calibration, the recent-state EWMA, and the operator
triage table — on placeholder numbers. To use your own study, replace step 2 with real
`route<i>_ml_ready.csv` files matching the schema in `docs/DATA_FORMATS.md`.
