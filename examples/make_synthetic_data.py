# ============================================================================
# make_synthetic_data.py  —  generate synthetic ml-ready data for the demo
# ============================================================================
# Purpose:  Write fake route<i>_ml_ready.csv files with the exact schema the
#           dashboard builder expects (see ../docs/DATA_FORMATS.md), so the
#           dashboard pipeline can be run end-to-end without any real data.
# Outputs:  route1_ml_ready.csv, route2_ml_ready.csv in the current directory.
# Usage:    python make_synthetic_data.py
# Requires: numpy, pandas
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no real data.
# ============================================================================
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
PARTICIPANTS = [f"P{n:02d}" for n in range(1, 11)]   # 10 synthetic operators
PROBES_PER_ROUTE = 12
ROUTES = [1, 2]


def make_route(route):
    rows = []
    for pid in PARTICIPANTS:
        op_skill = rng.normal(0, 1)                       # stable per-operator effect
        for q in range(1, PROBES_PER_ROUTE + 1):
            dwell = float(np.clip(rng.beta(2, 2) + 0.15 * op_skill, 0, 1))
            # latent risk: less target dwell (and lower skill) -> more risk
            risk = 0.5 - 0.6 * dwell - 0.2 * op_skill + rng.normal(0, 0.25)
            p_err = 1 / (1 + np.exp(-4 * risk))
            accuracy = int(rng.random() > p_err)          # 1 = correct
            latency = float(np.clip(1.5 + 3.0 * risk + rng.normal(0, 0.6), 0.2, 8.0))
            confidence = int(np.clip(round(6 - 3 * risk + rng.normal(0, 0.8)), 1, 7))
            rows.append({
                "Participant_ID": pid,
                "Event": f"Q{q}",
                "Question_Type": "Sign" if q % 2 else "Animal",
                "Target_Accuracy": accuracy,
                "Target_Latency": latency,
                "Target_Confidence": confidence,
                "Before_Hazard_Encountered": int(rng.random() < 0.08),
                "Before_Dwell_Proportion_Target_Object": dwell,
                "Before_Saccade_Rate_Hz": float(np.clip(rng.normal(2.5, 0.6), 0.5, 5)),
                "Before_Mean_Saccadic_Velocity": float(np.clip(rng.normal(120, 30), 40, 300)),
                "Before_Road_Gaze_Pct": float(np.clip(rng.beta(3, 2), 0, 1)),
                "Before_Scanpath_Rate_px_s": float(np.clip(rng.normal(500, 150), 50, 1500)),
                "Before_Steer_Variance": float(abs(rng.normal(0.02, 0.01))),
                "Before_Speed_Variance": float(abs(rng.normal(0.5, 0.3))),
                "Before_Major_SRR": float(abs(rng.normal(0.3, 0.15))),
                "Before_Fine_SRR": float(abs(rng.normal(1.2, 0.4))),
                "Before_TRR": float(abs(rng.normal(0.8, 0.3))),
                "Before_Zero_Throttle_Pct": float(np.clip(rng.beta(2, 3), 0, 1)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for r in ROUTES:
        df = make_route(r)
        out = f"route{r}_ml_ready.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out}  ({len(df)} rows)")
    print("Done. Now run the dashboard builder from this directory:")
    print("  python ../dashboard/_build_dashboard_data.py")
