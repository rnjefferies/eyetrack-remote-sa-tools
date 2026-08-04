# ============================================================================
# _build_dashboard_data.py  —  build the cached input tables for the SA dashboard
# ============================================================================
# Purpose:  Recompute the validated, calibrated SA risk dials once and write the
#           dashboard's input tables. Uses fixed seeds for reproducibility.
# Inputs:   the study's model-ready dataset (derived data, not included here)
# Outputs:  SA_Dashboard_Data.csv (probe-level), SA_Dashboard_Ops.csv
#           (operator-level), SA_Dashboard_Meta.csv (calibrated thresholds)
# Usage:    python _build_dashboard_data.py
# Requires: pandas, numpy, scikit-learn, xgboost
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import pandas as pd, numpy as np, warnings, os, random
warnings.filterwarnings('ignore')
# Determinism guard: fixed seeds + single-threaded XGB so the dials/states are bit-reproducible
os.environ['PYTHONHASHSEED'] = '0'; random.seed(42); np.random.seed(42)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb

frames = []
for i in range(1, 7):
    d = pd.read_csv(f"route{i}_ml_ready.csv"); d['Route'] = i; frames.append(d)
df = pd.concat(frames, ignore_index=True).fillna(0.0)
df = df[df['Target_Latency'] >= 0].reset_index(drop=True)
for c in [c for c in df.columns if 'Before_Raw_Dwell' in c]:
    df[c.replace('Raw_Dwell', 'Dwell_Proportion')] = df[c] / 5.0
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].reset_index(drop=True)

FEATURES = ['Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object',
    'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct', 'Before_Scanpath_Rate_px_s',
    'Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR',
    'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct']
X = df[FEATURES]; groups = df['Participant_ID']; cv = GroupKFold(5)
y_err = (1 - df['Target_Accuracy']).astype(int)
y_lat = (df['Target_Latency'] >= 3.5).astype(int)
y_uns = (df['Target_Confidence'] <= 4).astype(int)
y_any = ((y_err == 1) | (y_lat == 1)).astype(int)
df['is_error'], df['is_delayed'], df['is_unsure'] = y_err.values, y_lat.values, y_uns.values
df['any_fail'] = y_any.values
df['lapse'] = (df['Before_Hazard_Encountered'] > 0).astype(int)
df['target_dwell'] = df['Before_Dwell_Proportion_Target_Object']

def oof(y, kind):
    if kind == 'xgb':
        w = (y == 0).sum() / max((y == 1).sum(), 1)
        m = xgb.XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42, n_jobs=1,
                              max_depth=2, learning_rate=0.03, n_estimators=100)
    elif kind == 'rf':
        m = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=3, random_state=42)
    else:
        m = Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                      ('m', LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42))])
    return cross_val_predict(m, X, y, groups=groups, cv=cv, method='predict_proba')[:, 1]

cal = lambda p, y: IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(p, y).transform(p)
# Keep the RAW out-of-fold predictions for honest discrimination reporting: in-sample
# isotonic calibration merges some wrong-ordered pairs into ties and nudges ROC up, so
# the reported ROC must be measured on the raw OOF scores, not the calibrated dials.
raw_err, raw_lat, raw_uns = oof(y_err, 'xgb'), oof(y_lat, 'rf'), oof(y_uns, 'lr')
df['error_dial']  = cal(raw_err, y_err)
df['latency_dial'] = cal(raw_lat, y_lat)
df['unsure_dial'] = cal(raw_uns, y_uns)
# Cap the displayed dials below literal certainty: isotonic's top plateau is 1.0 but it is
# built from a 2-case bin, so a calibrated 1.00 is small-sample over-confidence, not a real
# probability. Clip so no gauge ever reads as a guaranteed failure (display honesty only;
# the cap sits far above the Red threshold, so ranking and Green/Amber/Red states are unchanged).
DIAL_CAP = 0.90
for c in ['error_dial', 'latency_dial', 'unsure_dial']:
    df[c] = df[c].clip(upper=DIAL_CAP)
# Triage signal = worst of the two OBJECTIVE dials (max, no dilution)
df['attn_risk'] = df[['error_dial', 'latency_dial']].max(axis=1)

err_base, lat_base = y_err.mean(), y_lat.mean()

# session order within operator
df['evn'] = df['Event'].astype(str).str.extract(r'(\d+)').astype(int)
df = df.sort_values(['Participant_ID', 'Route', 'evn']).reset_index(drop=True)
df['probe_label'] = 'R' + df['Route'].astype(int).astype(str) + 'Q' + df['evn'].astype(str)
df['seq'] = df.groupby('Participant_ID').cumcount()

# RECENT-STATE (live deployment view): a trailing EWMA over each operator's probe
# sequence stands in for the continuous rolling estimate a deployed monitor would form
# from streaming gaze/telemetry (no SA probes in deployment). span ~ last 3 probes.
SPAN = 3
ewm = lambda c: df.groupby('Participant_ID')[c].transform(lambda x: x.ewm(span=SPAN, adjust=False).mean())
for c in ['error_dial', 'latency_dial', 'unsure_dial', 'attn_risk']:
    df[c + '_recent'] = ewm(c)

# thresholds on the recent-state band that is actually read out (recall-targeted, as in
# the manuscript). The traffic light reads the smoothed band, so the cut-points are set
# on the band's own distribution: setting them on the per-probe composite and applying
# them to the band would mis-target the recall, since the EWMA compresses the scale.
prec, rec, thr = precision_recall_curve(df['any_fail'], df['attn_risk_recent'])
AMBER = float(thr[np.argmin(np.abs(rec[:-1] - 0.85))])
RED   = float(thr[np.argmin(np.abs(rec[:-1] - 0.50))])
AMBER = min(AMBER, RED)

# trend = change in the recent-state signal vs the previous probe (per operator)
for c in ['error_dial', 'latency_dial', 'unsure_dial', 'attn_risk']:
    df[c + '_prev'] = df.groupby('Participant_ID')[c + '_recent'].shift(1)
# persistence on the recent-state attn signal, using the same Amber line
df['recent_flag'] = (df['attn_risk_recent'] >= AMBER).astype(int)
df['recent_prev_flag'] = df.groupby('Participant_ID')['recent_flag'].shift(1).fillna(0).astype(int)
df['recent_sustained'] = ((df['recent_flag'] == 1) & (df['recent_prev_flag'] == 1)).astype(int)
df['recent_level'] = np.where(df['attn_risk_recent'] >= RED, 'Red',
                       np.where(df['attn_risk_recent'] >= AMBER, 'Amber', 'Green'))

# ---- CONFIDENCE READ-OUT (triggered manual check) -------------------------
# Deployment framing: the error/latency dials stream continuously from gaze +
# telemetry, but a manual confidence rating is NOT a continuous input -- it is
# REQUESTED only when the continuous monitor raises a Red warning (the higher-
# precision, recall~0.50 alert). The operator's reply is an independent, high-
# precision human signal that resolves the low-precision Red into two classes
# and feeds back into the recent-state EWMA:
#   * confirmed  (reported confidence <= 4): the operator agrees something is
#     off. Empirically ~0.68 of confirmed-Red probes are true failures, so the
#     reply is injected into the risk series at that level, keeping the recent
#     state elevated (slower EWMA decay) -- a confirmed deficit persists.
#   * overconfident (reported confidence >= 6): the objective Red is NOT
#     acknowledged. The signal is left untouched (self-report never stands a
#     warning DOWN), but a standing overconfidence flag is raised -- the "looked
#     but failed to see" silent-miss risk a supervisor must not clear.
HI_CONF, LO_CONF = 6, 4
df['conf_prompt']    = (df['attn_risk_recent'] >= RED).astype(int)
df['conf_confirmed'] = ((df['conf_prompt'] == 1) & (df['Target_Confidence'] <= LO_CONF)).astype(int)
df['overconf_flag']  = ((df['conf_prompt'] == 1) & (df['Target_Confidence'] >= HI_CONF)).astype(int)
# data-derived confirmation level = empirical failure rate of a confirmed-Red probe
_cm = (df['conf_prompt'] == 1) & (df['Target_Confidence'] <= LO_CONF)
CONF_LEVEL = float(df.loc[_cm, 'is_error'].mean()) if _cm.any() else RED
df['risk_aug'] = np.where(df['conf_confirmed'] == 1,
                          np.maximum(df['attn_risk'], CONF_LEVEL), df['attn_risk'])
df['attn_risk_recent_adj'] = df.groupby('Participant_ID')['risk_aug'].transform(
    lambda x: x.ewm(span=SPAN, adjust=False).mean())
df['attn_risk_recent_adj_prev'] = df.groupby('Participant_ID')['attn_risk_recent_adj'].shift(1)
df['recent_level_adj'] = np.where(df['attn_risk_recent_adj'] >= RED, 'Red',
                          np.where(df['attn_risk_recent_adj'] >= AMBER, 'Amber', 'Green'))

# operator-level
op = df.groupby('Participant_ID').agg(
        probes=('any_fail', 'size'),
        error_dial=('error_dial', 'mean'), latency_dial=('latency_dial', 'mean'),
        unsure_dial=('unsure_dial', 'mean'), attn=('attn_risk', 'mean'),
        lapses=('lapse', 'sum'), n_error=('is_error', 'sum'),
        n_delayed=('is_delayed', 'sum'), fails=('any_fail', 'sum')).reset_index()
op['fail_rate'] = op['fails'] / op['probes']
# base-rate LIFT-corrected dominant mode (fixes the all-error artefact)
op['e_lift'] = op['error_dial'] / err_base
op['f_lift'] = op['latency_dial'] / lat_base
op['dominant_mode'] = np.where(op['e_lift'] >= op['f_lift'], 'error', 'latency')
op = op.sort_values('attn', ascending=False).reset_index(drop=True)
op['rank'] = np.arange(1, len(op) + 1)
op['cum_fail_frac'] = op['fails'].cumsum() / op['fails'].sum()

rho, pval = spearmanr(op['attn'], op['fail_rate'])

meta = pd.DataFrame([{
    'amber_thr': AMBER, 'red_thr': RED, 'err_base': err_base, 'lat_base': lat_base,
    'uns_base': y_uns.mean(), 'n_probes': len(df), 'n_ops': op.shape[0],
    'recent_span': SPAN, 'spearman_rho': rho, 'spearman_p': pval,
    'conf_level': CONF_LEVEL, 'hi_conf': HI_CONF, 'lo_conf': LO_CONF,
    # ROC on the RAW out-of-fold scores (not the calibrated dials) to avoid in-sample inflation
    'roc_error': roc_auc_score(y_err, raw_err),
    'roc_latency': roc_auc_score(y_lat, raw_lat),
    'roc_unsure': roc_auc_score(y_uns, raw_uns),
    'roc_attn_any': roc_auc_score(y_any, np.maximum(raw_err, raw_lat)),
}])

keep = ['Participant_ID', 'Route', 'evn', 'probe_label', 'seq', 'Question_Type',
        'Target_Accuracy', 'Target_Latency', 'Target_Confidence', 'target_dwell',
        'error_dial', 'latency_dial', 'unsure_dial', 'attn_risk',
        'error_dial_recent', 'latency_dial_recent', 'unsure_dial_recent', 'attn_risk_recent',
        'error_dial_prev', 'latency_dial_prev', 'unsure_dial_prev', 'attn_risk_prev',
        'recent_flag', 'recent_sustained', 'recent_level',
        'conf_prompt', 'conf_confirmed', 'overconf_flag',
        'attn_risk_recent_adj', 'attn_risk_recent_adj_prev', 'recent_level_adj',
        'is_error', 'is_delayed', 'is_unsure', 'any_fail', 'lapse']
df[keep].to_csv('SA_Dashboard_Data.csv', index=False)
op.to_csv('SA_Dashboard_Ops.csv', index=False)
meta.to_csv('SA_Dashboard_Meta.csv', index=False)

print("WROTE SA_Dashboard_Data.csv (%d probes), SA_Dashboard_Ops.csv (%d ops), SA_Dashboard_Meta.csv"
      % (len(df), len(op)))
print("Amber>=%.3f Red>=%.3f | base error=%.3f latency=%.3f" % (AMBER, RED, err_base, lat_base))
print("dominant mode (lift-corrected): error=%d latency=%d"
      % ((op.dominant_mode == 'error').sum(), (op.dominant_mode == 'latency').sum()))
print("Spearman(triage,fail_rate)=%.2f p=%.3f | dial ROC err/lat/uns = %.2f/%.2f/%.2f"
      % (rho, pval, meta.roc_error[0], meta.roc_latency[0], meta.roc_unsure[0]))
n_p, n_cf, n_of = int(df['conf_prompt'].sum()), int(df['conf_confirmed'].sum()), int(df['overconf_flag'].sum())
err_cf = int(df.loc[df['conf_confirmed']==1, 'is_error'].sum())
err_of = int(df.loc[df['overconf_flag']==1, 'is_error'].sum())
print("CONFIDENCE CHECK (Red-triggered): prompts=%d (%.0f%%) | confirmed=%d (%d/%d errors, %.0f%%) "
      "| overconfident=%d (%d actual errors) | inject level=%.2f"
      % (n_p, 100*n_p/len(df), n_cf, err_cf, n_cf, 100*err_cf/max(n_cf,1), n_of, err_of, CONF_LEVEL))
print("Top 5 triage:", op.head(5)[['rank','Participant_ID','attn','dominant_mode','fails']].to_dict('records'))
print("20_RT rank:", int(op.loc[op.Participant_ID.astype(str).str.contains('20_RT'),'rank'].iloc[0]))
