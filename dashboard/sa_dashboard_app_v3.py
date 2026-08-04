# ============================================================================
# sa_dashboard_app_v3.py  —  operator SA-state monitor (live demo dashboard)
# ============================================================================
# Purpose:  Dash/Plotly web app presenting the layered SA-state indicator: an
#           objective diagnosis layer (error and latency dials with a triggered
#           manual confidence read-out) and a per-operator triage layer with a
#           recent-state (EWMA) view, trend, and persistence. A deployment-framed
#           demonstration built on probe-resolution data.
# Inputs:   SA_Dashboard_Data.csv, SA_Dashboard_Ops.csv, SA_Dashboard_Meta.csv
#           (built by _build_dashboard_data.py; derived data, not included here)
# Usage:    python sa_dashboard_app_v3.py   then open http://127.0.0.1:8050
# Requires: dash, plotly, pandas, numpy, scikit-learn
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import pandas as pd, numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx

D  = pd.read_csv('SA_Dashboard_Data.csv')
OP = pd.read_csv('SA_Dashboard_Ops.csv')
M  = pd.read_csv('SA_Dashboard_Meta.csv').iloc[0]
AMBER, RED = float(M['amber_thr']), float(M['red_thr'])

# AMBER/RED are the SMOOTHED BAND's own recall-targeted cut-points (~85% / ~50% recall on
# attn_risk_recent): they drive the trigger and the traffic-light read-out, which read the band.
# The confidence check judges a SINGLE probe, so its per-probe clause (line ~78) compares against
# the composite's OWN 50%-recall line, not the band RED -- the EWMA compresses the scale, so the
# two rulers differ (they coincide numerically on this sample only because attn_risk is quantised).

from sklearn.metrics import precision_recall_curve as _prc
_pp, _pr, _pt = _prc(D['any_fail'], D['attn_risk'])
RED_PROBE = float(_pt[np.argmin(np.abs(_pr[:-1] - 0.50))])
EB, FB, UB = float(M['err_base']), float(M['lat_base']), float(M['uns_base'])
SPAN = int(M['recent_span'])

C_ERR, C_LAT, C_UNS = '#e4572e', '#3a7ca5', '#7b9e3f'   # error / latency / unsure
C_UP, C_DOWN = '#c1121f', '#2e8b57'                      # trend: rising risk / settling
C_AMBER, C_RED, C_BG, C_PANEL, C_TX = '#f4a259', '#c1121f', '#0f1722', '#16202e', '#dce6f0'
LEVEL_COL = {'Green': '#3a9d6e', 'Amber': C_AMBER, 'Red': C_RED}
CASE_OP, CASE_RT, CASE_Q = '20_RT', 3, 3
CASE_LABEL = f'R{CASE_RT}Q{CASE_Q}'

# ---------------------------------------------------------------------------
# v2 confidence-prompt policy: SCHEDULED checkpoints (start + mid) + Red warnings.
# Computed here (not in the shared build) so v1's frozen data are untouched -- v2 has
# its own dataframe copy, so we just overwrite the confidence columns in place.
#   * a manual confidence rating is requested at ~2nd probe (baseline), mid-run, and on
#     any Red warning. Combined this is ~3 prompts/operator (~26% of probes) -- still
#     intermittent, not continuous polling.
#   * a LOW reading (<=4) at ANY prompt escalates the recent-state EWMA (injects the
#     confirmed-failure level, then decays). Two escalation classes are kept distinct:
#     "confirmed" (objective-Red + low, ~67% fail) vs "sched_low" (objective-quiet + low, ~57% fail).
#   * a HIGH reading (>=6) at a Red warning raises the overconfidence flag (unchanged).
# Baseline-relative interpretation (reading a score as a drop from the operator's own
# norm) is deferred -- it needs more per-operator training data (noted for the report).

CONF_LEVEL = float(M['conf_level']) if 'conf_level' in M.index else 0.68
LO_CONF, HI_CONF = 4, 6
LAPSE_LEVEL = 0.90   # hard-rule level for an overt control lapse (a hazard event -> Red outright)

def _augment_confidence_v2(df):
    df = df.sort_values(['Participant_ID', 'seq']).reset_index(drop=True)
    g = df.groupby('Participant_ID')
    seq_in = g.cumcount()
    n_op = g['seq'].transform('size')
    df['scheduled'] = (seq_in == np.minimum(1, n_op - 1)) | (seq_in == (n_op // 2))
    red_now  = df['attn_risk_recent'] >= RED                  # band Red at this probe (state)
    red_prev = df.groupby('Participant_ID')['attn_risk_recent'].shift(1).fillna(0.0) >= RED
    
    # PROMPT POLICY C: request a rating only when the Red warning PERSISTS (>=2 probes in a row),
    # not on a one-off flicker. Fewer, cleaner prompts (~3.8/op vs 5.0), and it sharpens the
    # confirmed signal (sustained Red is more genuinely risky); matches the persistence-aware design.
    
    sustained_red = red_now & red_prev
    obj_red  = red_now | (df['attn_risk'] >= RED_PROBE - 1e-9)  # classification: risk elevated AT this probe (per-probe ruler)
    low, high = df['Target_Confidence'] <= LO_CONF, df['Target_Confidence'] >= HI_CONF
    df['is_red']        = sustained_red.astype(int)           # trigger reason: sustained-Red warning vs scheduled
    df['conf_prompt']   = (df['scheduled'] | sustained_red).astype(int)
    df['sched_prompt']  = (df['scheduled'] & ~sustained_red).astype(int)
    prompt = df['conf_prompt'] == 1
    
    # the confidence check is a PER-PROBE judgement (the operator is rating THIS answer), so
    # classification compares confidence to the objective risk at this probe -- the recent-state
    # OR the single probe at Red -- not only the smoothed recent-state.
    
    df['conf_confirmed'] = (prompt & obj_red & low).astype(int)   # elevated + agrees      (~64% fail)
    df['overconf_flag']  = (prompt & obj_red & high).astype(int)  # elevated + confident   (silent-miss risk)
    df['conf_sched_low'] = (prompt & ~obj_red & low).astype(int)  # objective quiet + low  (aware risk)
    df['conf_escalate']  = ((df['conf_confirmed'] == 1) | (df['conf_sched_low'] == 1)).astype(int)
    
    # HARD RULE (schematic): an overt pre-query control lapse (hazard encountered) forces the
    # state to Red regardless of the soft dials -- a categorical hazard event, not a calibrated
    # probability. It injects a hard-Red level that the recent-state EWMA then carries and decays.
    
    conf_inj  = np.where(df['conf_escalate'] == 1, CONF_LEVEL, 0.0)
    lapse_inj = np.where(df['lapse'] == 1, LAPSE_LEVEL, 0.0)
    df['risk_aug'] = np.maximum.reduce([df['attn_risk'].to_numpy(dtype=float), conf_inj, lapse_inj])
    df['attn_risk_recent_adj'] = df.groupby('Participant_ID')['risk_aug'].transform(
        lambda x: x.ewm(span=SPAN, adjust=False).mean())
    
    # OVERCONFIDENCE HOLD ("warning stands"): an unacknowledged objective-Red does not decay
    # below Red while the operator stays confident. This is a FLOOR on the displayed state
    # (not injected into the EWMA, so it holds the current reading without inflating later
    # probes) -- motivated by overconfident probes being ~2x enriched for actual hazards.
    
    oc = df['overconf_flag'] == 1
    df.loc[oc, 'attn_risk_recent_adj'] = np.maximum(df.loc[oc, 'attn_risk_recent_adj'], RED)
    df['attn_risk_recent_adj_prev'] = df.groupby('Participant_ID')['attn_risk_recent_adj'].shift(1)
    df['recent_level_adj'] = np.where(df['attn_risk_recent_adj'] >= RED, 'Red',
                              np.where(df['attn_risk_recent_adj'] >= AMBER, 'Amber', 'Green'))
    # persistence on the OPERATIVE (adjusted) recent state, so the diagnosis text matches the band
    df['recent_flag_adj'] = (df['attn_risk_recent_adj'] >= AMBER).astype(int)
    prev_flag = df.groupby('Participant_ID')['recent_flag_adj'].shift(1).fillna(0).astype(int)
    df['recent_sustained_adj'] = ((df['recent_flag_adj'] == 1) & (prev_flag == 1)).astype(int)
    return df

D = _augment_confidence_v2(D)

def _augment_ops_v2(D, OP):
    # AUGMENTED TRIAGE: rank operators by the same evidence the live view uses -- objective
    # dials + overt lapses + confidence -- but with CALIBRATED aggregate weights. Each event
    # class is injected at its EMPIRICAL failure rate, NOT the 0.90 hard-Red DISPLAY level
    # (which would let rare hazards dominate the mean and rank no better than objective).
    # Lifts Spearman(triage, fail-rate) from ~0.58 to ~0.66 on this sample.
    
    fr = lambda m: float(D.loc[m, 'any_fail'].mean()) if m.any() else 0.0
    lvl_conf, lvl_sched, lvl_lapse = fr(D['conf_confirmed'] == 1), fr(D['conf_sched_low'] == 1), fr(D['lapse'] == 1)
    inj_conf = np.where(D['conf_confirmed'] == 1, lvl_conf,
                        np.where(D['conf_sched_low'] == 1, lvl_sched, 0.0))
    inj_lap  = np.where(D['lapse'] == 1, lvl_lapse, 0.0)
    cal = np.maximum.reduce([D['attn_risk'].to_numpy(dtype=float), inj_conf, inj_lap])
    tri = D.assign(_cal=cal).groupby('Participant_ID')['_cal'].mean().rename('triage_aug')
    OP = OP.merge(tri, on='Participant_ID', how='left')
    OP['rank_aug'] = OP['triage_aug'].rank(ascending=False, method='first').astype(int)
    return OP

OP = _augment_ops_v2(D, OP)

def default_cursor(pid):
    """Live 'current moment': the worked-case probe for 20_RT, else the latest probe."""
    s = D[D['Participant_ID'] == pid].sort_values('seq')
    if pid == CASE_OP and (s['probe_label'] == CASE_LABEL).any():
        return CASE_LABEL
    return s['probe_label'].iloc[-1]

def row_at(pid, cursor):
    s = D[(D['Participant_ID'] == pid) & (D['probe_label'] == cursor)]
    return s.iloc[0] if len(s) else D[D['Participant_ID'] == pid].sort_values('seq').iloc[-1]

# ----- team triage board (left, always on) --------------------------------
def triage_board(selected):
    o = OP.sort_values('triage_aug').reset_index(drop=True)
    cols = [C_ERR if m == 'error' else C_LAT for m in o['dominant_mode']]
    lw   = [3 if p == selected else 0 for p in o['Participant_ID']]
    lc   = ['#ffffff' if p == selected else 'rgba(0,0,0,0)' for p in o['Participant_ID']]
    fig = go.Figure(go.Bar(
        x=o['triage_aug'], y=o['Participant_ID'], orientation='h',
        marker=dict(color=cols, line=dict(color=lc, width=lw)),
        customdata=np.stack([o['rank_aug'], o['fails'], o['dominant_mode'], o['lapses'], o['attn']], axis=-1),
        hovertemplate='<b>%{y}</b>  (rank %{customdata[0]})<br>triage score %{x:.2f}'
                      '<br>%{customdata[1]} failures · %{customdata[3]} lapse(s) · %{customdata[2]}-led'
                      '<br>objective-only score %{customdata[4]:.2f}<extra></extra>'))
    fig.add_vline(x=AMBER, line=dict(color=C_AMBER, dash='dot', width=1))
    fig.add_vline(x=RED,   line=dict(color=C_RED,   dash='dot', width=1))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
        margin=dict(l=8, r=8, t=82, b=28), height=650, font=dict(color=C_TX, size=10),
        title=dict(text='Team triage board<br>'
                        '<span style="font-size:10px;color:#9fb3c8">calibrated: dials + lapses + confidence'
                        '<br>bar colour = dominant mode · click to drill in</span>',
                   font=dict(size=13), y=0.98, yanchor='top'),
        xaxis=dict(title='triage score (dials + hazards + confidence)', gridcolor='#243447'),
        yaxis=dict(tickfont=dict(size=8)), showlegend=False, bargap=0.25)
    return fig

# ----- recent-state gauges (diagnosis) ------------------------------------
# Two OBJECTIVE, continuously-streamed risk dials (error, latency) + the manual,
# triggered CONFIDENCE read-out. The predicted "unsure" dial is retired: it was the
# weakest detector (ROC ~0.60) and near-redundant with error -- confidence resists
# prediction from behaviour, so it is captured directly from the operator instead.
C_CONF_LO, C_CONF_HI, C_CONF_MUTE = '#c1121f', '#3a9d6e', '#4a5a6a'

def gauge(value, prev, base, color, title):
    has_delta = prev is not None and not (isinstance(prev, float) and np.isnan(prev))
    return go.Indicator(
        mode='gauge+number+delta' if has_delta else 'gauge+number',
        value=value, number=dict(font=dict(size=20), valueformat='.2f'),
        delta=(dict(reference=round(float(prev), 2), valueformat='.2f',
                    increasing=dict(color=C_UP), decreasing=dict(color=C_DOWN),
                    font=dict(size=12)) if has_delta else None),
        title=dict(text=title, font=dict(size=12)),
        gauge=dict(axis=dict(range=[0, 1], tickwidth=1, tickcolor=C_TX),
                   bar=dict(color=color, thickness=0.75), bgcolor=C_PANEL, borderwidth=0,
                   steps=[dict(range=[0, base], color='#22303f')],
                   threshold=dict(line=dict(color='#ffffff', width=2), thickness=0.85, value=base)))

def confidence_gauge(conf, prompted, verdict):
    # manual confidence read-out (1-7). Lit with a verdict colour whenever a prompt fired
    # (scheduled checkpoint or Red warning); muted when no check was requested.
    conf = float(conf)
    if not prompted:
        col, sub = C_CONF_MUTE, "not requested"
    elif verdict == 'confirmed':
        col, sub = C_CONF_LO, "CONFIRMED · warning stands"
    elif verdict == 'overconfident':
        col, sub = C_AMBER, "OVERCONFIDENT · flag"
    elif verdict == 'sched_low':
        col, sub = C_CONF_LO, "scheduled check · LOW — escalated"
    elif verdict == 'clear':
        col, sub = (C_CONF_HI if conf >= HI_CONF else C_CONF_MUTE), "scheduled check · clear"
    else:
        col, sub = C_CONF_HI, "acknowledged"
    return go.Indicator(
        mode='gauge+number', value=conf,
        number=dict(font=dict(size=20, color=(C_TX if prompted else '#7d93a8')),
                    suffix='/7', valueformat='.0f'),
        title=dict(text=f"CONFIDENCE<br><span style='font-size:9px'>{sub}</span>", font=dict(size=12)),
        gauge=dict(axis=dict(range=[0, 7], tickwidth=1, tickcolor=C_TX, dtick=1),
                   bar=dict(color=col, thickness=0.75), bgcolor=C_PANEL, borderwidth=0,
                   steps=[dict(range=[0, 4], color='#3a2330')],   # low-confidence (<=4) zone
                   threshold=dict(line=dict(color='#ffffff', width=2), thickness=0.85, value=4.5)))

def dials_figure(pid, cursor):
    r = row_at(pid, cursor)
    prompted = int(r.get('conf_prompt', 0)) == 1
    verdict = ('confirmed'     if int(r.get('conf_confirmed', 0)) == 1 else
               'overconfident' if int(r.get('overconf_flag', 0)) == 1 else
               'sched_low'     if int(r.get('conf_sched_low', 0)) == 1 else
               'clear'         if int(r.get('sched_prompt', 0)) == 1 else 'ack')
    fig = go.Figure()
    fig.add_trace(gauge(r['error_dial_recent'],  r['error_dial_prev'],  EB, C_ERR,
                        f"ERROR<br><span style='font-size:9px'>recent · trend</span>"))
    fig.add_trace(gauge(r['latency_dial_recent'], r['latency_dial_prev'], FB, C_LAT,
                        f"LATENCY<br><span style='font-size:9px'>recent · trend</span>"))
    fig.add_trace(confidence_gauge(r['Target_Confidence'], prompted, verdict))
    # explicit padded domains (not an edge-to-edge grid) so the 0.2 / 0.8 axis-tick
    # labels at the semicircle edges are not clipped at the panel boundary
    domains = [(0.05, 0.30), (0.38, 0.62), (0.70, 0.95)]
    for i, (x0, x1) in enumerate(domains):
        fig.data[i].domain = dict(x=[x0, x1], y=[0, 1])
    fig.update_layout(template='plotly_dark', paper_bgcolor=C_BG, height=172,
                      margin=dict(l=6, r=6, t=52, b=6), font=dict(color=C_TX))
    return fig

# ----- session timeline (drill-in, scrubbable) ----------------------------
# Two failure TYPES are shown on separate strips beneath the axis (they are distinct
# outcomes and map to different dials): an accuracy failure (wrong answer) and a
# latency failure (delayed response). A probe can be both, so they get their own rows.
FAIL_Y_ERR = -0.05   # accuracy failure (wrong answer) -> matches the error dial
FAIL_Y_LAT = -0.10   # latency failure (delayed response) -> matches the latency dial

def timeline_figure(pid, cursor, show_conf=True):
    # show_conf: draw the triggered confidence-check layer (adjusted band + confirmed/
    # overconfident markers). On in the live app; OFF for the manuscript figure export,
    # which documents the objective-dials-only monitor.
    s = D[D['Participant_ID'] == pid].sort_values('seq').reset_index(drop=True)
    x = s['probe_label']
    fig = go.Figure()
    fig.add_hrect(y0=RED, y1=1,      fillcolor=C_RED,   opacity=0.10, line_width=0)
    fig.add_hrect(y0=AMBER, y1=RED,  fillcolor=C_AMBER, opacity=0.10, line_width=0)
    
    # single recent-state band = the OPERATIVE state. In the live app this is the adjusted EWMA
    # (objective dials + lapse hard rule + confidence checks); for the manuscript figure
    # (show_conf=False) it is the objective-only EWMA. Escalations show via the event markers.
    
    band = s['attn_risk_recent_adj'] if show_conf else s['attn_risk_recent']
    fig.add_trace(go.Scatter(x=x, y=band, mode='lines', name='recent risk',
                             line=dict(color='#9fb3c8', width=6), opacity=0.30, hoverinfo='skip'))
    for col, c, nm in [('error_dial', C_ERR, 'error'), ('latency_dial', C_LAT, 'latency')]:
        fig.add_trace(go.Scatter(x=x, y=s[col], mode='lines+markers', name=nm,
                                 line=dict(color=c, width=2), marker=dict(size=6)))
    # failures split by type, on strips BENEATH the axis (no relation to the risk y-scale)
    fe = s[s['is_error'] == 1]
    if len(fe):
        fig.add_trace(go.Scatter(x=fe['probe_label'], y=[FAIL_Y_ERR] * len(fe),
            mode='markers', marker=dict(symbol='x', size=11, color=C_ERR),
            name='accuracy fail (wrong)', hoverinfo='skip'))
    fd = s[s['is_delayed'] == 1]
    if len(fd):
        fig.add_trace(go.Scatter(x=fd['probe_label'], y=[FAIL_Y_LAT] * len(fd),
            mode='markers', marker=dict(symbol='x', size=11, color=C_LAT),
            name='latency fail (delayed)', hoverinfo='skip'))
    lp = s[s['lapse'] == 1]
    if len(lp):
        fig.add_trace(go.Scatter(x=lp['probe_label'], y=[0.02]*len(lp), mode='markers',
            marker=dict(symbol='triangle-up', size=11, color='#b388eb'), name='overt lapse', hoverinfo='skip'))
    # triggered confidence-check outcomes (only fire at a Red warning). Markers sit ON the
    # displayed recent-state line: a CONFIRMED reply rides the escalated (dotted) line, an
    # OVERCONFIDENT reply rides the un-escalated objective band (the warning stands, not raised).
    empty = s.iloc[0:0]
    # scheduled checks that came back clear: small hollow markers on the recent band, so the
    # viewer can see WHERE checks happened (vs no check at all)
    sb = s[(s['sched_prompt'] == 1) & (s['conf_sched_low'] == 0) &
           (s['overconf_flag'] == 0) & (s['conf_confirmed'] == 0)] if show_conf else empty
    if len(sb):
        fig.add_trace(go.Scatter(x=sb['probe_label'], y=sb['attn_risk_recent_adj'], mode='markers',
            marker=dict(symbol='circle-open', size=11, color='#9fb3c8', line=dict(width=2)),
            name='scheduled check: clear', hoverinfo='skip'))
    # escalations (low reading) ride the adjusted line: Red-triggered "confirmed" (green
    # diamond) and scheduled "sched_low" (orange diamond)
    cc = s[s['conf_confirmed'] == 1] if show_conf else empty
    if len(cc):
        fig.add_trace(go.Scatter(x=cc['probe_label'], y=cc['attn_risk_recent_adj'], mode='markers',
            marker=dict(symbol='diamond', size=13, color='#2e8b57', line=dict(color='#ffffff', width=1.5)),
            name='conf. check: confirmed (Red)', hoverinfo='skip'))
    sl = s[s['conf_sched_low'] == 1] if show_conf else empty
    if len(sl):
        fig.add_trace(go.Scatter(x=sl['probe_label'], y=sl['attn_risk_recent_adj'], mode='markers',
            marker=dict(symbol='diamond', size=13, color='#e4a11b', line=dict(color='#ffffff', width=1.5)),
            name='scheduled check: low', hoverinfo='skip'))
    oc = s[s['overconf_flag'] == 1] if show_conf else empty
    if len(oc):
        fig.add_trace(go.Scatter(x=oc['probe_label'], y=oc['attn_risk_recent_adj'], mode='markers',
            marker=dict(symbol='star', size=15, color='#c1121f', line=dict(color='#ffffff', width=1.5)),
            name='conf. check: overconfident', hoverinfo='skip'))
    # current-moment cursor
    fig.add_vline(x=cursor, line=dict(color='#ffffff', width=1.5))
    fig.add_annotation(x=cursor, y=1.02, yref='y', text='current', showarrow=False,
                       xanchor='right', xshift=-6, font=dict(size=9, color='#ffffff'))
    if pid == CASE_OP and cursor == CASE_LABEL:
        fig.add_annotation(x=cursor, y=0.72, text='under-flagged error;<br>recent state still Amber',
                           showarrow=True, arrowcolor='#ffffff', font=dict(size=9, color='#ffffff'),
                           ax=45, ay=-35, xanchor='left', align='left')
    fig.add_hline(y=RED,   line=dict(color=C_RED,   dash='dot', width=1))
    fig.add_hline(y=AMBER, line=dict(color=C_AMBER, dash='dot', width=1))
    fig.add_annotation(text=f'{pid} — click a probe to set the current moment', xref='paper', yref='paper',
                       x=1.0, y=-0.30, xanchor='right', yanchor='top', showarrow=False,
                       font=dict(size=10, color='#9fb3c8'))
    # generous top margin + bottom-anchored legend so the (multi-row) legend sits ABOVE the
    # plot and never covers the escalation peak (a lapse hard-Red reaches ~0.9)
    fig.update_layout(template='plotly_dark', paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
        height=400, margin=dict(l=40, r=12, t=112, b=70), font=dict(color=C_TX, size=10),
        yaxis=dict(title='risk dial (0–1)', range=[FAIL_Y_LAT - 0.04, 1.05], gridcolor='#243447',
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   zeroline=True, zerolinecolor='#8a97a5', zerolinewidth=1.2),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, xanchor='left', font=dict(size=9)))
    return fig

# ----- diagnosis text: current status + persistence + worked case ---------
def diagnosis(pid, cursor):
    o = OP[OP['Participant_ID'] == pid].iloc[0]
    r = row_at(pid, cursor)
    # operative (adjusted) state, so the text matches the timeline band and gauges
    level = r['recent_level_adj']
    sustained = int(r['recent_sustained_adj']) == 1
    flagged = int(r['recent_flag_adj']) == 1
    status = ('SUSTAINED caution' if sustained else
              ('caution (isolated so far)' if flagged else 'clear'))
    status_col = LEVEL_COL.get(level, C_TX) if flagged else '#3a9d6e'
    mode = o['dominant_mode']
    mtxt = ('accuracy-type (errors; co-occur with more erratic speed)' if mode == 'error'
            else 'latency-type (delayed responses; a slower, scanning profile)')
    lines = [
        html.H4(f"{pid}", style={'margin': '0 0 2px 0', 'color': C_TX}),
        html.Div(f"triage rank {int(o['rank_aug'])} of {len(OP)} (objective-only rank {int(o['rank'])})  ·  "
                 f"{int(o['fails'])} failures / {int(o['probes'])} probes  ·  {int(o['lapses'])} overt lapse(s)",
                 style={'fontSize': '11px', 'color': '#9fb3c8', 'marginBottom': '8px'}),
        html.Div([html.Span(f"Current moment {cursor}: ", style={'color': '#9fb3c8', 'fontSize': '12px'}),
                  html.B(f"{level} · {status}", style={'color': status_col, 'fontSize': '13px'})]),
        html.Div(f"recent-state risk {r['attn_risk_recent_adj']:.2f} (trailing ~{SPAN} probes, incl. rules + checks); "
                 f"the single probe alone reads {r['attn_risk']:.2f}.",
                 style={'fontSize': '11px', 'color': '#7d93a8', 'margin': '4px 0 8px 0'}),
        html.Div([html.Span("Leaning toward: ", style={'color': '#9fb3c8'}),
                  html.B(mtxt, style={'color': C_ERR if mode == 'error' else C_LAT})],
                 style={'fontSize': '12px'}),
        html.Div(f"error lift x{o['e_lift']:.1f} · latency lift x{o['f_lift']:.1f}  "
                 f"(session-level: which mode is unusually elevated)",
                 style={'fontSize': '10px', 'color': '#7d93a8'}),
        html.Div("the two dials share their main driver (low target dwell), so this is a tendency, "
                 "not a distinct cause", style={'fontSize': '10px', 'color': '#7d93a8', 'fontStyle': 'italic'}),
    ]
    if int(r.get('lapse', 0)) == 1:
        lines.append(html.Div("Overt control lapse (hazard) at this probe — hard rule forces Red "
                              "regardless of the dials; the recent state is held elevated and decays.",
                              style={'fontSize': '11px', 'color': C_RED, 'fontWeight': 'bold', 'margin': '6px 0 0 0'}))
    if int(r.get('conf_prompt', 0)) == 1:
        conf = int(r['Target_Confidence'])
        trig = 'Red warning' if int(r.get('is_red', 0)) == 1 else 'scheduled checkpoint'
        if int(r['conf_confirmed']) == 1:
            ctxt, ccol = (f"Confidence check ({trig}): operator reported {conf}/7 — "
                          f"CONFIRMED. Recent state held elevated: confirmed-Red probes fail ~67% of the time.", C_RED)
        elif int(r['overconf_flag']) == 1:
            ctxt, ccol = (f"Confidence check ({trig}): operator reported {conf}/7 — "
                          f"OVERCONFIDENCE FLAG. The model predicts high risk for this answer but the operator "
                          f"is confident — the 'looked but failed to see' silent-miss risk. The warning stands.", C_AMBER)
        elif int(r['conf_sched_low']) == 1:
            ctxt, ccol = (f"Confidence check ({trig}): operator reported {conf}/7 — "
                          f"LOW on a routine check, below the Red trigger. Recent state escalated: scheduled "
                          f"low readings fail ~57% of the time — an aware risk the objective dials missed.", C_RED)
        else:
            ctxt, ccol = (f"Confidence check ({trig}): operator reported {conf}/7 — "
                          f"clear, no escalation.", '#9fb3c8')
        lines.append(html.Div(ctxt, style={'fontSize': '11px', 'color': ccol, 'fontWeight': 'bold',
                                           'margin': '6px 0 0 0'}))
    if pid == CASE_OP and cursor == CASE_LABEL:
        lines += [html.Hr(style={'borderColor': '#243447'}),
            html.Div("Worked case — R3Q3 (Sign):", style={'fontWeight': 'bold', 'color': '#ffd166', 'fontSize': '12px'}),
            html.Div(f"Long dwell on the sign ({r['target_dwell']*5:.1f}s), latency {r['Target_Latency']:.1f}s, "
                     f"confidence {int(r['Target_Confidence'])}/7 — read the colour but not the lettering, so WRONG. "
                     f"The single-probe error dial sat at only {r['error_dial']:.2f} (below Amber): a long, "
                     f"attentive-looking dwell reads as good awareness.",
                     style={'fontSize': '11px', 'color': C_TX, 'marginTop': '4px'}),
            html.Div(f"But the recent-state view was already at a sustained Amber ({r['attn_risk_recent_adj']:.2f}): "
                     f"the operator had a delayed response on the previous probe, so a continuous monitor would have been in "
                     f"caution going in. That is the dip-before-incident this design is meant to catch.",
                     style={'fontSize': '11px', 'color': C_AMBER, 'marginTop': '4px'})]
    return html.Div(lines, style={'padding': '12px 16px', 'backgroundColor': C_PANEL,
                                  'borderRadius': '8px', 'border': '1px solid #243447'})

# ----- app layout ---------------------------------------------------------
app = Dash(__name__)
app.title = 'SA Operator Monitor (v2)'
opts = [{'label': f"{r.Participant_ID}  (rank {int(r['rank'])}, {int(r['fails'])} fails)",
         'value': r.Participant_ID} for _, r in OP.sort_values('rank').iterrows()]

app.layout = html.Div(style={'backgroundColor': C_BG, 'minHeight': '100vh',
                             'fontFamily': 'Inter, system-ui, sans-serif', 'color': C_TX, 'padding': '14px 18px'},
    children=[
    dcc.Store(id='cursor'),
    html.Div([
        html.H2('SA Operator Monitor', style={'margin': '0', 'fontWeight': '700'}),
        html.Div('Live demo · two objective dials (error, latency) + a triggered manual confidence check + '
                 'supervisor triage + drill-in.  Gauges show the RECENT STATE (trailing EWMA) with a trend '
                 'arrow and a persistence badge — "is this operator dipping now, and is it sustained?"  The '
                 'objective dials are a soft aid (per-probe ROC ~0.65-0.7); scheduled checkpoints and Red '
                 'warnings trigger a manual confidence read-out that confirms the alert (~67% of confirmed '
                 'warnings fail) or flags an unacknowledged overconfident risk. Overt lapses force Red (hard rule).',
                 style={'fontSize': '12px', 'color': '#9fb3c8'})], style={'marginBottom': '10px'}),
    html.Div(style={'display': 'flex', 'gap': '16px'}, children=[
        html.Div(dcc.Graph(id='board', config={'displayModeBar': False}), style={'flex': '0 0 360px'}),
        html.Div(style={'flex': '1'}, children=[
            html.Div([html.Span('Operator: ', style={'fontSize': '13px', 'color': '#9fb3c8'}),
                      dcc.Dropdown(id='pick', options=opts, value=CASE_OP, clearable=False,
                                   style={'width': '320px', 'color': '#111'})],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '8px'}),
            dcc.Graph(id='gauges', config={'displayModeBar': False}),
            html.Div(style={'marginTop': '4px'}, children=[
                dcc.Graph(id='timeline', config={'displayModeBar': False}),
                html.Div(id='diag', style={'marginTop': '8px'})])])])])

@app.callback(Output('cursor', 'data'),
              Input('pick', 'value'), Input('timeline', 'clickData'))
def set_cursor(pid, click):
    if ctx.triggered_id == 'timeline' and click:
        return click['points'][0]['x']
    return default_cursor(pid)          # operator changed or initial load

@app.callback(Output('board', 'figure'), Output('gauges', 'figure'),
              Output('timeline', 'figure'), Output('diag', 'children'),
              Input('pick', 'value'), Input('cursor', 'data'))
def refresh(pid, cursor):
    if not cursor or cursor not in set(D.loc[D['Participant_ID'] == pid, 'probe_label']):
        cursor = default_cursor(pid)
    return triage_board(pid), dials_figure(pid, cursor), timeline_figure(pid, cursor), diagnosis(pid, cursor)

@app.callback(Output('pick', 'value'), Input('board', 'clickData'))
def from_board(click):
    if not click:
        return no_update
    return click['points'][0]['y']

if __name__ == '__main__':
    app.run(debug=False, port=8051)
