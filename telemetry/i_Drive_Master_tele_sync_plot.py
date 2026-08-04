# ============================================================================
# i_Drive_Master_tele_sync_plot.py  —  align event markers to telemetry, plot and adjust
# ============================================================================
# Purpose:  Align event markers with the 1000 Hz workstation driving telemetry
#           per route, plot the two together, and allow manual adjustment of the
#           alignment (turn overrides).
# Inputs:   sorted event data; driving telemetry (ROS steering / joystick)
# Outputs:  per-route/participant synchronisation plots
# Usage:    python i_Drive_Master_tele_sync_plot.py
# Requires: pandas, matplotlib
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data;
#           edit the paths in CONFIGURATION to point at your own files.
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_ROUTE = 6  # <--- Change to 2, 3, 4, 5, or 6 for other routes

# List the specific participants you want to run (e.g., [4, 12, 27]). 
# Leave it completely empty [] to run all 37 participants!
TARGET_PARTICIPANTS = []  

# Use the format "ParticipantID_RouteID" (e.g., "04_RT_R1") for specific overrides
TURN_OVERRIDES = {
    # "23_RT_R5": 1,    # Route 2  = "01_RT_R2": 14, "04_RT_R2": 1, 
}

BASE_EYETRACK_DIR = "/Users/Ryan/EyeTrack/Data_Sorted"
BASE_DRIVING_DIR = "/Users/Ryan/EyeTrack/driving_data"

# ==========================================
# 1. TIMING & SYNC LOGIC
# ==========================================
def get_video_markers(filepath):
    df = pd.read_csv(filepath)
    try:
        rec_begin = df[df['name'] == 'recording.begin'].iloc[0]['timestamp [ns]']
        wheel_in = df[df['name'] == 'wheel_in'].iloc[0]['timestamp [ns]']
        video_in = df[df['name'] == 'video_in'].iloc[0]['timestamp [ns]']
        
        return {
            'wheel': (wheel_in - rec_begin) / 1e9,
            'video': (video_in - rec_begin) / 1e9,
            'rec_begin_ns': rec_begin
        }
    except Exception as e:
        return None

def get_ros_steering_start_time(filepath, turn_index=0):
    df = pd.read_csv(filepath, header=None, names=['ros_time', 'message'])
    parsed_rows = []
    
    for _, row in df.iterrows():
        try:
            match = re.search(r'axes:\s*\[(.*?)\]', str(row['message']))
            if match:
                vals = [float(x.strip()) for x in match.group(1).split(',')]
                steer = vals[0] if len(vals) > 0 else 0.0
                parsed_rows.append({'ros_time': float(row['ros_time']), 'steer': steer})
        except: continue
            
    parsed_df = pd.DataFrame(parsed_rows)
    candidates = parsed_df[parsed_df['steer'].abs() > 0.5]
    
    if candidates.empty: return None
        
    distinct_turns = []
    last_time = -999.0
    for _, row in candidates.iterrows():
        cand_time = row['ros_time']
        if cand_time - last_time > 2.0:
            distinct_turns.append(cand_time)
        last_time = cand_time
        
    if turn_index >= len(distinct_turns):
        return None
        
    selected_turn_time = distinct_turns[turn_index]
    
    # ==========================================
    # NEW: DYNAMIC BASELINE CALIBRATION
    # ==========================================
    # Look at a 2-second window BEFORE the turn to find the true resting state
    baseline_window = parsed_df[(parsed_df['ros_time'] < selected_turn_time - 3.0) & 
                                (parsed_df['ros_time'] > selected_turn_time - 5.0)]
    
    # If the wheel is resting off-center (e.g., -0.05), this captures it
    true_resting_state = baseline_window['steer'].mean() if not baseline_window.empty else 0.0
    
    # Walk backward from the spike
    pre_spike = parsed_df[(parsed_df['ros_time'] <= selected_turn_time) & 
                          (parsed_df['ros_time'] > selected_turn_time - 3.0)].iloc[::-1]
    
    for _, prev_row in pre_spike.iterrows():
        # Stop when we get within 0.02 of the TRUE resting state, not absolute zero
        if abs(prev_row['steer'] - true_resting_state) < 0.02:
            return prev_row['ros_time']
            
    return selected_turn_time

# ==========================================
# 2. DATA PARSING & MERGING
# ==========================================
def parse_ros_file(filepath, ros_start, video_start, file_type="joystick"):
    df = pd.read_csv(filepath, header=None, names=['ros_time', 'message'])
    clean_data = []
    
    for _, row in df.iterrows():
        try:
            ros_t = float(row['ros_time'])
            msg = str(row['message'])
            synced_t = (ros_t - ros_start) + video_start
            
            if file_type == "joystick":
                match = re.search(r'axes:\s*\[(.*?)\]', msg)
                if match:
                    vals = [float(x.strip()) for x in match.group(1).split(',')]
                    steer = vals[0] if len(vals) > 0 else 0.0
                    throttle = (vals[1] + 1.0) / 2.0 if len(vals) > 1 else 0.0
                    brake = (vals[2] + 1.0) / 2.0 if len(vals) > 2 else 0.0
                    clean_data.append({'synced_timestamp_sec': synced_t, 'steering_angle': steer, 'throttle': throttle, 'brake': brake})
                    
            elif file_type == "vehicle":
                match = re.search(r'longitudinalSpeed:\s*([0-9\.\-]+)', msg)
                if match:
                    speed_raw = float(match.group(1))
                    speed_mph = speed_raw * 0.621371
                    clean_data.append({'synced_timestamp_sec': synced_t, 'speed_mph': speed_mph})
        except: continue
        
    return pd.DataFrame(clean_data).sort_values('synced_timestamp_sec')

# ==========================================
# 3. PLOTTING (HEADLESS)
# ==========================================
def create_plots(merged_df, events_df, markers, p_id, route_num, output_dir):
    key_events = events_df[events_df['name'].str.contains('Q|wheel_in|video_in|collision|Wrong_Way|X|Lost_Control', case=False, na=False)]
    colors = {'Q1_Q': 'purple', 'Q1_A': 'magenta', 'Q2_Q': 'purple', 'Q2_A': 'magenta', 
              'Q3_Q': 'purple', 'Q3_A': 'magenta', 'wheel_in': 'black', 'video_in': 'gray'}

    def draw_figure(df_plot, title, filename, time_offset=0.0):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(title, fontsize=16)

        ax1.plot(df_plot['plot_time'], df_plot['steering_angle'], color='blue', label='Steering')
        ax1.set_ylabel('Steering Input')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        ax2.plot(df_plot['plot_time'], df_plot['speed_mph'], color='darkorange', label='Speed (mph)', linewidth=2)
        ax2.set_ylabel('Speed (mph)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        ax3.plot(df_plot['plot_time'], df_plot['throttle'], color='green', alpha=0.7, label='Throttle')
        ax3.plot(df_plot['plot_time'], df_plot['brake'], color='red', alpha=0.7, label='Brake')
        ax3.set_ylabel('Pedals (0 to 1)')
        ax3.set_xlabel('Timeline (Seconds)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        # --- Draw Standard Lines ---
        for _, row in key_events.iterrows():
            event_time = ((row['timestamp [ns]'] - markers['rec_begin_ns']) / 1e9) - time_offset
            if event_time >= df_plot['plot_time'].min() and event_time <= df_plot['plot_time'].max():
                event_name = row['name']
                if re.search(r'collision|Wrong_Way|X|Lost_Control', event_name, re.IGNORECASE):
                    color = 'red'
                else:
                    color = colors.get(event_name, 'black')
                    
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(x=event_time, color=color, linestyle='--', alpha=0.8)
                    if ax == ax1:
                        ax.text(event_time, ax.get_ylim()[1]*0.9, f" {event_name}", rotation=90, 
                                verticalalignment='top', color=color, fontsize=9)
                                
        # --- NEW: Draw Disruption Zones ---
        dis_in = events_df[events_df['name'] == 'Dis_in']['timestamp [ns]'].tolist()
        dis_out = events_df[events_df['name'] == 'Dis_out']['timestamp [ns]'].tolist()
        
        for d_in, d_out in zip(dis_in, dis_out):
            start_t = ((d_in - markers['rec_begin_ns']) / 1e9) - time_offset
            end_t = ((d_out - markers['rec_begin_ns']) / 1e9) - time_offset
            
            if end_t >= df_plot['plot_time'].min() and start_t <= df_plot['plot_time'].max():
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start_t, end_t, color='gray', alpha=0.3, lw=0)
                    if ax == ax1:
                        midpoint = start_t + (end_t - start_t) / 2
                        ax.text(midpoint, ax.get_ylim()[0] + 0.1, "DISRUPTION ZONE", 
                                horizontalalignment='center', color='black', 
                                weight='bold', alpha=0.5, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    sync_df = merged_df.copy()
    sync_df['plot_time'] = sync_df['synced_timestamp_sec'] - markers['wheel']
    # sync_df = sync_df[sync_df['plot_time'] >= -10.0]
    draw_figure(sync_df, f'[{p_id} | Route {TARGET_ROUTE}] Verification: Clapperboard Pull', f"{p_id}_R{TARGET_ROUTE}_sync_verification.png", time_offset=markers['wheel'])

    vid_df = merged_df[merged_df['synced_timestamp_sec'] >= markers['video']].copy()
    vid_df['plot_time'] = vid_df['synced_timestamp_sec'] - markers['video']
    draw_figure(vid_df, f'[{p_id} | Route {TARGET_ROUTE}] Clean Driving Window', f"{p_id}_R{TARGET_ROUTE}_clean_timeline.png", time_offset=markers['video'])

# ==========================================
# MAIN BATCH LOOP
# ==========================================
def main():
    print(f"🚀 Starting Batch Telemetry Sync Process for ROUTE {TARGET_ROUTE}...")
    
    # 1. Determine who we are running
    if len(TARGET_PARTICIPANTS) > 0:
        participants_to_run = TARGET_PARTICIPANTS
        print(f"🎯 Targeted run active. Only processing participants: {participants_to_run}")
    else:
        participants_to_run = range(1, 38)
        print("🌍 Full batch run active. Processing all 37 participants.")
    
    # 2. Run the loop
    for p_num in tqdm(participants_to_run, desc=f"Processing Participants (Route {TARGET_ROUTE})", position=0):
        p_id = f"{p_num:02d}_RT"
        folder_id = f"R{TARGET_ROUTE}_{p_num:02d}"
        
        EVENTS_CSV = os.path.join(BASE_EYETRACK_DIR, folder_id, "events_with_AI.csv")
        CONDITION_DIR = os.path.join(BASE_DRIVING_DIR, f"participant_{p_id}", f"condition_{TARGET_ROUTE}")
        JOYSTICK_CSV = os.path.join(CONDITION_DIR, "_Operator_InputDevices_joystick.csv")
        VEHICLE_CSV = os.path.join(CONDITION_DIR, "_Operator_VehicleBridge_vehicle_data.csv")
        OUTPUT_CSV = os.path.join(CONDITION_DIR, f"synced_master_telemetry_{p_id}_R{TARGET_ROUTE}.csv")

        if not os.path.exists(EVENTS_CSV) or not os.path.exists(JOYSTICK_CSV):
            tqdm.write(f"⚠️ Skipping {p_id}: Missing required CSV files for Route {TARGET_ROUTE}.")
            continue
            
        markers = get_video_markers(EVENTS_CSV)
        if not markers: 
            tqdm.write(f"❌ ERROR: Missing markers (wheel_in/video_in) for {p_id}.")
            continue
        
        # Build the override key (e.g., "04_RT_R1")
        override_key = f"{p_id}_R{TARGET_ROUTE}"
        target_turn = TURN_OVERRIDES.get(override_key, 0)
        
        ros_time = get_ros_steering_start_time(JOYSTICK_CSV, turn_index=target_turn)
        
        if not ros_time: 
            tqdm.write(f"❌ ERROR: Could not find steering data for {p_id}.")
            continue
        
        joy_df = parse_ros_file(JOYSTICK_CSV, ros_time, markers['wheel'], "joystick")
        veh_df = parse_ros_file(VEHICLE_CSV, ros_time, markers['wheel'], "vehicle")
        
        merged_df = pd.merge_asof(joy_df, veh_df, on='synced_timestamp_sec', direction='nearest')
        merged_df['speed_mph'] = merged_df['speed_mph'].fillna(0.0)
        
        merged_df.to_csv(OUTPUT_CSV, index=False)
        
        events_df = pd.read_csv(EVENTS_CSV)
        create_plots(merged_df, events_df, markers, p_id, TARGET_ROUTE, CONDITION_DIR)
        
    print(f"\n✅ Processing Complete for Route {TARGET_ROUTE}!")

if __name__ == "__main__":
    main()