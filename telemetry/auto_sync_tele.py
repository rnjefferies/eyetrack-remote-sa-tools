# ============================================================================
# auto_sync_tele.py  —  align event markers to workstation joystick telemetry
# ============================================================================
# Purpose:  Using shared markers (recording begin, wheel/pedal-in), align the
#           event log with the 1000 Hz joystick telemetry and write the
#           synchronised telemetry out. Non-plotting counterpart of
#           i_Drive_Master_tele_sync_plot.py.
# Inputs:   events CSV; joystick telemetry CSV
# Outputs:  synced telemetry CSV
# Usage:    python auto_sync_tele.py
# Requires: pandas
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data;
#           edit the paths in CONFIGURATION to point at your own files.
# ============================================================================

import pandas as pd
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================
EVENTS_CSV = "/Users/Ryan/EyeTrack/Data_Sorted/R1_02/events_with_AI.csv"
JOYSTICK_CSV = "/Users/Ryan/EyeTrack/driving_data/participant_02_RT/condition_1/_Operator_InputDevices_joystick.csv"
OUTPUT_CSV = "/Users/Ryan/EyeTrack/driving_data/participant_02_RT/condition_1/synced_joystick_telemetry.csv"

def get_video_markers(filepath):
    """Gets the relative timestamps for both the wheel and the pedal from the video."""
    df = pd.read_csv(filepath)
    try:
        rec_begin = df[df['name'] == 'recording.begin'].iloc[0]['timestamp [ns]']
        wheel_in = df[df['name'] == 'wheel_in'].iloc[0]['timestamp [ns]']
        pedal_in = df[df['name'] == 'pedal_in'].iloc[0]['timestamp [ns]']
        
        rel_wheel = (wheel_in - rec_begin) / 1e9
        rel_pedal = (pedal_in - rec_begin) / 1e9
        
        return rel_wheel, rel_pedal
    except Exception as e:
        print("❌ ERROR: Could not find recording.begin, wheel_in, or pedal_in.")
        return None, None

def get_ros_steering_start_time(filepath, expected_gap_sec):
    """Finds the true 'Mark' by checking forward in time for a throttle press, then walking backward to the exact start of the turn."""
    df = pd.read_csv(filepath, header=None, names=['ros_time', 'message'])
    parsed_rows = []
    
    # Pass 1: Parse all data
    for index, row in df.iterrows():
        try:
            ros_time = float(row['ros_time'])
            message = str(row['message'])
            match = re.search(r'axes:\s*\[(.*?)\]', message)
            if match:
                axes_str = match.group(1)
                axes_vals = [float(x.strip()) for x in axes_str.split(',')]
                steer = axes_vals[0] if len(axes_vals) > 0 else 0.0
                throttle = axes_vals[1] if len(axes_vals) > 1 else -1.0
                parsed_rows.append({'ros_time': ros_time, 'steer': steer, 'throttle': throttle})
        except Exception:
            continue
            
    parsed_df = pd.DataFrame(parsed_rows)
    
    # Pass 2: Find all major steering movements (> 0.5) to act as candidates
    candidates = parsed_df[parsed_df['steer'].abs() > 0.5]
    
    # Pass 3: Test each candidate looking forward for throttle
    for idx, row in candidates.iterrows():
        candidate_time = row['ros_time']
        expected_throttle_time = candidate_time + expected_gap_sec
        
        future_window = parsed_df[
            (parsed_df['ros_time'] >= expected_throttle_time - 2.5) & 
            (parsed_df['ros_time'] <= expected_throttle_time + 2.5)
        ]
        
        # If we see a throttle press in this window, WE FOUND IT!
        if not future_window[future_window['throttle'] > -0.9].empty:
            
            # --- NEW STEP: THE LOOK-BACK ---
            # We know candidate_time is the middle of the turn (> 0.5).
            # Let's grab the 2 seconds right before this spike and walk backwards.
            pre_spike_data = parsed_df[(parsed_df['ros_time'] <= candidate_time) & 
                                       (parsed_df['ros_time'] > candidate_time - 2.0)]
            
            # Reverse the dataframe to read it backwards in time
            pre_spike_data = pre_spike_data.iloc[::-1]
            
            true_start_time = candidate_time
            for _, prev_row in pre_spike_data.iterrows():
                # The moment the wheel drops back into "resting noise" territory (< 0.05),
                # we have found the exact millisecond their hand first moved the wheel!
                if abs(prev_row['steer']) < 0.05:
                    true_start_time = prev_row['ros_time']
                    break
                    
            return true_start_time
            
    print("❌ ERROR: Checked all steering movements, none had a matching throttle press.")
    return None
def parse_and_sync_ros_joystick(filepath, ros_start_time, eye_relative_start_time):
    """Parses ROS data and shifts it to the 0-based Video Timeline."""
    df = pd.read_csv(filepath, header=None, names=['ros_time', 'message'])
    clean_data = []
    
    for index, row in df.iterrows():
        try:
            ros_time = float(row['ros_time'])
            message = str(row['message'])
            
            match = re.search(r'axes:\s*\[(.*?)\]', message)
            if match:
                axes_str = match.group(1)
                axes_vals = [float(x.strip()) for x in axes_str.split(',')]
                
                steering = axes_vals[0] if len(axes_vals) > 0 else 0.0
                throttle = axes_vals[1] if len(axes_vals) > 1 else -1.0
                brake = axes_vals[2] if len(axes_vals) > 2 else -1.0
                
                # MAGIC MATH: Zero out the ROS time, then add the video timestamp!
                synced_relative_time = (ros_time - ros_start_time) + eye_relative_start_time
                
                clean_data.append({
                    'synced_timestamp_sec': synced_relative_time, 
                    'original_ros_time': ros_time,
                    'steering_angle': steering,
                    'throttle': throttle,
                    'brake': brake
                })
        except Exception:
            continue
            
    return pd.DataFrame(clean_data)

def main():
    rel_wheel, rel_pedal = get_video_markers(EVENTS_CSV)
    if not rel_wheel: return
    
    gap_sec = rel_pedal - rel_wheel
    
    ros_time = get_ros_steering_start_time(JOYSTICK_CSV, gap_sec)
    if not ros_time: return
    
    print(f"\n--- SYNC MATH ---")
    print(f"Video 'wheel_in': {rel_wheel:.3f}s")
    print(f"Video 'pedal_in': {rel_pedal:.3f}s")
    print(f"Time Gap: {gap_sec:.3f}s")
    print(f"True ROS 'wheel_in' found at absolute time: {ros_time:.3f}s")
    
    print("\nGenerating clean, Relative-Time telemetry CSV...")
    synced_df = parse_and_sync_ros_joystick(JOYSTICK_CSV, ros_time, rel_wheel)
    synced_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Success! Saved perfectly synced telemetry to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()