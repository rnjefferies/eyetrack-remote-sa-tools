# ============================================================================
# AI_GazeV5.py  —  interactive gaze-to-scene mapping and AOI dwell analysis
# ============================================================================
# Purpose:  Map operators' gaze onto the scene video, define areas of interest
#           by YOLO object detection (with per-operator gaze recalibration and
#           manual box correction), classify fixations, and compute
#           area-of-interest dwell over event windows.
# Inputs:   scene video, gaze data, a JSON config, a trained YOLO model
# Outputs:  per-frame/AOI tracking data and dwell metrics
# Usage:    python AI_GazeV5.py --config <config.json>
# Requires: opencv-python, ultralytics (YOLO), numpy, pandas
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================


import cv2
import pandas as pd
from tqdm import tqdm
import collections
import sys
import json
import numpy as np
import os
import argparse
from ultralytics import YOLO

click_position = None

# --- Manual Box Variables ---
drawing = False
manual_box_start = None
manual_box_end = None

# --- Visualization Settings ---
RAW_GAZE_COLOR = (0, 255, 0)
AOI_BBOX_COLOR = (255, 0, 255)
FIXATION_COLOR_DEFAULT = (0, 0, 220)
FIXATION_COLOR_HIT = (0, 128, 255)
RECALIBRATION_DOT_COLOR = (0, 255, 255)
RECALIBRATION_CONFIRM_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FIXATION_CIRCLE_RADIUS = 20
RAW_GAZE_CIRCLE_RADIUS = 5
FIXATION_CIRCLE_THICKNESS = 3

def mouse_callback(event, x, y, flags, param):
    global click_position, drawing, manual_box_start, manual_box_end
    
    if event == cv2.EVENT_LBUTTONDOWN and param == "recalibrate":
        click_position = (x, y)
        
    elif event == cv2.EVENT_LBUTTONDOWN and param == "draw_box":
        drawing = True
        manual_box_start = (x, y)
        manual_box_end = (x, y) 
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing and param == "draw_box":
        manual_box_end = (x, y)
        
    elif event == cv2.EVENT_LBUTTONUP and param == "draw_box":
        drawing = False
        manual_box_end = (x, y)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if 'ai_model_path' not in config['settings']:
            config['settings']['ai_model_path'] = "Route2_Brain.pt" 
        if 'use_existing_tracking_data' not in config['settings']:
            config['settings']['use_existing_tracking_data'] = False
        if 'recalibrate_coordinates_id' not in config['settings']:
            config['settings']['recalibrate_coordinates_id'] = 15
        if 'event_analysis' not in config:
            config['event_analysis'] = {'enabled': False, 'query_codes': []}
        if 'advanced_metrics' not in config:
            config['advanced_metrics'] = {'enabled': False, 'pre_event_window_s': 5.0, 'post_event_window_s': 5.0}
        
        # --- Catch the overrides if they don't exist ---
        if 'manual_overrides' not in config:
            config['manual_overrides'] = []
            
        return config
    except Exception as e:
        print(f"ERROR: Configuration loading failed: {e}")
        sys.exit()

def calculate_windowed_metrics(start_time, end_time, fixations_df, saccades_df, blinks_df):
    duration_s = end_time - start_time
    if duration_s <= 0: return {}

    section_saccades = saccades_df[saccades_df['timestamp_sec'].between(start_time, end_time)]
    section_blinks = blinks_df[blinks_df['timestamp_sec'].between(start_time, end_time)]

    saccade_count = len(section_saccades)
    blink_count = len(section_blinks)

    saccade_rate_hz = saccade_count / duration_s if duration_s > 0 else 0
    blink_rate_bpm = blink_count / (duration_s / 60.0) if duration_s > 0 else 0
    
    scanpath_length_px = section_saccades['amplitude [px]'].sum() if not section_saccades.empty and 'amplitude [px]' in section_saccades.columns else 0
    mean_saccadic_velocity = section_saccades['mean velocity [px/s]'].mean() if not section_saccades.empty and 'mean velocity [px/s]' in section_saccades.columns else 0

    return {
        'Window_Duration_s': f"{duration_s:.2f}",
        'Saccade_Count': saccade_count,
        'Blink_Count': blink_count,
        'Saccade_Rate_Hz': f"{saccade_rate_hz:.2f}",
        'Blink_Rate_BPM': f"{blink_rate_bpm:.2f}",
        'Scanpath_Length_px': f"{scanpath_length_px:.2f}",
        'Mean_Saccadic_Velocity_px_s': f"{mean_saccadic_velocity:.2f}",
    }

def run_recalibration(config, fixations_df, cap, map1, map2, anchor_time=None):
    global click_position
    click_position = None
    recalibration_offset = (0, 0)
    
    valid_fixes = pd.DataFrame()
    if anchor_time is not None:
        window_start = max(0, anchor_time - 2.0)
        window_end = anchor_time + 5.0
        valid_fixes = fixations_df[
            (fixations_df['timestamp_sec'] >= window_start) & 
            (fixations_df['timestamp_sec'] <= window_end)
        ]
    
    candidate_fixes = []
    if not valid_fixes.empty:
        all_fixes = valid_fixes.sort_values(by='timestamp_sec', ascending=True)
        for _, fix in all_fixes.iterrows():
            candidate_fixes.append(fix)
    else:
        fix_id = config['settings']['recalibrate_coordinates_id']
        fix_series = fixations_df[fixations_df['fixation id'] == fix_id]
        if not fix_series.empty:
            candidate_fixes.append(fix_series.iloc[0])

    if not candidate_fixes:
        print("ERROR: No valid fixations found for calibration.")
        return recalibration_offset

    print("\n--- RECALIBRATION MODE ---")
    fps = cap.get(cv2.CAP_PROP_FPS)
    window_name = "Recalibrate Coordinates"
    
    for i, fixation in enumerate(candidate_fixes):
        fix_id = int(fixation['fixation id'])
        dur = fixation['duration_sec']
        
        frame_idx = int(fixation['timestamp_sec'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        recal_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()
        raw_x, raw_y = int(fixation['fixation x [px]']), int(fixation['fixation y [px]'])
        cv2.circle(recal_frame, (raw_x, raw_y), FIXATION_CIRCLE_RADIUS, RECALIBRATION_DOT_COLOR, FIXATION_CIRCLE_THICKNESS)
        
        cv2.putText(recal_frame, f"Fixation {fix_id} ({dur:.2f}s). Click TRUE CENTER.", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(recal_frame, f"Press [SPACE] to confirm, or [N] for Next Fixation ({i+1}/{len(candidate_fixes)})", (10, 60), FONT, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, recal_frame)
        cv2.setMouseCallback(window_name, mouse_callback, param="recalibrate")
        
        print(f"Showing Fixation {fix_id}. Press 'N' to skip, or click your target and press SPACE.")

        click_position = None 
        confirmed = False
        skip = False
        
        while True:
            display_copy = recal_frame.copy()
            if click_position: 
                cv2.circle(display_copy, click_position, FIXATION_CIRCLE_RADIUS // 2, RECALIBRATION_CONFIRM_COLOR, -1)
            cv2.imshow(window_name, display_copy)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('n') or key == ord('N'):
                skip = True
                break
            elif key == 32:
                if click_position:
                    confirmed = True
                    break
                else:
                    print("Please click on the screen first before pressing Space!")
            elif key == 27:
                break

        if confirmed:
            recalibration_offset = (click_position[0] - raw_x, click_position[1] - raw_y)
            print(f"Offset calculated: {recalibration_offset} using Fixation {fix_id}.")
            cv2.destroyAllWindows()
            return recalibration_offset
            
        if not skip and key == 27:
            break

    print("No calibration confirmed. Continuing with zero offset.")
    cv2.destroyAllWindows()
    return (0, 0)

def main(config_path):
    global drawing, manual_box_start, manual_box_end
    
    config = load_config(config_path)
    metadata, files, settings, headless, event_analysis_settings, advanced_metrics_settings = \
        config['metadata'], config['input_files'], config['settings'], \
        config['headless_overrides'], config.get('event_analysis', {'enabled': False}), \
        config.get('advanced_metrics', {'enabled': False})
        
    manual_overrides = config.get('manual_overrides', [])
    drawn_overrides = {} 
    active_cv2_trackers = {} # Smart Optical Tracker Dictionary

    DATA_DIR = files['data_directory']
    video_path = os.path.join(DATA_DIR, files['video_file'])
    fixations_path = os.path.join(DATA_DIR, files['fixations_file'])
    events_path = os.path.join(DATA_DIR, files['events_file'])
    camera_path = os.path.join(DATA_DIR, files['scene_camera_file'])
    saccades_path = os.path.join(DATA_DIR, files.get('saccades_file', 'saccades.csv'))
    blinks_path = os.path.join(DATA_DIR, files.get('blinks_file', 'blinks.csv'))
    gaze_path = os.path.join(DATA_DIR, files.get('gaze_file', 'gaze.csv'))

    output_video_dir = os.path.join('videos', metadata['participant_id'])
    os.makedirs(output_video_dir, exist_ok=True)
    
    prefix = metadata.get('output_prefix', '')
    if prefix and not prefix.endswith('_'): prefix += '_'

    OUTPUT_VIDEO_FILE = os.path.join(output_video_dir, f"{prefix}{metadata['event_id']}.mp4")
    TRACKING_DATA_FILE = os.path.join(output_video_dir, f"{prefix}tracking_data.json") 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    map1, map2 = None, None
    try:
        with open(camera_path, 'r') as f:
            cam_data = json.load(f)
            camera_matrix = np.array(cam_data['camera_matrix'], dtype=np.float32)
            dist_data = cam_data.get('dist_coeffs', cam_data.get('distortion_coefficients'))
            if dist_data is not None:
                dist_coeffs = np.array(dist_data, dtype=np.float32)
                map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (frame_width, frame_height), 5)
    except Exception as e:
        print(f"WARNING: No undistortion setup found. {e}")

    try:
        fixations_df = pd.read_csv(fixations_path)
        events_df = pd.read_csv(events_path)
        saccades_df = pd.read_csv(saccades_path)
        blinks_df = pd.read_csv(blinks_path)
        gaze_df = pd.read_csv(gaze_path)
        
        start_time_ns = events_df[events_df['name'] == 'recording.begin'].iloc[0]['timestamp [ns]']

        fixations_df['timestamp_sec'] = (fixations_df['start timestamp [ns]'] - start_time_ns) / 1e9
        fixations_df['duration_sec'] = fixations_df['duration [ms]'] / 1000.0
        saccades_df['timestamp_sec'] = (saccades_df['start timestamp [ns]'] - start_time_ns) / 1e9
        blinks_df['timestamp_sec'] = (blinks_df['start timestamp [ns]'] - start_time_ns) / 1e9
        events_df['timestamp_sec'] = (events_df['timestamp [ns]'] - start_time_ns) / 1e9
        gaze_df['timestamp_sec'] = (gaze_df['timestamp [ns]'] - start_time_ns) / 1e9
        gaze_df['fixation id'] = gaze_df['fixation id'].fillna(-1).astype(int)
    except Exception as e:
        print(f"ERROR loading data: {e}"); sys.exit()

    analysis_start_frame, analysis_end_frame = 0, total_frames
    
    video_in = events_df[events_df['name'] == 'video_in']
    video_out = events_df[events_df['name'] == 'video_out']
    if not video_in.empty and not video_out.empty:
        analysis_start_frame = max(0, int(video_in.iloc[0]['timestamp_sec'] * fps))
        analysis_end_frame = min(total_frames, int(video_out.iloc[0]['timestamp_sec'] * fps))

    analysis_start_sec = analysis_start_frame / fps

    disruptions = []
    dis_in_events = events_df[events_df['name'] == 'Dis_in'].sort_values('timestamp_sec')['timestamp_sec'].tolist()
    dis_out_events = events_df[events_df['name'] == 'Dis_out'].sort_values('timestamp_sec')['timestamp_sec'].tolist()
    
    for i in range(min(len(dis_in_events), len(dis_out_events))):
        disruptions.append((dis_in_events[i], dis_out_events[i]))

    def is_paused(current_sec):
        return any(d_in <= current_sec <= d_out for d_in, d_out in disruptions)

    anchor_event_name = settings.get('calibration_anchor_event', 'Q1_Q')
    anchor_event = events_df[events_df['name'] == anchor_event_name]
    
    anchor_time = None
    if not anchor_event.empty:
        anchor_time = anchor_event.iloc[0]['timestamp_sec']
        print(f"📍 Auto-hunting for best calibration fixations near '{anchor_event_name}'...")

    recalibration_offset = (0, 0)
    
    if not headless['use_headless_settings'] and settings['recalibrate_coordinates']:
        recalibration_offset = run_recalibration(config, fixations_df, cap, map1, map2, anchor_time)
        
        if recalibration_offset != (0, 0):
            print(f"\n💾 Auto-saving offset {recalibration_offset} to config...")
            config['headless_overrides']['recalibration_offset_x'] = int(recalibration_offset[0])
            config['headless_overrides']['recalibration_offset_y'] = int(recalibration_offset[1])
            config['settings']['recalibrate_coordinates'] = False
            config['headless_overrides']['use_headless_settings'] = True
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Config updated successfully! Next run will use these offsets automatically.")
            
    elif headless['use_headless_settings'] and not settings['recalibrate_coordinates']:
        recalibration_offset = (int(headless.get('recalibration_offset_x', 0)), int(headless.get('recalibration_offset_y', 0)))
        print(f"🤖 Loaded saved calibration offset: {recalibration_offset}")

    offset_x, offset_y = recalibration_offset

    tracking_data_history = {}
    use_saved_tracking = settings.get('use_existing_tracking_data', False)
    active_aois = []
    
    if use_saved_tracking and os.path.exists(TRACKING_DATA_FILE):
        print(f"\n--- LOADING TRACKING DATA FROM '{TRACKING_DATA_FILE}' ---")
        with open(TRACKING_DATA_FILE, 'r') as f: tracking_data_history = json.load(f)
        
        all_aoi_names = set()
        aoi_start_frames = {}
        for msec_str, data in tracking_data_history.items():
            try:
                frame_idx = int((float(msec_str) / 1000.0) * fps)
                for name in data.keys():
                    all_aoi_names.add(name)
                    if name not in aoi_start_frames or frame_idx < aoi_start_frames[name]:
                        aoi_start_frames[name] = frame_idx
            except ValueError: pass
        for name in all_aoi_names:
            active_aois.append({'name': name, 'start_timestamp_sec': aoi_start_frames.get(name, analysis_start_frame) / fps})
    else:
        print(f"\n--- PASS 1: AI TRACKING WITH {settings['ai_model_path']} ---")
        ai_model = YOLO(settings['ai_model_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_start_frame)
        
        progress_bar = tqdm(total=analysis_end_frame - analysis_start_frame, desc="AI Processing", dynamic_ncols=True, leave=True, position=0)
        all_aoi_names_set = set()
        aoi_start_timestamps = {}
        aoi_frame_counts = collections.defaultdict(int)

        for frame_idx in range(analysis_start_frame, analysis_end_frame):
            ret, frame = cap.read()
            if not ret: break

            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_sec = current_msec / 1000.0
            
            true_time_s = current_sec - analysis_start_sec
            
            if is_paused(current_sec):
                tracking_data_history[f"{current_msec:.2f}"] = {} 
                progress_bar.update(1)
                continue

            processing_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame
            
            # --- JSON-DRIVEN FORWARD-FILL OVERRIDE ---
            for idx, override in enumerate(manual_overrides):
                b_start = override.get('start_time_s', 0)
                b_end = override.get('end_time_s', 0)
                override_aoi = override.get('aoi_name', 'manual_aoi')
                
                if b_start <= true_time_s <= b_end and idx not in drawn_overrides:
                    print(f"\n⏸️ AUTO-PAUSED at CSV Time: {true_time_s:.3f}s. Draw the '{override_aoi}' box!")
                    
                    cv2.imshow("Draw Override", processing_frame)
                    cv2.setMouseCallback("Draw Override", mouse_callback, param="draw_box")
                    
                    while True:
                        draw_frame = processing_frame.copy()
                        if manual_box_start and manual_box_end:
                            cv2.rectangle(draw_frame, manual_box_start, manual_box_end, (0, 255, 0), 2)
                        cv2.imshow("Draw Override", draw_frame)
                        
                        if cv2.waitKey(20) & 0xFF == 32: 
                            if manual_box_start and manual_box_end:
                                x1, y1 = manual_box_start
                                x2, y2 = manual_box_end
                                x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                                drawn_overrides[idx] = [x, y, w, h]
                                
                                # SMART TRACKER INITIALIZATION
                                try:
                                    tracker = cv2.TrackerCSRT_create()
                                except AttributeError:
                                    tracker = cv2.TrackerMIL_create()
                                
                                tracker.init(processing_frame, (x, y, w, h))
                                active_cv2_trackers[idx] = tracker
                                
                                print(f"Box for '{override_aoi}' captured! Tracking visually until {b_end:.2f}s...")
                                cv2.destroyWindow("Draw Override")
                                break
                                    
            # Standard AI Processing
            results = ai_model.track(processing_frame, persist=True, conf=0.20, tracker="botsort.yaml", imgsz=640, verbose=False)
            current_frame_data = {}

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy() 
                track_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                best_det_per_class = {}
                for obj_idx, (cls, conf) in enumerate(zip(classes, confs)):
                    cls_id = int(cls)
                    if cls_id not in best_det_per_class or conf > best_det_per_class[cls_id]['conf']:
                        best_det_per_class[cls_id] = {'idx': obj_idx, 'conf': conf}

                valid_indices = [v['idx'] for v in best_det_per_class.values()]

                for valid_idx in valid_indices:
                    box = boxes[valid_idx]
                    track_id = track_ids[valid_idx]
                    cls = classes[valid_idx]
                    
                    cx, cy, w, h = box
                    x, y = int(cx - w / 2), int(cy - h / 2)
                    
                    base_class_name = ai_model.names[int(cls)]
                    
                    # ==========================================
                    # SPATIAL ANCHORING: THE WHEEL FIX
                    # ==========================================
                    if base_class_name in ['lwheel', 'rwheel']:
                        center_x = x + (w / 2)
                        if center_x < (frame_width / 2):
                            aoi_name = 'lwheel'
                        else:
                            aoi_name = 'rwheel'
                            
                    elif base_class_name in ['bumper', 'speed', 'map', 'name']:
                        aoi_name = base_class_name
                    else:
                        aoi_name = f"{base_class_name}_{int(track_id)}" 
                    # ==========================================
                        
                    current_frame_data[aoi_name] = [x, y, int(w), int(h)]
                    
                    all_aoi_names_set.add(aoi_name)
                    aoi_frame_counts[aoi_name] += 1
                    if aoi_name not in aoi_start_timestamps:
                        aoi_start_timestamps[aoi_name] = current_sec

            tracking_data_history[f"{current_msec:.2f}"] = current_frame_data

            # Stamp any drawn boxes into the current frame!
            for idx, override in enumerate(manual_overrides):
                b_start = override.get('start_time_s', 0)
                b_end = override.get('end_time_s', 0)
                override_aoi = override.get('aoi_name', 'manual_aoi')
                
                if b_start <= true_time_s <= b_end and idx in drawn_overrides:
                    
                    # UPDATE BOX POSITION BASED ON HEAD MOVEMENT
                    if idx in active_cv2_trackers:
                        success, new_box = active_cv2_trackers[idx].update(processing_frame)
                        if success:
                            drawn_overrides[idx] = [int(v) for v in new_box]
                    
                    if f"{current_msec:.2f}" not in tracking_data_history:
                        tracking_data_history[f"{current_msec:.2f}"] = {}
                    
                    tracking_data_history[f"{current_msec:.2f}"][override_aoi] = drawn_overrides[idx]
                    
                    all_aoi_names_set.add(override_aoi)
                    aoi_frame_counts[override_aoi] += 1
                    if override_aoi not in aoi_start_timestamps:
                        aoi_start_timestamps[override_aoi] = true_time_s

            progress_bar.update(1) 
            
        progress_bar.close()

        min_frames = 5
        valid_aois = [name for name, count in aoi_frame_counts.items() if count >= min_frames]
             
        for msec in tracking_data_history:
            tracking_data_history[msec] = {k: v for k, v in tracking_data_history[msec].items() if k in valid_aois}

        for name in valid_aois:
            active_aois.append({'name': name, 'start_timestamp_sec': aoi_start_timestamps.get(name, 0.0)})

        with open(TRACKING_DATA_FILE, 'w') as f: json.dump(tracking_data_history, f)
    
    event_windows = {}
    
    if event_analysis_settings.get('enabled', False):
        for q_code in event_analysis_settings.get('query_codes', []):
            q_event, a_event = events_df[events_df['name'] == f"{q_code}_Q"], events_df[events_df['name'] == f"{q_code}_A"]
            if not q_event.empty and not a_event.empty:
                event_windows[q_code] = {'q': q_event.iloc[0]['timestamp_sec'], 'a': a_event.iloc[0]['timestamp_sec']}

    print("\n--- Applying Dynamic Temporal Hallucination Filters ---")
    aoi_mapping = event_analysis_settings.get('aoi_mapping', {})
    BUFFER_SEC = 15.0 
    aoi_valid_windows = {}
    
    # ROUTE 3, ROUTE 4, ROUTE 5 SIGNS IMMUNITY LIST
    for base_aoi, q_list in aoi_mapping.items():
        if base_aoi in [
            'bumper', 'speed', 'map', 'lwheel', 'rwheel', 'cone', 'name', 
            'r3_construct', 'r3_rabbit', 'r3_goat', 'r3_markings',
            'r4_duck_sign', 'r4_pig', 'r4_cat', 'r4_flood', 'r5_dog', 'r5_cow',
            'r5_snow_sign', 'r5_frost_sign'
        ]: 
            continue 
            
        valid_starts = []
        valid_ends = []
        for q in q_list:
            if q in event_windows:
                valid_starts.append(event_windows[q]['q'] - BUFFER_SEC)
                valid_ends.append(event_windows[q]['a'] + BUFFER_SEC)
                
        if valid_starts and valid_ends:
            aoi_valid_windows[base_aoi] = (min(valid_starts), max(valid_ends))

    removed_counts = collections.defaultdict(int)

    for msec_str in list(tracking_data_history.keys()):
        current_sec_filter = float(msec_str) / 1000.0
        frame_boxes = tracking_data_history[msec_str]
        keys_to_remove = []
        
        for aoi_name in frame_boxes.keys():
            if aoi_name.rsplit('_', 1)[-1].isdigit():
                base_name = aoi_name.rsplit('_', 1)[0]
            else:
                base_name = aoi_name
            
            if base_name in aoi_valid_windows:
                min_sec, max_sec = aoi_valid_windows[base_name]
                if not (min_sec <= current_sec_filter <= max_sec):
                    keys_to_remove.append(aoi_name)
                    removed_counts[base_name] += 1
                    
        for k in keys_to_remove:
            del tracking_data_history[msec_str][k]

    if removed_counts:
        for base_name, count in removed_counts.items():
            print(f"Cleaned up {count} hallucinated frames for '{base_name}'.")
    else:
        print("Timeline clean! No out-of-bounds hallucinations detected.")

    print("\n--- Interpolating Dropped Bounding Boxes (Bridging Gaps) ---")
    MAX_GAP_MSEC = 1000.0 
    
    sorted_msecs = sorted(tracking_data_history.keys(), key=lambda x: float(x))
    
    all_known_aois = set()
    for msec in sorted_msecs:
        for aoi in tracking_data_history[msec].keys():
            all_known_aois.add(aoi)

    for aoi in all_known_aois:
        last_seen_msec = None
        last_seen_box = None
        
        for msec in sorted_msecs:
            current_msec_float = float(msec)
            
            if aoi in tracking_data_history[msec]:
                if last_seen_msec is not None:
                    gap_duration = current_msec_float - last_seen_msec
                    
                    if 35.0 < gap_duration <= MAX_GAP_MSEC:
                        start_box = last_seen_box
                        end_box = tracking_data_history[msec][aoi]
                        
                        for fill_msec in sorted_msecs:
                            fill_msec_float = float(fill_msec)
                            if last_seen_msec < fill_msec_float < current_msec_float:
                                weight = (fill_msec_float - last_seen_msec) / gap_duration
                                
                                interp_box = [
                                    int(start_box[0] + (end_box[0] - start_box[0]) * weight),
                                    int(start_box[1] + (end_box[1] - start_box[1]) * weight),
                                    int(start_box[2] + (end_box[2] - start_box[2]) * weight),
                                    int(start_box[3] + (end_box[3] - start_box[3]) * weight)
                                ]
                                tracking_data_history[fill_msec][aoi] = interp_box
                
                last_seen_msec = current_msec_float
                last_seen_box = tracking_data_history[msec][aoi]

    print("\n--- PASS 2: RENDERING VIDEO & CALCULATING METRICS ---")
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    report_metrics = {aoi['name']: {'dwell_time_s': 0.0, 'hit_fixation_ids': set(), 'first_timestamp': float('inf')} for aoi in active_aois}
    report_metrics['cellboundary'] = {'dwell_time_s': 0.0, 'hit_fixation_ids': set(), 'first_timestamp': float('inf')}

    raw_gaze_tracker = {aoi['name']: {'current_start': None, 'points': [], 'completed_glances': []} for aoi in active_aois}

    fixation_visualization_queue = collections.deque()
    event_windows = {}
    
    if event_analysis_settings.get('enabled', False):
        for q_code in event_analysis_settings.get('query_codes', []):
            q_event, a_event = events_df[events_df['name'] == f"{q_code}_Q"], events_df[events_df['name'] == f"{q_code}_A"]
            if not q_event.empty and not a_event.empty:
                event_windows[q_code] = {'q': q_event.iloc[0]['timestamp_sec'], 'a': a_event.iloc[0]['timestamp_sec']}

    ongoing = fixations_df[(fixations_df['timestamp_sec'] <= analysis_start_sec) & ((fixations_df['timestamp_sec'] + fixations_df['duration_sec']) > analysis_start_sec)]
    for _, fixation in ongoing.iterrows():
        fixation_visualization_queue.append(fixation)

    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_start_frame)
    render_bar = tqdm(total=analysis_end_frame - analysis_start_frame, desc="Rendering")

    for frame_idx in range(analysis_start_frame, analysis_end_frame):
        ret, frame = cap.read()
        if not ret: break
        
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_msec / 1000.0
        frame_duration_s = 1.0 / fps
        processing_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame
        
        active_q_code = None
        answered_q_code = None
        
        for q_code, window in event_windows.items():
            if window['q'] <= current_sec <= window['a']:
                active_q_code = q_code
                break
            elif window['a'] < current_sec <= (window['a'] + 2.0):
                answered_q_code = q_code
                break
        
        if active_q_code:
            header_overlay = processing_frame.copy()
            cv2.rectangle(header_overlay, (0, 0), (frame_width, 60), (0, 255, 255), -1)
            cv2.addWeighted(header_overlay, 0.5, processing_frame, 0.5, 0, processing_frame)
            cv2.putText(processing_frame, f"QUESTION ACTIVE: {active_q_code}", (frame_width//2 - 250, 40), FONT, 1.2, (0, 0, 0), 3)
            
        elif answered_q_code:
            header_overlay = processing_frame.copy()
            cv2.rectangle(header_overlay, (0, 0), (frame_width, 60), (0, 255, 0), -1)
            cv2.addWeighted(header_overlay, 0.5, processing_frame, 0.5, 0, processing_frame)
            cv2.putText(processing_frame, f"ANSWER RECORDED: {answered_q_code}", (frame_width//2 - 250, 40), FONT, 1.2, (0, 0, 0), 3)

        while fixation_visualization_queue and (fixation_visualization_queue[0]['timestamp_sec'] + fixation_visualization_queue[0]['duration_sec']) < current_sec:
            fixation_visualization_queue.popleft()
        for _, fixation in fixations_df[fixations_df['timestamp_sec'].between(current_sec, current_sec + frame_duration_s, inclusive='left')].iterrows():
             if not any(f['fixation id'] == fixation['fixation id'] for f in fixation_visualization_queue):
                 fixation_visualization_queue.append(fixation)

        if is_paused(current_sec):
            cv2.rectangle(processing_frame, (0, 0), (frame_width, 100), (0, 0, 255), -1)
            cv2.putText(processing_frame, "ANALYSIS PAUSED (DISRUPTION)", (frame_width//2 - 250, 60), FONT, 1.2, (255, 255, 255), 3)
            out.write(processing_frame)
            render_bar.update(1)
            continue

        overlay = processing_frame.copy()
        current_aoi_bboxes = tracking_data_history.get(f"{current_msec:.2f}", {})

       # =========================================================
        # JSON-DRIVEN DYNAMIC AOI PADDING
        # =========================================================
        custom_padding = settings.get('aoi_padding', {})
        
        for aoi_name in list(current_aoi_bboxes.keys()):
            # Strip the YOLO tracking ID to match the base name (e.g., 'map_14' -> 'map')
            base_name = aoi_name.rsplit('_', 1)[0] if aoi_name.rsplit('_', 1)[-1].isdigit() else aoi_name
            
            if base_name in custom_padding:
                pad_x, pad_y, pad_w, pad_h = custom_padding[base_name]
                x, y, w, h = current_aoi_bboxes[aoi_name]
                
                # Apply the specific JSON padding to this box
                current_aoi_bboxes[aoi_name] = [
                    x + pad_x, 
                    y + pad_y, 
                    w + pad_w, 
                    h + pad_h
                ]
        # =========================================================

        for aoi_name, bbox in current_aoi_bboxes.items():
            x,y,w,h = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), AOI_BBOX_COLOR, 2)
            cv2.putText(overlay, aoi_name, (x, y - 10), FONT, 0.7, AOI_BBOX_COLOR, 2)

        current_gaze_point_data = gaze_df[gaze_df['timestamp_sec'].between(current_sec, current_sec + frame_duration_s, inclusive='left')]
        current_gaze_px, current_gaze_py = None, None
        hit_aois_this_frame = set()
        
        if not current_gaze_point_data.empty:
            gaze_row = current_gaze_point_data.iloc[0]
            current_gaze_px, current_gaze_py = int(gaze_row['gaze x [px]']) + offset_x, int(gaze_row['gaze y [px]']) + offset_y
            
            for aoi_name, bbox in current_aoi_bboxes.items():
                x, y, w, h = [int(v) for v in bbox]
                if x <= current_gaze_px <= x + w and y <= current_gaze_py <= y + h:
                    hit_aois_this_frame.add(aoi_name)

        for aoi_name in raw_gaze_tracker.keys():
            if aoi_name in hit_aois_this_frame:
                if raw_gaze_tracker[aoi_name]['current_start'] is None:
                    raw_gaze_tracker[aoi_name]['current_start'] = current_sec
                    raw_gaze_tracker[aoi_name]['points'] = [(current_gaze_px, current_gaze_py)]
                else:
                    raw_gaze_tracker[aoi_name]['points'].append((current_gaze_px, current_gaze_py))
            else:
                if raw_gaze_tracker[aoi_name]['current_start'] is not None:
                    duration = current_sec - raw_gaze_tracker[aoi_name]['current_start']
                    if duration > 0:
                        xs = [p[0] for p in raw_gaze_tracker[aoi_name]['points']]
                        ys = [p[1] for p in raw_gaze_tracker[aoi_name]['points']]
                        dispersion = float(np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)) if xs else 0.0
                        
                        raw_gaze_tracker[aoi_name]['completed_glances'].append({
                            'start': raw_gaze_tracker[aoi_name]['current_start'],
                            'duration': duration,
                            'dispersion_px': dispersion
                        })
                    raw_gaze_tracker[aoi_name]['current_start'] = None
                    raw_gaze_tracker[aoi_name]['points'] = []

        gaze_hit_aoi_this_frame = False
        
        for fixation in fixation_visualization_queue:
            fix_id = fixation['fixation id']
            px = int(fixation['fixation x [px]']) + offset_x
            py = int(fixation['fixation y [px]']) + offset_y

            hit_aoi_name = next((name for name, bbox in current_aoi_bboxes.items() if bbox[0] <= px <= bbox[0]+bbox[2] and bbox[1] <= py <= bbox[1]+bbox[3]), None)

            if hit_aoi_name:
                metrics = report_metrics[hit_aoi_name]
                metrics['dwell_time_s'] += frame_duration_s

                if fix_id not in metrics['hit_fixation_ids']:
                    metrics['hit_fixation_ids'].add(fix_id)
                    metrics['first_timestamp'] = min(metrics['first_timestamp'], fixation['timestamp_sec'])

                gaze_hit_aoi_this_frame = True
            else:
                report_metrics['cellboundary']['dwell_time_s'] += frame_duration_s
            
            is_hit_any = any(fix_id in m['hit_fixation_ids'] for n, m in report_metrics.items() if n != 'cellboundary')
            cv2.circle(overlay, (px, py), FIXATION_CIRCLE_RADIUS, FIXATION_COLOR_HIT if is_hit_any else FIXATION_COLOR_DEFAULT, -1 if is_hit_any else FIXATION_CIRCLE_THICKNESS)

        if current_gaze_px is not None:
            cv2.circle(overlay, (current_gaze_px, current_gaze_py), RAW_GAZE_CIRCLE_RADIUS, RAW_GAZE_COLOR, -1)
            cv2.circle(overlay, (current_gaze_px, current_gaze_py), RAW_GAZE_CIRCLE_RADIUS, (0, 0, 0), 1)

        if current_gaze_px is not None and not fixation_visualization_queue and not gaze_hit_aoi_this_frame:
            if not any(bbox[0] <= current_gaze_px <= bbox[0]+bbox[2] and bbox[1] <= current_gaze_py <= bbox[1]+bbox[3] for bbox in current_aoi_bboxes.values()):
                report_metrics['cellboundary']['dwell_time_s'] += frame_duration_s

        true_time_s = current_sec - analysis_start_sec
        cv2.putText(processing_frame, f"CSV Data Time: {true_time_s:.3f}s", (20, frame_height - 30), FONT, 1.0, (255, 255, 255), 2)        

        cv2.addWeighted(overlay, 0.6, processing_frame, 0.4, 0, processing_frame)
        out.write(processing_frame)
        render_bar.update(1)

    for aoi_name in raw_gaze_tracker.keys():
        if raw_gaze_tracker[aoi_name]['current_start'] is not None:
            duration = current_sec - raw_gaze_tracker[aoi_name]['current_start']
            xs = [p[0] for p in raw_gaze_tracker[aoi_name]['points']]
            ys = [p[1] for p in raw_gaze_tracker[aoi_name]['points']]
            dispersion = float(np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)) if xs else 0.0
            raw_gaze_tracker[aoi_name]['completed_glances'].append({
                'start': raw_gaze_tracker[aoi_name]['current_start'], 
                'duration': duration,
                'dispersion_px': dispersion
            })

    render_bar.close()
    cap.release(); out.release()

    print("\n--- Calculating and Saving Split Reports ---")
    incident_rows = []
    exclude_strings = ['recording.begin', 'recording.end', 'video_in', 'video_out', 'pedal_in', 'Dis_in', 'Dis_out']
    incidents_df = events_df[~events_df['name'].isin(exclude_strings) & ~events_df['name'].str.contains(r'_Q$|_A$', regex=True)]
    
    pre_duration = advanced_metrics_settings.get('pre_event_window_s', 5.0)
    post_duration = advanced_metrics_settings.get('post_event_window_s', 5.0)

    for _, row in incidents_df.iterrows():
        i_name = row['name']
        i_ts = row['timestamp_sec']
        closest_q, min_dist, phase, time_since_q = "None", float('inf'), "Independent Event", None
        
        for q_code, times in event_windows.items():
            dist = abs(i_ts - times['q'])
            if dist < min_dist:
                min_dist = dist
                closest_q = q_code
                time_since_q = i_ts - times['q']
                if i_ts < times['q']: phase = "Before Question Asked"
                elif times['q'] <= i_ts <= times['a']: phase = "During Answer Window"
                else: phase = "After Answer Given"
                
        pre_metrics = calculate_windowed_metrics(max(0, i_ts - pre_duration), i_ts, fixations_df, saccades_df, blinks_df)
        post_metrics = calculate_windowed_metrics(i_ts, i_ts + post_duration, fixations_df, saccades_df, blinks_df)

        row_data = {
            'participant_id': metadata.get('participant_id', 'N/A'),
            'route_number': metadata.get('route_number', 'N/A'),
            'Incident_Name': i_name,
            'Timestamp_s': f"{i_ts:.2f}",
            'Closest_Question': closest_q,
            'Phase_Context': phase,
            'Time_From_Question_Start_s': f"{time_since_q:.2f}" if time_since_q is not None else "N/A"
        }
        for k, v in pre_metrics.items(): row_data[f"Pre_{k}"] = v
        for k, v in post_metrics.items(): row_data[f"Post_{k}"] = v
        incident_rows.append(row_data)

    if incident_rows:
        pd.DataFrame(incident_rows).to_csv(os.path.join(output_video_dir, f"{prefix}incident_metrics.csv"), index=False)
        print(f"  > Saved: {prefix}incident_metrics.csv")

    aoi_overall_rows, aoi_interaction_rows = [], []
    aoi_mapping = event_analysis_settings.get('aoi_mapping', {})
    
    PIXELS_PER_DEGREE = 13.2 
    
    for aoi_name, metrics in report_metrics.items():
        total_dwell_time = metrics['dwell_time_s']
        num_fixations = len(metrics['hit_fixation_ids'])
        
        first_ts = float('inf')
        hit_fix_df = pd.DataFrame()
        if num_fixations > 0:
            hit_fix_df = fixations_df[fixations_df['fixation id'].isin(metrics['hit_fixation_ids'])]
            if not hit_fix_df.empty: first_ts = hit_fix_df['timestamp_sec'].min()
        
        avg_dwell_per_fix = total_dwell_time / num_fixations if num_fixations > 0 else 0

        raw_glances = raw_gaze_tracker.get(aoi_name, {}).get('completed_glances', [])
        total_raw_dwell = sum(g['duration'] for g in raw_glances)
        glances_gt_30ms = [g for g in raw_glances if g['duration'] >= 0.03]
        
        max_glance_dur = 0.0
        max_glance_dispersion_px = 0.0
        max_glance_dispersion_deg = 0.0
        
        if raw_glances:
            longest_glance = max(raw_glances, key=lambda g: g['duration'])
            max_glance_dur = longest_glance['duration']
            max_glance_dispersion_px = longest_glance['dispersion_px']
            max_glance_dispersion_deg = max_glance_dispersion_px / PIXELS_PER_DEGREE

        aoi_overall_rows.append({
            'participant_id': metadata.get('participant_id', 'N/A'),
            'event_id': metadata.get('event_id', 'N/A'),
            'AOI_Name': aoi_name,
            'Total_Fixation_Dwell_Time_s': f"{total_dwell_time:.2f}",
            'Number_of_Fixations': num_fixations,
            'Average_Dwell_per_Fixation_s': f"{avg_dwell_per_fix:.2f}" if num_fixations > 0 else "0.00",
            'First_Fixation_Timestamp_s': f"{first_ts:.2f}" if first_ts != float('inf') else "N/A",
            'Total_Raw_Gaze_Dwell_s': f"{total_raw_dwell:.3f}",
            'Raw_Glances_GT_30ms': len(glances_gt_30ms),
            'Max_Raw_Glance_Duration_s': f"{max_glance_dur:.3f}",
            'Max_Glance_Dispersion_px': f"{max_glance_dispersion_px:.1f}",
            'Max_Glance_Dispersion_deg': f"{max_glance_dispersion_deg:.2f}"
        })

        if aoi_name != 'cellboundary':
            # Only chop if the last part is a tracking number!
            if aoi_name.rsplit('_', 1)[-1].isdigit():
                base_aoi_name = aoi_name.rsplit('_', 1)[0]
            else:
                base_aoi_name = aoi_name
                
            applicable_questions = aoi_mapping.get(base_aoi_name, list(event_windows.keys()))

            for q_code in applicable_questions:
                if q_code not in event_windows: continue
                q_ts, a_ts = event_windows[q_code]['q'], event_windows[q_code]['a']
                after_ans_time = a_ts + post_duration
                
                before_df = hit_fix_df[hit_fix_df['timestamp_sec'].between(q_ts - pre_duration, q_ts, inclusive='left')] if not hit_fix_df.empty else []
                during_df = hit_fix_df[hit_fix_df['timestamp_sec'].between(q_ts, a_ts, inclusive='both')] if not hit_fix_df.empty else []
                after_df = hit_fix_df[hit_fix_df['timestamp_sec'].between(a_ts, after_ans_time, inclusive='right')] if not hit_fix_df.empty else []
                
                b_glances = [g for g in raw_glances if (q_ts - pre_duration) <= g['start'] < q_ts]
                d_glances = [g for g in raw_glances if q_ts <= g['start'] <= a_ts]
                a_glances = [g for g in raw_glances if a_ts < g['start'] <= after_ans_time]

                if not hit_fix_df.empty or raw_glances:
                    aoi_interaction_rows.append({
                        'participant_id': metadata.get('participant_id', 'N/A'),
                        'event_id': metadata.get('event_id', 'N/A'),
                        'AOI_Name': aoi_name,
                        'Question_ID': q_code,
                        
                        'Fixations_Before': len(before_df),
                        'Dwell_Before_s': f"{before_df['duration_sec'].sum():.2f}" if not hit_fix_df.empty else "0.00",
                        'Raw_Glances_Before': len([g for g in b_glances if g['duration'] >= 0.03]),
                        'Raw_Gaze_Dwell_Before_s': f"{sum(g['duration'] for g in b_glances):.3f}",
                        
                        'Fixations_During': len(during_df),
                        'Dwell_During_s': f"{during_df['duration_sec'].sum():.2f}" if not hit_fix_df.empty else "0.00",
                        'Raw_Glances_During': len([g for g in d_glances if g['duration'] >= 0.03]),
                        'Raw_Gaze_Dwell_During_s': f"{sum(g['duration'] for g in d_glances):.3f}",
                        
                        'Fixations_After': len(after_df),
                        'Dwell_After_s': f"{after_df['duration_sec'].sum():.2f}" if not hit_fix_df.empty else "0.00",
                        'Raw_Glances_After': len([g for g in a_glances if g['duration'] >= 0.03]),
                        'Raw_Gaze_Dwell_After_s': f"{sum(g['duration'] for g in a_glances):.3f}"
                    })

    if aoi_overall_rows:
        pd.DataFrame(aoi_overall_rows).to_csv(os.path.join(output_video_dir, f"{prefix}aoi_overall_metrics.csv"), index=False)
        print(f"  > Saved: {prefix}aoi_overall_metrics.csv")
        
    if aoi_interaction_rows:
        pd.DataFrame(aoi_interaction_rows).to_csv(os.path.join(output_video_dir, f"{prefix}aoi_question_interactions.csv"), index=False)
        print(f"  > Saved: {prefix}aoi_question_interactions.csv")

    print("\n--- Generating Route Fixations Timeline ---")
    analysis_start_sec = analysis_start_frame / fps
    analysis_end_sec = analysis_end_frame / fps
    
    route_fixations = fixations_df[
        (fixations_df['timestamp_sec'] >= analysis_start_sec) & 
        (fixations_df['timestamp_sec'] <= analysis_end_sec)
    ].copy()
    
    def get_hit_aois(fix_id):
        hits = [aoi for aoi, m in report_metrics.items() if fix_id in m['hit_fixation_ids'] and aoi != 'cellboundary']
        return ", ".join(hits) if hits else "None"
        
    timeline_export = pd.DataFrame({
        'Participant_ID': metadata.get('participant_id', 'N/A'),
        'Fixation_ID': route_fixations['fixation id'],
        'Global_Time_s': route_fixations['timestamp_sec'].round(3),
        'Video_Time_s': (route_fixations['timestamp_sec'] - analysis_start_sec).round(3),
        'Duration_s': route_fixations['duration_sec'].round(3),
        'X_px': route_fixations['fixation x [px]'].round(1),
        'Y_px': route_fixations['fixation y [px]'].round(1),
        'Hit_AOI': route_fixations['fixation id'].apply(get_hit_aois)
    })
    
    TIMELINE_FILE = os.path.join(output_video_dir, f"{prefix}all_fixations_timeline.csv")
    timeline_export.to_csv(TIMELINE_FILE, index=False)
    print(f"  > Saved: {prefix}all_fixations_timeline.csv")

    print("\n--- Generating Route Raw Glances Timeline ---")
    all_glances_rows = []
    
    for aoi_name, tracker_data in raw_gaze_tracker.items():
        if aoi_name == 'cellboundary': continue
        for glance in tracker_data.get('completed_glances', []):
            if glance['duration'] >= 0.03:
                disp_px = glance['dispersion_px']
                disp_deg = disp_px / PIXELS_PER_DEGREE
                
                all_glances_rows.append({
                    'Participant_ID': metadata.get('participant_id', 'N/A'),
                    'AOI_Name': aoi_name,
                    'Global_Time_s': round(glance['start'], 3),
                    'Video_Time_s': round(glance['start'] - analysis_start_sec, 3),
                    'Duration_s': round(glance['duration'], 3),
                    'Dispersion_px': round(disp_px, 1),
                    'Dispersion_deg': round(disp_deg, 2)
                })
                
    if all_glances_rows:
        glances_df = pd.DataFrame(all_glances_rows)
        glances_df = glances_df.sort_values(by='Global_Time_s') 
        GLANCES_TIMELINE_FILE = os.path.join(output_video_dir, f"{prefix}all_raw_glances_timeline.csv")
        glances_df.to_csv(GLANCES_TIMELINE_FILE, index=False)
        print(f"  > Saved: {prefix}all_raw_glances_timeline.csv")

    print("\n--- Generating Master ML Eye-Tracking Dataset ---")
    ml_rows = []
    
    hazard_events = events_df[events_df['name'].str.contains('collision|Wrong_Way|X|Lost_Control', case=False, na=False)]
    
    def append_ml_row(event_name, win_name, start_t, end_t, is_hazard_flag=0):
        duration = end_t - start_t
        if duration <= 0: return
        
        base_metrics = calculate_windowed_metrics(start_t, end_t, fixations_df, saccades_df, blinks_df)
        
        row = {
            'Participant_ID': metadata.get('participant_id', 'N/A'),
            'Route_ID': metadata.get('route_number', 'N/A'),
            'Event': event_name,
            'Window': win_name,
            'Duration_s': round(duration, 3),
            'Hazard_Encountered': is_hazard_flag
        }
        
        row['Saccade_Count'] = base_metrics.get('Saccade_Count', 0)
        row['Blink_Count'] = base_metrics.get('Blink_Count', 0)
        row['Saccade_Rate_Hz'] = base_metrics.get('Saccade_Rate_Hz', 0)
        row['Blink_Rate_BPM'] = base_metrics.get('Blink_Rate_BPM', 0)
        row['Scanpath_Length_px'] = base_metrics.get('Scanpath_Length_px', 0)
        row['Mean_Saccadic_Velocity'] = base_metrics.get('Mean_Saccadic_Velocity_px_s', 0)
        
        aoi_summaries = {}
        for aoi_name in report_metrics.keys():
            if aoi_name == 'cellboundary': continue
            base_aoi_name = aoi_name.rsplit('_', 1)[0]
            if base_aoi_name not in aoi_summaries:
                aoi_summaries[base_aoi_name] = {'fix_dwell': 0.0, 'raw_dwell': 0.0}
                
            # --- OVERLAP MATH FOR FIXATIONS ---
            hit_fix_ids = report_metrics[aoi_name]['hit_fixation_ids']
            if hit_fix_ids:
                hit_fix_df = fixations_df[fixations_df['fixation id'].isin(hit_fix_ids)]
                for _, fix_row in hit_fix_df.iterrows():
                    f_start = fix_row['timestamp_sec']
                    f_end = f_start + fix_row['duration_sec']
                    # Find how much of this fixation actually falls inside the [start_t, end_t] window
                    overlap_start = max(start_t, f_start)
                    overlap_end = min(end_t, f_end)
                    overlap_duration = max(0.0, overlap_end - overlap_start)
                    aoi_summaries[base_aoi_name]['fix_dwell'] += overlap_duration
            
            # --- OVERLAP MATH FOR RAW GLANCES ---
            raw_glances = raw_gaze_tracker.get(aoi_name, {}).get('completed_glances', [])
            for g in raw_glances:
                if g['duration'] >= 0.03: # Only count glances > 30ms
                    g_start = g['start']
                    g_end = g_start + g['duration']
                    # Find how much of this raw glance actually falls inside the [start_t, end_t] window
                    overlap_start = max(start_t, g_start)
                    overlap_end = min(end_t, g_end)
                    overlap_duration = max(0.0, overlap_end - overlap_start)
                    aoi_summaries[base_aoi_name]['raw_dwell'] += overlap_duration
            
        for base_aoi_name, vals in aoi_summaries.items():
            row[f'Fix_Dwell_{base_aoi_name}'] = round(vals['fix_dwell'], 3)
            row[f'Raw_Dwell_{base_aoi_name}'] = round(vals['raw_dwell'], 3)
            
        ml_rows.append(row)

    def check_hazard_in_window(start_t, end_t):
        for _, haz_row in hazard_events.iterrows():
            if start_t <= haz_row['timestamp_sec'] <= end_t:
                return 1
        return 0

    for q_code, times in event_windows.items():
        q_ts = times['q']
        a_ts = times['a']
        
        b_start, b_end = max(0, q_ts - pre_duration), q_ts
        b_hazard = check_hazard_in_window(b_start, b_end)
        append_ml_row(q_code, "Before", b_start, b_end, b_hazard)
        
        d_start, d_end = q_ts, a_ts
        d_hazard = check_hazard_in_window(d_start, d_end)
        append_ml_row(q_code, "During", d_start, d_end, d_hazard)
        
        a_start, a_end = a_ts, a_ts + post_duration
        a_hazard = check_hazard_in_window(a_start, a_end)
        append_ml_row(q_code, "After", a_start, a_end, a_hazard)

    hazard_counters = {} 
    
    for _, row in hazard_events.iterrows():
        base_name = row['name']
        hazard_counters[base_name] = hazard_counters.get(base_name, 0) + 1
        
        unique_event_name = f"{base_name}_{hazard_counters[base_name]}"
        i_ts = row['timestamp_sec']
        
        append_ml_row(unique_event_name, "Before", max(0, i_ts - pre_duration), i_ts, 1)
        append_ml_row(unique_event_name, "After", i_ts, i_ts + post_duration, 1)

    if ml_rows:
        ml_df = pd.DataFrame(ml_rows)
        ML_FILE = os.path.join(output_video_dir, f"{prefix}eyetracking_ml_features.csv")
        ml_df.to_csv(ML_FILE, index=False)
        print(f"  > Saved: {prefix}eyetracking_ml_features.csv")
        print("  > Columns:", list(ml_df.columns))

    print("\n--- DONE! All processes completed successfully. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Gaze and Object Tracking Analysis Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args.config)