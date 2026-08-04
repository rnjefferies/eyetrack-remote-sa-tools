import whisper
import pandas as pd
import moviepy.editor as mp
import os
import glob
import re
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "Data_Sorted"   # The new folder we created
FLAGGED_CSV = "flagged_events.csv"
COLLISIONS_CSV = "collisions.csv"
ANCHOR_WORD = "mark"

def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", "_temp_vid_audio.wav")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path

def boost_audio(audio_path):
    boosted_path = audio_path.replace(".wav", "_boosted.wav")
    audio = mp.AudioFileClip(audio_path)
    boosted = audio.volumex(10.0)
    boosted.write_audiofile(boosted_path, logger=None)
    audio.close()
    return boosted_path

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word.lower()).strip()

def find_anchor_time(result, anchor):
    for segment in result['segments']:
        for word_info in segment['words']:
            if clean_word(word_info['word']) == anchor:
                return word_info['start']
    return None

def extract_keyword_from_event(event_id):
    clean_name = str(event_id).split('_')[-1].replace('.wav', '').lower()
    number_map = {"speed0": "zero", "speed1": "one", "speed2": "two", "speed3": "three", "speed4": "four", "speed5": "five"}
    if clean_name in number_map: return number_map[clean_name]
    if "frogs" in clean_name: return "frog"
    if "duck" in clean_name: return "duck"
    return clean_name

def find_exact_spoken_time(transcript_result, rough_human_time, expected_word, accuracy_status):
    search_start = rough_human_time - 3.5
    search_end = rough_human_time + 1.0
    words_in_window = []
    for segment in transcript_result['segments']:
        for word_info in segment['words']:
            w_start, w_text = word_info['start'], clean_word(word_info['word'])
            if search_start <= w_start <= search_end:
                words_in_window.append(word_info)
                if accuracy_status == "Correct" and expected_word in w_text:
                    return w_start
    return words_in_window[0]['start'] if words_in_window else rough_human_time - 1.5

def process_all_videos():
    print("🚀 Initializing Whisper AI (Small Model)...")
    model = whisper.load_model("small")
    df_flags = pd.read_csv(FLAGGED_CSV)
    df_col = pd.read_csv(COLLISIONS_CSV)
    
    # Get list of folders like R1_01, R1_02
    sessions = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Found {len(sessions)} folders. Starting sync...")

    for folder in tqdm(sessions):
        path = os.path.join(DATA_DIR, folder)
        video_files = glob.glob(os.path.join(path, "*.mp4"))
        audio_files = [f for f in glob.glob(os.path.join(path, "*.wav")) if "_temp" not in f and "_boosted" not in f]
        
        if not video_files or not audio_files: continue
        
        # Parse Route/Participant from folder name "R1_02"
        match = re.match(r'R(\d+)_(\d+)', folder)
        if not match: continue
        route_id, part_id_num = int(match.group(1)), match.group(2)
        part_id_csv = f"{part_id_num}_RT"

        try:
            # 1. Audio Sync
            v_aud = extract_audio(video_files[0])
            v_res = model.transcribe(v_aud, word_timestamps=True, initial_prompt="mark")
            v_anchor = find_anchor_time(v_res, ANCHOR_WORD)
            
            a_boost = boost_audio(audio_files[0])
            a_res = model.transcribe(a_boost, word_timestamps=True, initial_prompt="mark")
            a_anchor = find_anchor_time(a_res, ANCHOR_WORD)
            
            if v_anchor is None or a_anchor is None:
                print(f"Skipping {folder}: 'Mark' not found.")
                continue
            
            offset = v_anchor - a_anchor
            
            # 2. Map Events
            events = []
            # Flags
            s_flags = df_flags[(df_flags['Participant ID'] == part_id_csv) & (df_flags['Route'] == route_id)]
            for _, f_row in s_flags.iterrows():
                e_id = str(f_row['Event ID'])
                q_num = e_id.split('_')[0].replace('E', '')
                q_time = float(f_row['Question Timestamp (s)']) + offset
                events.append({"name": f"Q{q_num}_Q", "timestamp_sec": round(q_time, 3)})
                
                if pd.notna(f_row['Answer Timestamp (s)']):
                    rough_a = float(f_row['Answer Timestamp (s)']) + offset
                    word = extract_keyword_from_event(e_id)
                    exact_a = find_exact_spoken_time(v_res, rough_a, word, str(f_row['Answer Accuracy']))
                    events.append({"name": f"Q{q_num}_A", "timestamp_sec": round(exact_a, 3)})
            
            # Collisions
            s_cols = df_col[(df_col['Participant ID'] == part_id_csv) & (df_col['Route'] == route_id)]
            for _, c_row in s_cols.iterrows():
                if pd.notna(c_row['Collision Timestamp (s)']):
                    events.append({"name": "collision", "timestamp_sec": round(float(c_row['Collision Timestamp (s)']) + offset, 3)})

            # 3. Save
            if events:
                pd.DataFrame(events).sort_values("timestamp_sec").to_csv(os.path.join(path, f"{folder}_custom_events.csv"), index=False)

        finally: # Clean up
            if 'v_aud' in locals() and os.path.exists(v_aud): os.remove(v_aud)
            if 'a_boost' in locals() and os.path.exists(a_boost): os.remove(a_boost)

if __name__ == "__main__":
    process_all_videos()