# ============================================================================
# AI_Clapper.py  —  Whisper-based audio synchronisation (single-file prototype)
# ============================================================================
# Purpose:  Transcribe the scene video's audio and the app WAV with Whisper,
#           locate a spoken 'mark' anchor in each, and report the timeline
#           offset that synchronises them.
# Inputs:   one scene video (.mp4) and one response recording (.wav)
# Outputs:  printed synchronisation offset
# Usage:    python AI_Clapper.py
# Requires: openai-whisper, moviepy, pandas
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import whisper
import pandas as pd
import moviepy.editor as mp
import os
import re

# --- 1. CONFIGURATION ---
VIDEO_FILE = "c8e90669_0.0-145.975.mp4" 
WAV_FILE = "recording_02_RT_R1_1764087305.wav"
ANCHOR_WORD = "mark"

def extract_audio(video_path):
    print(f"Extracting audio track from {video_path}...")
    audio_path = "temp_video_audio.wav"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

# Boost the (often quiet) app audio before transcribing.
def boost_audio(audio_path):
    print(f"Boosting volume of {audio_path} by 10x...")
    boosted_path = "temp_boosted_audio.wav"
    audio = mp.AudioFileClip(audio_path)
    # Multiply the volume by 10
    boosted = audio.volumex(10.0)
    boosted.write_audiofile(boosted_path, logger=None)
    return boosted_path

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word.lower()).strip()

def find_anchor_time(result, anchor):
    for segment in result['segments']:
        for word_info in segment['words']:
            if clean_word(word_info['word']) == anchor:
                return word_info['start']
    return None

def test_clapperboard_sync():
    print("Loading Whisper AI (Small Model)...")
    model = whisper.load_model("small")
    
    # --- STEP 1: Process the Eye-Tracking Video ---
    video_audio_path = extract_audio(VIDEO_FILE)
    print("Transcribing Video Audio...")
    video_result = model.transcribe(video_audio_path, word_timestamps=True, initial_prompt="mark")
    video_anchor = find_anchor_time(video_result, ANCHOR_WORD)
    
    # --- STEP 2: Process the App's WAV File ---
    # Boost the app audio before transcribing.
    boosted_wav_path = boost_audio(WAV_FILE)
    print(f"Transcribing Boosted App Audio...")
    app_result = model.transcribe(boosted_wav_path, word_timestamps=True, initial_prompt="mark")
    
    print("\n" + "="*50)
    print("🔍 DIAGNOSTIC: RAW APP AUDIO TRANSCRIPT (BOOSTED)")
    print("="*50)
    print(app_result["text"].strip())
    print("="*50 + "\n")
    
    app_anchor = find_anchor_time(app_result, ANCHOR_WORD)
    
    # Cleanup temporary files
    for temp_file in [video_audio_path, boosted_wav_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    # --- STEP 3: The Math ---
    if video_anchor is None or app_anchor is None:
        print(f"❌ ERROR: Could not find the word '{ANCHOR_WORD}' in one or both files.")
        print(f"Video Anchor: {video_anchor} | App Anchor: {app_anchor}")
        return
        
    offset = video_anchor - app_anchor
    
    print("\n" + "="*50)
    print("SYNCHRONIZATION RESULTS")
    print("="*50)
    print(f"Eye-Tracking Video Anchor ('{ANCHOR_WORD}'): {video_anchor:.3f} seconds")
    print(f"App WAV File Anchor ('{ANCHOR_WORD}'):       {app_anchor:.3f} seconds")
    print("-" * 50)
    print(f"⏱️ GLOBAL TIMELINE OFFSET:       {offset:+.3f} seconds")
    print("="*50)

if __name__ == "__main__":
    test_clapperboard_sync()