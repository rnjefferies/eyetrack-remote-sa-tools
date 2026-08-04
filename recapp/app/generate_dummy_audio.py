# ============================================================================
# generate_dummy_audio.py  —  create placeholder query-prompt tones (testing)
# ============================================================================
# Purpose:  Generate dummy .wav tone files (one per label in labels.csv) so
#           the Recapp GUI can be exercised without the real prompt audio.
# Inputs:   labels.csv
# Outputs:  dummy .wav files in audio/
# Usage:    python generate_dummy_audio.py
# Requires: numpy
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import csv
import wave
import os
import numpy as np

# Paths
csv_file = "labels.csv"  # Adjust this path to your actual labels.csv location
output_directory = "audio"  # Directory where dummy files will be saved

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

def generate_dummy_wav(file_path, duration=1, sample_rate=44100):
    """Generates a dummy .wav file with a simple tone."""
    amplitude = 32767  # Max amplitude for 16-bit audio
    frequency = 440.0  # A4 tone
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    
    # Save as .wav file
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())

def generate_files_from_csv(csv_path, output_dir):
    """Reads the CSV file and generates dummy audio files for each entry."""
    try:
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for column in ['Audio1', 'Audio2', 'Audio3']:
                    audio_file = row[column]
                    if audio_file:
                        output_path = os.path.join(output_dir, audio_file)
                        if not os.path.exists(output_path):  # Avoid overwriting existing files
                            print(f"Generating {output_path}...")
                            generate_dummy_wav(output_path)
                        else:
                            print(f"{output_path} already exists. Skipping.")
        print("All dummy audio files generated successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Run the script
generate_files_from_csv(csv_file, output_directory)
