# ============================================================================
# wavstomp.py  —  speech-segment detection and Q/A timing from recordings
# ============================================================================
# Purpose:  Detect speech segments in the recorded sessions (voice-activity
#           detection), separate questions from answers by expected
#           alternation, log segment timings, and plot annotated waveforms.
# Inputs:   session .wav files; question-flag CSV
# Outputs:  segment-timing CSV; waveform plots
# Requires: librosa, webrtcvad, numpy, matplotlib
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import webrtcvad

def load_question_flags(flag_csv_file, participant_id, condition):
    question_times = []
    with open(flag_csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Participant ID'] == participant_id and row['Condition'] == condition:
                question_times.append(float(row['Question Timestamp (s)']))
    return question_times

def vad_detect_speech(audio, sr, frame_duration=30):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # 0: most aggressive, 3: least aggressive
    
    # Resample the audio to 16000 Hz if necessary
    if sr != 16000:
        print(f"Resampling audio from {sr} Hz to 16000 Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Convert audio to 16-bit PCM format
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Calculate frame size in samples
    frame_size = int(sr * frame_duration / 1000)
    
    speech_segments = []
    current_start = None
    current_end = None

    print("Running VAD on audio frames...")
    for start in range(0, len(audio_int16) - frame_size, frame_size):
        end = start + frame_size
        try:
            is_speech = vad.is_speech(audio_int16[start:end].tobytes(), sr)
        except ValueError as e:
            print(f"Error processing frame {start}-{end}: {e}")
            continue
        
        if is_speech:
            if current_start is None:
                current_start = start
            current_end = end
        else:
            if current_start is not None:
                speech_segments.append((current_start, current_end))
                current_start = None
    
    if current_start is not None:
        speech_segments.append((current_start, current_end))

    print(f"Detected {len(speech_segments)} speech segments")
    return [(start / sr, end / sr) for start, end in speech_segments]

def analyze_audio_with_vad(file_path, question_times):
    audio, sr = librosa.load(file_path, sr=None)
    segments = []

    for question_time in question_times:
        segments.append(('Question', question_time, question_time))
        
        # Detect speech after the question timestamp
        vad_segments = vad_detect_speech(audio, sr)
        for start, end in vad_segments:
            if start > question_time:
                segments.append(('Answer', start, end))
                break  # Assume only the first segment after the question is the answer
    
    return segments, audio, sr

def save_segments_to_csv(csv_file, segments, participant_id, condition):
    event_id = 1  # Start event ID
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        
        question_time = None
        
        for segment_type, start, end in segments:
            if segment_type == 'Question':
                question_time = start  # Start and end are the same for questions
            elif segment_type == 'Answer' and question_time is not None:
                time_difference = end - question_time
                writer.writerow([participant_id, condition, f'E{event_id}', f'{question_time:.2f}', f'{end:.2f}', f'{time_difference:.2f}'])
                event_id += 1  # Increment event ID for each new Q&A pair

def plot_segments(segments, audio, sr, plot_file):
    time = np.linspace(0, len(audio) / sr, len(audio))
    plt.figure(figsize=(14, 8))
    plt.plot(time, audio, label='Waveform')

    for segment_type, start, end in segments:
        color = 'yellow' if segment_type == 'Question' else 'red'
        plt.axvspan(start, end, color=color, alpha=0.3, label=segment_type)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform with Detected Segments')
    plt.legend(loc='upper right')
    plt.savefig(plot_file)
    plt.close()  # Close the plot to save memory

def extract_info_from_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    participant_id = parts[1]
    condition = parts[2][1]  # Take the second character after the underscore
    return participant_id, condition

def process_directory(main_directory, flag_csv, output_csv):
    # Ensure the data directory exists
    data_directory = os.path.dirname(output_csv)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Prepare the CSV file with a header
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Participant ID', 'Condition', 'Event ID', 'Question Timestamp (s)', 'Answer Timestamp (s)', 'Time Difference (s)'])
    
    all_segments = []

    # Recursively walk through the directory
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                participant_id, condition = extract_info_from_filename(filename)
                
                # Load the question times from the flag CSV file
                question_times = load_question_flags(flag_csv, participant_id, condition)
                
                # Analyze the audio and detect segments using VAD
                segments, audio, sr = analyze_audio_with_vad(file_path, question_times)
                
                # Prepare the segments for sorting
                event_id = 1
                for segment_type, start, end in segments:
                    if segment_type == 'Question':
                        question_time = start
                    elif segment_type == 'Answer' and question_time is not None:
                        time_difference = end - question_time
                        all_segments.append([participant_id, condition, f'E{event_id}', question_time, end, time_difference])
                        event_id += 1

                # Generate and save the plot
                plot_file = os.path.join(root, f'{os.path.splitext(filename)[0]}.png')
                plot_segments(segments, audio, sr, plot_file)
                print(f"Processed {filename}, saved plot to {plot_file}")
    
    # Sort all segments by Participant ID, Condition, and Event ID
    all_segments.sort(key=lambda x: (x[0], x[1], int(x[2][1:])))  # Assuming Event ID format is 'E<number>'
    
    # Write sorted segments to the CSV
    with open(output_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        for segment in all_segments:
            writer.writerow([segment[0], segment[1], segment[2], f'{segment[3]:.2f}', f'{segment[4]:.2f}', f'{segment[5]:.2f}'])

if __name__ == "__main__":
    # Calculate the absolute path to the main directory
    main_directory = os.path.abspath(os.path.join('..', 'participants'))  # Move up one directory level to reach the participants directory
    flag_csv = os.path.abspath(os.path.join('..', 'data', 'flagged_events.csv'))  # Path to the CSV file with question flags
    output_csv = os.path.abspath(os.path.join('..', 'data', 'main_segments.csv'))  # Save the main_segments.csv in the data subdirectory
    
    # Process all WAV files in the main directory and its subdirectories
    process_directory(main_directory, flag_csv, output_csv)
    print("Processing complete.")
