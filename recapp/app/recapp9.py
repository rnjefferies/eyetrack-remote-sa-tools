# ============================================================================
# recapp9.py  —  Recapp: SA query delivery and response recording (GUI, v9)
# ============================================================================
# Purpose:  Tkinter app that plays the spoken situation-awareness query
#           prompts, records the operator's spoken response, and flags the
#           end of each question and answer as timestamped events.
# Inputs:   query prompt audio; participant ID and condition (GUI)
# Outputs:  per-session response .wav files; flagged-events CSV
# Usage:    python recapp9.py
# Requires: sounddevice, numpy, Pillow, tkinter
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import sys
import os
import numpy as np
import sounddevice as sd
import tkinter as tk
import time
import csv
import wave
import random
from PIL import Image, ImageTk
from tkinter import messagebox
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.audio_player import AudioPlayer

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recapp: for SA Data Collection")

        # Initialise label and audio management
        self.labels_data = {}
        self.collisions = []  
        self.current_label_index = 0
        self.current_audio_index = 0

        self.load_labels_from_csv()

        self.set_app_icon()
        self.condition_audio_directory = None
        self.recording = False
        self.start_time = None
        self.recorded_chunks = []
        self.played_audio_tracker = {}
        self.event_id = 1
        self.participant_id = None
        self.condition = None
        self.confidence_ratings = {}  # Store confidence ratings by event ID
        self.collision_flags = {}  # Tracks collision status (Yes/No) by event ID
        self.answer_timestamp = None   # Holds the timestamp for the answer
        self.question_played = False  # Initialize the flag to prevent AttributeError
        self.is_answer_incorrect = False  # Flag for answer correctness
        self.question_times = {}
        self.answer_times = {}
        self.correctness_flags = {}
        self.last_flagged = None
        self.audio_player = None
        self.current_question = 1
        self.last_keypress_time = 0  # Tracks last key press time for double-tap prevention
        self.collision_flagged = False  # Tracks if a collision occurred for the current event
        self.collision_timestamp = None  # Holds the time of the collision



        # Adjust paths according to your directory structure
        self.main_directory = os.path.join('..', 'participants')
        os.makedirs(self.main_directory, exist_ok=True)

        self.data_directory = os.path.join('..', 'data')
        os.makedirs(self.data_directory, exist_ok=True)

        self.audio_directory = os.path.join('..', 'audio')
        if not os.path.exists(self.audio_directory):
            messagebox.showerror("Error", "Audio questions directory not found!")
            self.root.quit()

        self.csv_filename = os.path.join(self.data_directory, 'flagged_events.csv')
        self.ensure_csv_file_exists()

        self.collision_csv_filename = os.path.join(self.data_directory, 'collisions.csv')
        self.ensure_collision_csv_file_exists()

        validate_numeric_command = root.register(self.validate_numeric_input)
        validate_letter_command = root.register(self.validate_letter_condition)

        # UI setup
        self.content_frame = tk.Frame(root)
        self.content_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.participant_id_label = tk.Label(self.content_frame, text="Participant ID:")
        self.participant_id_label.pack(pady=(10, 5))

        self.participant_id_entry = tk.Entry(self.content_frame)
        self.participant_id_entry.pack(pady=(0, 20))

        self.condition_label = tk.Label(self.content_frame, text="Condition (A_R, A_L, B_R, B_L, C_R, C_L):")
        self.condition_label.pack(pady=(10, 5))

        self.condition_entry = tk.Entry(self.content_frame)
        self.condition_entry.pack(pady=(0, 20))

        self.start_button = tk.Button(self.content_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=(10, 5))

        self.stop_button = tk.Button(self.content_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=(5, 5))

        self.label_label = tk.Label(self.content_frame, text="Current Label: None")
        self.label_label.pack(pady=(5, 5))

        self.audio_label = tk.Label(self.content_frame, text="Current Audio: None")
        self.audio_label.pack(pady=(5, 5))

        self.flag_question_button = tk.Button(self.content_frame, text="Flag Question", command=self.flag_question, state=tk.DISABLED)
        self.flag_question_button.pack(pady=(5, 5))

        self.flag_answer_button = tk.Button(self.content_frame, text="Finalise Answer", command=self.finalise_answer, state=tk.DISABLED)
        self.flag_answer_button.pack(pady=(5, 10))

        self.status_label = tk.Label(self.content_frame, text="Status: Idle")
        self.status_label.pack(pady=(10, 5))

        self.recording_time_label = tk.Label(self.content_frame, text="Recording Time: 00:00")
        self.recording_time_label.pack(pady=(5, 10))

        self.confidence_label = tk.Label(self.content_frame, text="Confidence Rating: Not Selected", fg="orange")
        self.confidence_label.pack(pady=(5, 10))

        self.finalisation_status_label = tk.Label(self.content_frame, text="Not finalised", fg="red")
        self.finalisation_status_label.pack(pady=(5, 10))

                # Add "Flag Collision" Button
        self.flag_collision_button = tk.Button(
            self.content_frame,
            text="Flag Collision",
            command=self.flag_collision,
            state=tk.DISABLED
        )
        self.flag_collision_button.pack(pady=(5, 10))


        # Key bindings
        self.root.bind('<q>', self.flag_question_key)
        self.root.bind('<a>', self.capture_correct_answer)
        self.root.bind('<d>', self.capture_incorrect_answer)
        self.root.bind('<space>', self.finalise_answer)
        self.root.bind('<Up>', lambda e: self.previous_label())
        self.root.bind('<Down>', lambda e: self.next_label())
        self.root.bind('<Left>', lambda e: self.previous_audio())
        self.root.bind('<Right>', lambda e: self.next_audio())
        self.root.bind('<Return>', self.play_current_audio)
        self.root.bind('<BackSpace>', self.toggle_correctness)
        self.root.bind('<l>', self.flag_collision_key)


        for i in range(10):  # 0 through 9
            self.root.bind(f'<Key-{i}>', self.handle_number_key)
        
        self.sample_rate = 44100
        self.channels = 1
        self.dtype = np.int16

        self.update_clock()

        self.update_ui()

    def load_labels_from_csv(self):
        """Loads labels and audio files from the CSV file."""
        csv_path = 'labels.csv'  # Path to your CSV file
        try:
            with open(csv_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    label = row['Label']
                    audio_files = [row['Audio1'], row['Audio2'], row['Audio3']]

                    # Initialise the label data if not already present
                    if label not in self.labels_data:
                        self.labels_data[label] = {'audio_files': audio_files, 'played_audio': []}

                    # Do not shuffle audio here, we'll select randomly during playback
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            sys.exit(1)



    def update_ui(self):
        """Updates the UI to display the current label."""
        if self.current_label_index >= len(self.labels_data):
            self.status_label.config(text="All labels have been played.", fg="green")
            return  # Exit if all labels have been processed

        current_label = list(self.labels_data.keys())[self.current_label_index]

        if current_label not in self.labels_data:
            print(f"ERROR: Label '{current_label}' not found in labels_data.")
            self.status_label.config(text=f"Error: Label '{current_label}' not found.", fg="red")
            return

        audio_files = self.labels_data[current_label].get('audio_files', [])
        played_audio = self.labels_data[current_label].get('played_audio', [])

        # Update the UI
        self.label_label.config(text=f"Current Label: {current_label}")
        self.audio_label.config(text=f"Audio Files: {audio_files}")



        self.update_clock()



    def set_app_icon(self):
        try:
            icon_image = Image.open('icon2.png')
            icon_photo = ImageTk.PhotoImage(icon_image)
            self.root.iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Error loading application icon: {e}")

    def validate_numeric_input(self, input):
        """Allows valid Participant ID input in the format XX_OL or XX_RT."""
        import re
        # Match two digits followed by _OL or _RT
        pattern = r"^\d{2}_(OL|RT)$"
        return re.match(pattern, input) is not None or input == ''


    def validate_letter_condition(self, input):
        """Allows letters and underscores progressively."""
        allowed_conditions = {'A_R', 'A_L', 'B_R', 'B_L', 'C_R', 'C_L'}
        # Allow partial matches during typing
        return input == '' or any(cond.startswith(input) for cond in allowed_conditions)


    def ensure_csv_file_exists(self):
        if not os.path.isfile(self.csv_filename):
            print("Creating new CSV file with headers...")
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant ID', 'Condition', 'Event ID', 'Question Timestamp (s)',
                             'Answer Timestamp (s)', 'Time Difference (s)', 'Answer Accuracy', 'Confidence'])
        else:
            print("CSV file already exists. Skipping header creation.")
    
    def ensure_collision_csv_file_exists(self):
        if not os.path.isfile(self.collision_csv_filename):
            print("Creating new Collision CSV file with headers...")
            with open(self.collision_csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant ID', 'Condition', 'Collision Timestamp (s)', 'Event ID'])
        else:
            print("Collision CSV file already exists. Skipping header creation.")

    def handle_number_key(self, event):
        """Handles number key presses based on the app's state."""
        if self.recording:
            # Handle number keys for Confidence Rating input during recording
            if event.char in "1234567":
                self.set_confidence_rating(event)
            else:
                self.status_label.config(text="Invalid confidence rating. Use 1-7.", fg="red")
                print("Invalid confidence rating. Use keys 1-7.")
        return "break"  # Stop further processing of the event
    # Do nothing to allow the default behavior for Participant ID input


    def start_recording(self):
        participant_id_input = self.participant_id_entry.get().strip()
        condition_input = self.condition_entry.get().strip()

        # Validate Participant ID
        if not participant_id_input or not self.validate_numeric_input(participant_id_input):
            print("Invalid Participant ID format. Must be in the format '01_OL' or '02_RT'.")
            self.status_label.config(text="Invalid Participant ID!", fg="red")
            return

        # Validate Condition
        if condition_input not in {'A_R', 'A_L', 'B_R', 'B_L', 'C_R', 'C_L'}:
            print("Invalid Condition format. Must be one of A_R, A_L, B_R, B_L, C_R, C_L.")
            self.status_label.config(text="Invalid Condition!", fg="red")
            return

        # Reset collision tracking when a new session starts
        self.collisions = []  # Clear previous session's collisions
        self.collision_flags = {}  # Reset the dictionary tracking flagged collisions

        # Extract the base condition (e.g., "A" from "A_R") for directory matching
        base_condition = condition_input.split('_')[0]

        # Set the condition_audio_directory
        self.condition_audio_directory = os.path.join(self.audio_directory, base_condition)
        if not os.path.exists(self.condition_audio_directory):
            messagebox.showerror("Error", f"Audio directory for condition {base_condition} not found!")
            return

        # If the Participant ID or Condition changes, reset values
        if self.participant_id != participant_id_input or self.condition != condition_input:
            self.participant_id = participant_id_input
            self.condition = condition_input
            self.event_id = 1
            self.question_times = {}
            self.answer_times = {}
            self.last_flagged = None
            self.create_participant_directory()
            self.load_last_event_id()

        # Start recording logic remains the same
        self.recording = True
        self.start_time = time.time()
        self.recorded_chunks = []
        self.status_label.config(text="Status: Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.flag_question_button.config(state=tk.NORMAL)
        self.flag_answer_button.config(state=tk.NORMAL)
        self.participant_id_entry.config(state=tk.DISABLED)
        self.condition_entry.config(state=tk.DISABLED)

        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            callback=self.audio_callback
        )
        self.stream.start()





    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.status_label.config(text="Status: Idle")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.flag_question_button.config(state=tk.DISABLED)
            self.flag_answer_button.config(state=tk.DISABLED)

            self.participant_id_entry.config(state=tk.NORMAL)
            self.condition_entry.config(state=tk.NORMAL)

            # Safely stop and close the audio stream
            if hasattr(self, 'stream'):
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")

            filename = f"recording_{self.participant_id}_C{self.condition}_{int(time.time())}.wav"
            filepath = os.path.join(self.participant_directory, filename)
            try:
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(np.iinfo(self.dtype).bits // 8)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.recorded_chunks))
                print(f"Recording saved as {filepath}")
            except Exception as e:
                print(f"Error saving recording: {e}")

            self.save_flagged_events()
            self.save_collisions()

    def prepare_audio_for_playback(self):
        """Prepares the current label and audio for playback."""
        print(f"DEBUG: Preparing audio for label index {self.current_label_index}")

        if self.current_label_index >= len(self.labels_data):
            print("DEBUG: All labels have been processed.")
            self.status_label.config(text="All labels have been played.", fg="green")
            return None, None

        current_label = list(self.labels_data.keys())[self.current_label_index]
        base_label = ''.join(filter(str.isalpha, current_label))  # Remove numbers
        print(f"DEBUG: Base label: {base_label}")

        if not self.condition_audio_directory:
            print("Error: Audio directory not set!")
            return None, None

        # Fetch the audio files for the current label
        audio_files = [
            os.path.join(self.condition_audio_directory, file)
            for file in self.labels_data[current_label]['audio_files']
            if os.path.exists(os.path.join(self.condition_audio_directory, file))
        ]

        # **Snippet 4**: Handle missing audio files or directories
        if not audio_files:
            print(f"Error: No audio files found for {current_label} in {self.condition_audio_directory}.")
            self.status_label.config(text=f"No audio files for {current_label}.", fg="red")
            return None, None

        # Ensure played_audio_tracker for base_label
        if base_label not in self.played_audio_tracker:
            self.played_audio_tracker[base_label] = []

        played_audio = self.played_audio_tracker[base_label]

        print(f"DEBUG: Audio files for {current_label}: {audio_files}")
        print(f"DEBUG: Played audio for {base_label}: {played_audio}")

        # Reset and shuffle if all audio files have been played
        if len(played_audio) == len(audio_files):
            print(f"DEBUG: All audio for {base_label} has been played. Resetting for next round.")
            self.played_audio_tracker[base_label] = []
            played_audio = []

        remaining_audio = [audio for audio in audio_files if audio not in played_audio]
        print(f"DEBUG: Remaining audio for {base_label}: {remaining_audio}")

        if remaining_audio:
            selected_audio = random.choice(remaining_audio)
            self.played_audio_tracker[base_label].append(selected_audio)
            print(f"DEBUG: Selected audio for {current_label}: {selected_audio}")
            return current_label, selected_audio

        print(f"DEBUG: No remaining audio for {base_label}. Moving to next label.")
        return None, None




    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.recorded_chunks.append(indata.tobytes())

    def play_current_audio(self, event=None):
        """Plays the current audio file."""
        print(f"DEBUG: play_current_audio called. Current label index: {self.current_label_index}")

        # Prepare the audio for playback
        current_label, selected_audio = self.prepare_audio_for_playback()

        # If no valid label or audio was returned, do not proceed
        if not current_label or not selected_audio:
            print("DEBUG: No audio selected for playback.")
            return

        print(f"DEBUG: Playing audio for label: {current_label}, audio: {selected_audio}")

        # Reset the finalisation status label
        self.finalisation_status_label.config(text="Not finalised", fg="red")

        # Update the played_audio list
        self.labels_data[current_label]['played_audio'].append(selected_audio)
        self.label_label.config(text=f"Current Label: {current_label}")
        self.audio_label.config(text=f"Current Audio: {selected_audio}")

        # Path to the selected audio file
        audio_file = os.path.join(self.audio_directory, selected_audio)

        # Check if the file exists and play it
        if os.path.exists(audio_file):
            print(f"DEBUG: Audio file path exists: {audio_file}")
            self.play_audio_in_thread(audio_file)
        else:
            print(f"ERROR: Audio file {selected_audio} not found!")
            messagebox.showerror("Error", f"Audio file {selected_audio} not found!")



    def play_audio_in_thread(self, audio_file):
        """Plays the audio file in a separate thread."""
        self.audio_player = AudioPlayer(audio_file, on_playback_end=self.auto_flag_end_of_audio)
        print("Starting playback in a new thread...")  # Debugging
        self.audio_player.play()  # Playback starts in a separate thread


    def auto_flag_end_of_audio(self):
        """Automatically flag the end of the audio playback."""
        print("Entered auto_flag_end_of_audio callback.")

        if self.current_label_index >= len(self.labels_data):
            print("Error: All labels processed or invalid index.")
            return

        current_label = list(self.labels_data.keys())[self.current_label_index]

        # Ensure that the played_audio list is not empty before accessing it
        if not self.labels_data[current_label]['played_audio']:
            self.status_label.config(text="No audio has been played yet.", fg="red")
            print("Error: No audio has been played yet.")
            return  # Exit if no audio has been played

        # Get the last played audio file
        selected_audio = self.labels_data[current_label]['played_audio'][-1]
        elapsed_time = time.time() - self.start_time
        event_id_with_audio = f"E{self.event_id}_{selected_audio}"

        # Log the question timestamp
        self.question_times[event_id_with_audio] = f"{elapsed_time:.2f}"
        print(f"Question time flagged at {elapsed_time:.2f} seconds for event: {event_id_with_audio}")
        self.last_flagged = 'question'

        # Fix: Allow answer flagging after audio finishes playing
        self.question_played = True  # Set this to True after audio has played
        print("Question has been played. Answer flagging enabled.")



    def update_ui_post_playback(self):
        """Updates the UI after playback completion."""
        self.flag_question_button.config(state=tk.DISABLED)
        self.flag_answer_button.config(state=tk.NORMAL)
        self.status_label.config(text="Audio playback complete.", fg="green")

    def previous_audio(self, event=None):
        """Move to the previous audio file."""
        current_label = list(self.labels_data.keys())[self.current_label_index]
        audio_files = self.labels_data[current_label]['audio_files']
        
        if self.current_audio_index > 0:
            self.current_audio_index -= 1
        else:
            self.status_label.config(text="No previous audio.", fg="orange")
        
        self.update_ui()

    def next_audio(self, event=None):
        """Move to the next audio file."""
        current_label = list(self.labels_data.keys())[self.current_label_index]
        audio_files = self.labels_data[current_label]['audio_files']
        
        if self.current_audio_index < len(audio_files) - 1:
            self.current_audio_index += 1
        else:
            self.status_label.config(text="No more audio.", fg="orange")
        
        self.update_ui()


    def previous_label(self, event=None):
        """Move to the previous label."""
        if self.current_label_index > 0:
            self.current_label_index -= 1
            self.current_audio_index = 0  # Reset audio index for the new label
            self.update_ui()
        else:
            self.status_label.config(text="No previous label.", fg="orange")

    def next_label(self, event=None):
        """Move to the next label."""
        if self.current_label_index < len(self.labels_data) - 1:
            # Increment the label index and reset the audio index
            self.current_label_index += 1
            self.current_audio_index = 0
            self.update_ui()
        else:
            # If at the last label, update the UI to indicate completion
            self.status_label.config(text="All labels have been finalised.", fg="green")
            self.finalisation_status_label.config(text="All labels complete", fg="green")
            print("DEBUG: Reached the last label. No further labels to process.")





    def prevent_double_tap(self):
        """Helper to prevent accidental double-tap of A or D keys."""
        current_time = time.time()
        if current_time - self.last_keypress_time < 0.5:  # 0.5-second cooldown
            return False
        self.last_keypress_time = current_time
        return True
    
    def flag_collision(self, event=None):
        """Flags a collision during the recording session."""
        if self.recording:
            collision_time = time.time() - self.start_time
            current_event_id = None

            # Associate collision with the current event, if applicable
            if self.current_label_index < len(self.labels_data):
                current_label = list(self.labels_data.keys())[self.current_label_index]
                if self.labels_data[current_label]['played_audio']:
                    current_event_id = f"E{self.event_id}_{self.labels_data[current_label]['played_audio'][-1]}"

            # Log the collision
            self.collisions.append({
                "timestamp": round(collision_time, 2),
                "event_id": current_event_id
            })

            print(f"Collision flagged at {collision_time:.2f} seconds (Event: {current_event_id or 'NA'}).")
            self.status_label.config(
                text=f"Collision flagged at {collision_time:.2f}s (Event: {current_event_id or 'NA'})",
                fg="red"
            )
    
    def save_collisions(self):
        """Save only the current session's collisions to the CSV file."""
        print("Saving collisions to Collision CSV...")

        if not self.collisions:
            print("No new collisions to save.")
            return  # Do not write anything if there are no new collisions

        try:
            # Ensure we only write new session's collisions and do not reload old ones
            with open(self.collision_csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                for collision in self.collisions:
                    writer.writerow([
                        self.participant_id or "Unknown",
                        self.condition or "Unknown",
                        collision["timestamp"],
                        collision["event_id"] or "NA"
                    ])
            print("New session's collisions saved successfully.")
        except Exception as e:
            print(f"Error saving collisions: {e}")

        # Clear collisions AFTER they are saved to prevent duplication in future sessions
        self.collisions = []




    
    def flag_collision_key(self, event):
        """Handles collision flagging when the L key is pressed."""
        self.flag_collision()


    def flag_question(self):
        """Flags the current question timestamp during recording."""
        if self.recording and self.last_flagged != 'question':
            # Get the elapsed time from the start of recording
            elapsed_time = time.time() - self.start_time
            self.question_times[self.event_id] = f"{elapsed_time:.2f}"

            # Log the flagging
            print(f"Question flagged at {elapsed_time:.2f} seconds (Event ID: E{self.event_id})")

            # Update the last flagged state
            self.last_flagged = 'question'

            # Disable the flag_question_button to prevent multiple flags for the same question
            self.flag_question_button.config(state=tk.DISABLED)

            # Enable the flag_answer_button to allow finalisation of the answer
            self.flag_answer_button.config(state=tk.NORMAL)

            # Update the status label
            self.status_label.config(text=f"Question flagged at {elapsed_time:.2f}s", fg="green")
        else:
            print("Cannot flag question: either not recording or already flagged.")

    def capture_correct_answer(self, event=None):
        """Captures the answer timestamp and marks it as correct."""
        if self.recording:
            if not self.question_played:
                self.status_label.config(text="Cannot flag answer before playing question!", fg="red")
                print("Error: Cannot flag answer before playing the question.")
                return  # Prevent flagging before a question is played

            if not self.prevent_double_tap():
                return  

            self.answer_timestamp = time.time() - self.start_time
            self.is_answer_incorrect = False
            self.status_label.config(text=f"Correct answer logged at {self.answer_timestamp:.2f}s", fg="green")
            print(f"Answer timestamp captured at {self.answer_timestamp:.2f} seconds (Correct)")

            # Enable finalise button
            self.flag_answer_button.config(state=tk.NORMAL)


    def capture_incorrect_answer(self, event=None):
        """Captures the answer timestamp and marks it as incorrect."""
        if self.recording:
            if not self.question_played:
                self.status_label.config(text="Cannot flag answer before playing question!", fg="red")
                print("Error: Cannot flag answer before playing the question.")
                return  # Prevent flagging before a question is played

            if not self.prevent_double_tap():
                return  

            self.answer_timestamp = time.time() - self.start_time
            self.is_answer_incorrect = True
            self.status_label.config(text=f"Incorrect answer logged at {self.answer_timestamp:.2f}s", fg="red")
            print(f"Answer timestamp captured at {self.answer_timestamp:.2f} seconds (Incorrect)")

            # Enable finalise button
            self.flag_answer_button.config(state=tk.NORMAL)


    def toggle_correctness(self, event=None):
        """Toggles the correctness flag without altering the timestamp."""
        if self.recording and self.answer_timestamp is not None:
            self.is_answer_incorrect = not self.is_answer_incorrect
            status = "Incorrect" if self.is_answer_incorrect else "Correct"
            color = "red" if self.is_answer_incorrect else "green"
            self.status_label.config(text=f"Correctness toggled to {status}", fg=color)
            print(f"Correctness toggled. Answer is now marked as {status}")
    
    def set_confidence_rating(self, event):
        """Sets the confidence rating based on the pressed key."""
        self.confidence_rating = int(event.char)
        self.confidence_label.config(text=f"Confidence Rating: {self.confidence_rating}", fg="blue")
        print(f"Confidence rating set to {self.confidence_rating}")

    def finalise_answer(self, event=None):
        """Finalises the answer and ensures a confidence rating is selected."""
        if self.recording:
            if self.answer_timestamp is None:
                self.status_label.config(text="Press A or D to log an answer first", fg="orange")
                print("Error: Attempted to finalise without a logged answer.")
                return

            if self.confidence_rating is None:
                self.status_label.config(text="Please select a confidence rating (1-7) before finalising", fg="red")
                print("Error: Confidence rating not selected.")
                return

            current_label = list(self.labels_data.keys())[self.current_label_index]
            if not self.labels_data[current_label]['played_audio']:
                self.status_label.config(text="No audio has been played for this label.", fg="red")
                print("Error: No audio has been played for this label.")
                return

            selected_audio = self.labels_data[current_label]['played_audio'][-1]
            event_id_with_audio = f"E{self.event_id}_{selected_audio}"

            status = "Incorrect" if self.is_answer_incorrect else "Correct"
            self.correctness_flags[event_id_with_audio] = not self.is_answer_incorrect
            self.answer_times[event_id_with_audio] = f"{self.answer_timestamp:.2f}"
            self.confidence_ratings[event_id_with_audio] = self.confidence_rating  

            print(f"Answer finalised for {event_id_with_audio} at {self.answer_timestamp:.2f} seconds ({status}, Confidence: {self.confidence_rating})")

            self.status_label.config(
                text=f"Answer finalised as {status} for {event_id_with_audio} with Confidence {self.confidence_rating}", fg="blue"
            )

            self.finalisation_status_label.config(
                text=f"Finalised: {current_label} | {status} | Confidence: {self.confidence_rating}", fg="green"
            )

            self.event_id += 1

            # **Reset question tracking for next question**
            self.question_played = False
            self.flag_answer_button.config(state=tk.DISABLED)

            self.answer_timestamp = None
            self.confidence_rating = None
            self.confidence_label.config(text="Confidence Rating: Not Selected", fg="orange")
            self.is_answer_incorrect = False
            self.last_flagged = None

            self.next_label()

            self.flag_question_button.config(state=tk.NORMAL)







    def save_flagged_events(self):
        """Save flagged events to the CSV file."""
        print("Saving flagged events to CSV...")
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                for event_id_with_audio in self.answer_times.keys():
                    question_time = self.question_times.get(event_id_with_audio, '')
                    answer_time = self.answer_times[event_id_with_audio]
                    is_correct = self.correctness_flags.get(event_id_with_audio, True)
                    confidence = self.confidence_ratings.get(event_id_with_audio, "Not Rated")

                    writer.writerow([
                        self.participant_id or "Unknown",
                        self.condition or "Unknown",
                        event_id_with_audio,
                        question_time,
                        answer_time,
                        f"{float(answer_time) - float(question_time):.2f}" if question_time and answer_time else "",
                        "Correct" if is_correct else "Incorrect",
                        confidence
                    ])
                print("Flagged events saved successfully.")
        except Exception as e:
            print(f"Error saving flagged events: {e}")

    


    def flag_question_key(self, event):
        """Flags the question timestamp when Q is pressed."""
        self.flag_question()

    def flag_answer_key(self, event):
        self.flag_answer()

    def load_last_event_id(self):
        last_event_id = 0
        try:
            with open(self.csv_filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['Participant ID'] == self.participant_id and row['Condition'] == self.condition:
                        event_id = int(row['Event ID'][1:])
                        last_event_id = max(last_event_id, event_id)
        except Exception as e:
            print(f"Error loading last event ID: {e}")
        self.event_id = last_event_id + 1

    def create_participant_directory(self):
        self.participant_directory = os.path.join(self.main_directory, f"participant_{self.participant_id}")
        os.makedirs(self.participant_directory, exist_ok=True)

    def update_clock(self):
        if self.recording:
            elapsed_time = int(time.time() - self.start_time)
            minutes, seconds = divmod(elapsed_time, 60)
            self.recording_time_label.config(text=f"Recording Time: {minutes:02}:{seconds:02}")
        self.root.after(1000, self.update_clock)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Shutting down...")
        if hasattr(app, 'stream') and app.stream:
            app.stream.stop()
            app.stream.close()
        root.quit()