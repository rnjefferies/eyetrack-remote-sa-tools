import sys
import os
import numpy as np
import sounddevice as sd
import tkinter as tk
import time
import csv
import wave
from PIL import Image, ImageTk
from tkinter import messagebox
import threading

# Ensure the scripts directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.audio_player import AudioPlayer

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recapp: for SA Data Collection")

        # Initialise label and audio management
        self.labels_data = {} # Will be loaded after route selection
        self.collisions = []
        self.current_label_index = 0

        self.set_app_icon()
        self.route_audio_directory = None
        self.recording = False
        self.start_time = None
        self.recorded_chunks = []
        self.event_id = 1
        self.participant_id = None
        self.route = None
        self.confidence_ratings = {}  # Store confidence ratings by event ID
        self.collision_flags = {}  # Tracks collision status (Yes/No) by event ID
        self.answer_timestamp = None    # Holds the timestamp for the answer
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

        validate_route_command = root.register(self.validate_route_input)

        # UI setup
        self.content_frame = tk.Frame(root)
        self.content_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.participant_id_label = tk.Label(self.content_frame, text="Participant ID:")
        self.participant_id_label.pack(pady=(10, 5))

        self.participant_id_entry = tk.Entry(self.content_frame)
        self.participant_id_entry.pack(pady=(0, 20))

        self.condition_label = tk.Label(self.content_frame, text="Route (1-9):")
        self.condition_label.pack(pady=(10, 5))

        self.condition_entry = tk.Entry(self.content_frame, validate="key", validatecommand=(validate_route_command, '%P'))
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

    def load_labels_for_route(self, route):
        """Loads the specific 3 labels and their designated audio file for the given route."""
        self.labels_data = {}  # Clear previous data
        csv_path = 'labels.csv'
        route = int(route)

        # Determine the start and end index for the labels based on the route
        start_index = (route - 1) * 3
        end_index = start_index + 3

        try:
            with open(csv_path, 'r') as file:
                all_rows = list(csv.DictReader(file))
                
                if len(all_rows) < end_index:
                    messagebox.showerror("Error", f"Not enough labels in labels.csv for Route {route}.")
                    return False

                # Get the 3 specific rows for the current route
                route_rows = all_rows[start_index:end_index]

                for row in route_rows:
                    label = row['Label']
                    # Directly get the audio file from the single audio column (e.g., 'Audio1')
                    specific_audio_file = row['Audio1']
                    
                    # Store the single, specific audio file for this label
                    if label not in self.labels_data:
                        self.labels_data[label] = {'audio_file': specific_audio_file, 'played_audio': []}
                
                return True  # Indicate success

        except KeyError:
            messagebox.showerror("Error", "Could not find the 'Audio1' column in labels.csv. Please check your file.")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load labels for route: {e}")
            return False

    def update_ui(self):
        """Updates the UI to display the current label."""
        if not self.labels_data or self.current_label_index >= len(self.labels_data):
            self.label_label.config(text="Current Label: None")
            self.audio_label.config(text="Current Audio: None")
            return

        current_label = list(self.labels_data.keys())[self.current_label_index]
        if current_label not in self.labels_data:
            self.status_label.config(text=f"Error: Label '{current_label}' not found.", fg="red")
            return

        audio_file = self.labels_data[current_label].get('audio_file', 'N/A')
        self.label_label.config(text=f"Current Label: {current_label}")
        self.audio_label.config(text=f"Audio File: {audio_file}")

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
        pattern = r"^\d{2}_(OL|RT)$"
        return re.match(pattern, input) is not None or input == ''

    def validate_route_input(self, value):
        """Allows only a single digit from 1 to 9."""
        if value == "":
            return True
        return len(value) == 1 and value in "123456789"

    def ensure_csv_file_exists(self):
        if not os.path.isfile(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant ID', 'Route', 'Event ID', 'Question Timestamp (s)',
                                 'Answer Timestamp (s)', 'Time Difference (s)', 'Answer Accuracy', 'Confidence'])

    def ensure_collision_csv_file_exists(self):
        if not os.path.isfile(self.collision_csv_filename):
            with open(self.collision_csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant ID', 'Route', 'Collision Timestamp (s)', 'Event ID'])

    def handle_number_key(self, event):
        """Handles number key presses for Confidence Rating input."""
        if self.recording and event.char in "1234567":
            self.set_confidence_rating(event)
        return "break"

    def start_recording(self):
        participant_id_input = self.participant_id_entry.get().strip()
        route_input = self.condition_entry.get().strip()

        if not participant_id_input or not self.validate_numeric_input(participant_id_input):
            self.status_label.config(text="Invalid Participant ID!", fg="red")
            return

        if not (route_input and route_input in "123456789"):
            self.status_label.config(text="Invalid Route! Must be 1-9.", fg="red")
            return

        self.collisions = []
        self.collision_flags = {}

        # Load the specific labels for this route
        if not self.load_labels_for_route(route_input):
            return # Stop if labels can't be loaded

        self.route_audio_directory = os.path.join(self.audio_directory, route_input)
        if not os.path.exists(self.route_audio_directory):
            messagebox.showerror("Error", f"Audio directory for Route {route_input} not found!")
            return

        if self.participant_id != participant_id_input or self.route != route_input:
            self.participant_id = participant_id_input
            self.route = route_input
            self.event_id = 1
            self.question_times = {}
            self.answer_times = {}
            self.last_flagged = None
            self.create_participant_directory()
            self.load_last_event_id()

        self.current_label_index = 0 # Reset for the new session
        self.update_ui() # Display the first label

        self.recording = True
        self.start_time = time.time()
        self.recorded_chunks = []
        self.status_label.config(text="Status: Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.flag_question_button.config(state=tk.NORMAL)
        self.flag_answer_button.config(state=tk.NORMAL)
        self.flag_collision_button.config(state=tk.NORMAL)
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
            self.flag_collision_button.config(state=tk.DISABLED)
            self.participant_id_entry.config(state=tk.NORMAL)
            self.condition_entry.config(state=tk.NORMAL)

            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

            filename = f"recording_{self.participant_id}_R{self.route}_{int(time.time())}.wav"
            filepath = os.path.join(self.participant_directory, filename)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(np.iinfo(self.dtype).bits // 8)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_chunks))
            print(f"Recording saved as {filepath}")

            self.save_flagged_events()
            self.save_collisions()

    def prepare_audio_for_playback(self):
        """Prepares the next sequential audio file for playback based on the current label index."""
        if self.current_label_index >= len(self.labels_data):
            self.status_label.config(text="All labels for this route have been played.", fg="green")
            return None, None

        # Get the current label and its data
        current_label = list(self.labels_data.keys())[self.current_label_index]
        label_info = self.labels_data[current_label]
        
        # Get the pre-determined audio file for this label
        audio_filename = label_info['audio_file']
        
        # Construct the full path to the audio file
        audio_filepath = os.path.join(self.route_audio_directory, audio_filename)

        if not os.path.exists(audio_filepath):
            self.status_label.config(text=f"Audio file not found: {audio_filename}", fg="red")
            messagebox.showerror("Error", f"Audio file not found at path: {audio_filepath}")
            return None, None

        return current_label, audio_filepath

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.recorded_chunks.append(indata.tobytes())

    def play_current_audio(self, event=None):
        current_label, selected_audio_path = self.prepare_audio_for_playback()
        if not current_label or not selected_audio_path:
            return

        self.finalisation_status_label.config(text="Not finalised", fg="red")
        selected_audio_filename = os.path.basename(selected_audio_path)

        # Store question timestamp
        elapsed_time = time.time() - self.start_time
        event_id_with_audio = f"E{self.event_id}_{selected_audio_filename}"
        self.question_times[event_id_with_audio] = f"{elapsed_time:.2f}"
        self.last_flagged = 'question'
        self.question_played = True
        
        # Log the audio file that was played for this label
        self.labels_data[current_label]['played_audio'].append(selected_audio_filename)
        self.label_label.config(text=f"Current Label: {current_label}")
        self.audio_label.config(text=f"Current Audio: {selected_audio_filename}")

        if os.path.exists(selected_audio_path):
            self.play_audio_in_thread(selected_audio_path)
        else:
            messagebox.showerror("Error", f"Audio file {selected_audio_path} not found!")

    def play_audio_in_thread(self, audio_file):
        self.audio_player = AudioPlayer(audio_file, on_playback_end=self.auto_flag_end_of_audio)
        self.audio_player.play()

    def auto_flag_end_of_audio(self):
        if self.current_label_index >= len(self.labels_data):
            return

        current_label = list(self.labels_data.keys())[self.current_label_index]
        if not self.labels_data[current_label]['played_audio']:
            return

        # Commented out block of code for storing question timestamp at end of question audio. 
        # selected_audio = self.labels_data[current_label]['played_audio'][-1]
        # elapsed_time = time.time() - self.start_time
        # event_id_with_audio = f"E{self.event_id}_{selected_audio}"
        # self.question_times[event_id_with_audio] = f"{elapsed_time:.2f}"
        # self.last_flagged = 'question'
        # self.question_played = True

    def update_ui_post_playback(self):
        self.flag_question_button.config(state=tk.DISABLED)
        self.flag_answer_button.config(state=tk.NORMAL)
        self.status_label.config(text="Audio playback complete.", fg="green")

    def previous_label(self, event=None):
        if self.current_label_index > 0:
            self.current_label_index -= 1
            self.update_ui()

    def next_label(self, event=None):
        if self.current_label_index < len(self.labels_data) - 1:
            self.current_label_index += 1
            self.update_ui()
        else:
            self.status_label.config(text="All labels for this route have been finalised.", fg="green")
            self.finalisation_status_label.config(text="Route complete", fg="green")

    def prevent_double_tap(self):
        current_time = time.time()
        if current_time - self.last_keypress_time < 0.5:
            return False
        self.last_keypress_time = current_time
        return True

    def flag_collision(self, event=None):
        if self.recording:
            collision_time = time.time() - self.start_time
            current_event_id = None
            if self.current_label_index < len(self.labels_data):
                current_label = list(self.labels_data.keys())[self.current_label_index]
                if self.labels_data[current_label]['played_audio']:
                    current_event_id = f"E{self.event_id}_{self.labels_data[current_label]['played_audio'][-1]}"
            self.collisions.append({"timestamp": round(collision_time, 2), "event_id": current_event_id})
            self.status_label.config(text=f"Collision flagged at {collision_time:.2f}s", fg="red")

    def save_collisions(self):
        if not self.collisions:
            return
        with open(self.collision_csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            for collision in self.collisions:
                writer.writerow([self.participant_id or "Unknown", self.route or "Unknown",
                                 collision["timestamp"], collision["event_id"] or "NA"])
        self.collisions = []

    def flag_collision_key(self, event):
        self.flag_collision()

    def flag_question(self):
        if self.recording and self.last_flagged != 'question':
            elapsed_time = time.time() - self.start_time
            self.question_times[self.event_id] = f"{elapsed_time:.2f}"
            self.last_flagged = 'question'
            self.flag_question_button.config(state=tk.DISABLED)
            self.flag_answer_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Question flagged at {elapsed_time:.2f}s", fg="green")

    def capture_correct_answer(self, event=None):
        if self.recording and self.question_played and self.prevent_double_tap():
            self.answer_timestamp = time.time() - self.start_time
            self.is_answer_incorrect = False
            self.status_label.config(text=f"Correct answer logged at {self.answer_timestamp:.2f}s", fg="green")
            self.flag_answer_button.config(state=tk.NORMAL)

    def capture_incorrect_answer(self, event=None):
        if self.recording and self.question_played and self.prevent_double_tap():
            self.answer_timestamp = time.time() - self.start_time
            self.is_answer_incorrect = True
            self.status_label.config(text=f"Incorrect answer logged at {self.answer_timestamp:.2f}s", fg="red")
            self.flag_answer_button.config(state=tk.NORMAL)

    def toggle_correctness(self, event=None):
        if self.recording and self.answer_timestamp is not None:
            self.is_answer_incorrect = not self.is_answer_incorrect
            status = "Incorrect" if self.is_answer_incorrect else "Correct"
            color = "red" if self.is_answer_incorrect else "green"
            self.status_label.config(text=f"Correctness toggled to {status}", fg=color)

    def set_confidence_rating(self, event):
        self.confidence_rating = int(event.char)
        self.confidence_label.config(text=f"Confidence Rating: {self.confidence_rating}", fg="blue")

    def finalise_answer(self, event=None):
        if self.recording:
            if self.answer_timestamp is None:
                self.status_label.config(text="Press A or D to log an answer first", fg="orange")
                return
            if not hasattr(self, 'confidence_rating') or self.confidence_rating is None:
                self.status_label.config(text="Please select a confidence rating (1-7)", fg="red")
                return

            current_label = list(self.labels_data.keys())[self.current_label_index]
            if not self.labels_data[current_label]['played_audio']:
                return

            selected_audio = self.labels_data[current_label]['played_audio'][-1]
            event_id_with_audio = f"E{self.event_id}_{selected_audio}"
            status = "Incorrect" if self.is_answer_incorrect else "Correct"
            self.correctness_flags[event_id_with_audio] = not self.is_answer_incorrect
            self.answer_times[event_id_with_audio] = f"{self.answer_timestamp:.2f}"
            self.confidence_ratings[event_id_with_audio] = self.confidence_rating

            self.status_label.config(text=f"Finalised: {status} | Confidence: {self.confidence_rating}", fg="blue")
            self.finalisation_status_label.config(text=f"Finalised: {current_label}", fg="green")
            self.event_id += 1
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
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            for event_id_with_audio, answer_time in self.answer_times.items():
                question_time = self.question_times.get(event_id_with_audio, '')
                is_correct = self.correctness_flags.get(event_id_with_audio, True)
                confidence = self.confidence_ratings.get(event_id_with_audio, "Not Rated")
                writer.writerow([
                    self.participant_id or "Unknown",
                    self.route or "Unknown",
                    event_id_with_audio,
                    question_time,
                    answer_time,
                    f"{float(answer_time) - float(question_time):.2f}" if question_time else "",
                    "Correct" if is_correct else "Incorrect",
                    confidence
                ])

    def flag_question_key(self, event):
        self.flag_question()

    def flag_answer_key(self, event):
        self.finalise_answer()

    def load_last_event_id(self):
        last_event_id = 0
        try:
            with open(self.csv_filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['Participant ID'] == self.participant_id and row['Route'] == self.route:
                        event_id_str = ''.join(filter(str.isdigit, row['Event ID'].split('_')[0]))
                        if event_id_str:
                            event_id = int(event_id_str)
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

# The following lines should be at the end of your recapp9.py file
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        if hasattr(app, 'stream') and app.stream.is_active():
            app.stream.stop()
            app.stream.close()
        root.quit()