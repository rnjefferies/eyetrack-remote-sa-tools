# ============================================================================
# audio_player.py  —  threaded WAV playback with end-tone and completion callback
# ============================================================================
# Purpose:  AudioPlayer class used by Recapp to play a clip, emit an
#           end-of-clip tone, and fire a callback on completion.
# Requires: sounddevice, numpy
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import sounddevice as sd
import numpy as np
import wave
import os
import threading

class AudioPlayer:
    def __init__(self, filename, end_tone_frequency=500, end_tone_duration=0.5, volume=0.4, tone_file="end_tone.wav", on_playback_end=None):
        self.filename = filename
        self.end_tone_frequency = end_tone_frequency
        self.end_tone_duration = end_tone_duration
        self.volume = volume
        self.tone_file = tone_file
        self.on_playback_end = on_playback_end  # Store the callback

        # Generate and save the tone if it doesn't already exist
        if not os.path.exists(self.tone_file):
            self.save_end_tone(self.tone_file)

    def play(self):
        # Start playback in a new thread
        threading.Thread(target=self._play_audio).start()

    def _play_audio(self):
        try:
            print("Starting main audio playback...")
            with wave.open(self.filename, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                dtype = np.int16

                def callback(outdata, frames, time, status):
                    data = wf.readframes(frames)
                    outdata.fill(0)
                    data_np = np.frombuffer(data, dtype=dtype)
                    outdata[:len(data_np)] = data_np.reshape(-1, channels)

                    if len(data) < frames * channels * np.dtype(dtype).itemsize:
                        raise sd.CallbackStop

                with sd.OutputStream(channels=channels, samplerate=sample_rate, callback=callback, dtype=dtype, blocksize=4096):
                    sd.sleep(int(wf.getnframes() / sample_rate * 1000))  # Sleep until audio playback is complete

            # self.play_saved_tone(self.tone_file)
        finally:
            print("Playback finished.")
            # Trigger the callback once playback completes (after both the main audio and end tone)
            if self.on_playback_end:
                self.on_playback_end()  # This will call auto_flag_end_of_question in the main app

    def save_end_tone(self, file_path):
        sample_rate = 44100
        t = np.linspace(0, self.end_tone_duration, int(sample_rate * self.end_tone_duration), False)
        tone = np.sin(2 * np.pi * self.end_tone_frequency * t) * self.volume
        tone = tone.astype(np.float32)

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((tone * 32767).astype(np.int16).tobytes())

    def play_saved_tone(self, file_path):
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            dtype = np.int16

            def callback(outdata, frames, time, status):
                data = wf.readframes(frames)
                outdata.fill(0)
                data_np = np.frombuffer(data, dtype=dtype)
                outdata[:len(data_np)] = data_np.reshape(-1, channels)
                if len(data) < frames * channels * np.dtype(dtype).itemsize:
                    raise sd.CallbackStop

            with sd.OutputStream(channels=channels, samplerate=sample_rate, callback=callback, dtype=dtype, blocksize=4096):
                sd.sleep(int(wf.getnframes() / sample_rate * 1000))
        print("End tone playback complete.")
