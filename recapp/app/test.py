import sounddevice as sd
import numpy as np

def play_test_tone(frequency=1000, duration=0.5, sample_rate=44100, volume=0.8):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    tone = tone.astype(np.float32)  # Use float32 for better compatibility with sounddevice

    print("Playing test tone...")
    sd.play(tone, samplerate=sample_rate)
    sd.wait()  # Wait until the tone is played completely
    print("Tone playback complete.")

if __name__ == "__main__":
    play_test_tone()
