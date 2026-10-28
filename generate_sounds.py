import numpy as np
import wave
import struct

def generate_beep(filename, freq, duration, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    # Generate a sine wave
    audio = np.sin(freq * t * 2 * np.pi)
    
    # Fade in/out to avoid clicks
    fade_duration = 0.05
    fade_samples = int(sample_rate * fade_duration)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    if len(audio) > 2 * fade_samples:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    
    # Convert to 16-bit PCM
    audio_data = np.int16(audio * 32767)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for sample in audio_data:
            wav_file.writeframes(struct.pack('<h', sample))

generate_beep("beep_mode1.wav", 440.0, 0.2)  # A4
generate_beep("beep_mode2.wav", 523.25, 0.2)  # C5
generate_beep("beep_mode3.wav", 659.25, 0.2)  # E5
generate_beep("beep_mode4.wav", 880.0, 0.3)  # A5

print("Sound effects generated.")
