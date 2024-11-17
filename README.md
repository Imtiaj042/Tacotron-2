import torch
import numpy as np
import librosa
import pyaudio
import wave
import soundfile as sf
from synthesizer import Tacotron2, WaveNet, SpeakerEncoder
from denoising_model import DenoiseModel

# Loading the models
speaker_encoder = SpeakerEncoder()
speaker_encoder.load_model('path_to_speaker_encoder_model.pth')

tacotron2 = Tacotron2()
tacotron2.load_model('path_to_tacotron2_model.pth')

wavenet = WaveNet()
wavenet.load_model('path_to_wavenet_model.pth')

denoise_model = DenoiseModel()
denoise_model.load_model('path_to_denoising_model.pth')

# Function to generate speech from text
def generate_speech_from_text(text, speaker_id):
    # Extracting speaker embedding
    speaker_embedding = speaker_encoder.get_embedding(speaker_id)
    
    # Generating mel spectrogram using Tacotron2
    mel_spectrogram = tacotron2.text_to_mel(text, speaker_embedding)

    # Generating audio using WaveNet
    audio = wavenet.mel_to_audio(mel_spectrogram)

    # Denoising the audio
    clean_audio = denoise_model.denoise_audio(audio)

    # Saving the audio
    sf.write('generated_speech_with_speaker.wav', clean_audio, 22050)
    print("Speech generated successfully for the speaker ID!")

# Function for real-time voice cloning
def real_time_voice_cloning():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    
    print("Recording...")
    frames = []

    # Recording for 5 seconds
    for i in range(0, int(16000 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Saving the recorded audio
    wf = wave.open('real_time_audio.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Denoising the recorded audio
    clean_audio = denoise_model.denoise_audio('real_time_audio.wav')

    # Saving the denoised audio
    sf.write('clean_real_time_audio.wav', clean_audio, 22050)
    print("Real-time voice cloning done.")

# Function to fine-tune the model
def fine_tune_model():
    tacotron2.train_model('path_to_your_custom_dataset')

    # Saving the fine-tuned model
    tacotron2.save_model('fine_tuned_tacotron2_model.pth')
    print("Model fine-tuned and saved!")

# Example of generating speech with speaker embeddings
generate_speech_from_text("Hello, this is a test message for voice cloning.", speaker_id=3)

# Example of real-time voice cloning
real_time_voice_cloning()

# Starting fine-tuning
fine_tune_model()


