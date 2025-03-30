import numpy as np
import sounddevice as sd
import librosa
from scipy.io import wavfile
import os
from typing import Tuple, Dict, Optional, List

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = None
        self.is_recording = False

    def record_audio(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration.
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            np.ndarray: Recorded audio data
        """
        self.is_recording = True
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        self.is_recording = False
        return self.recording

    def save_audio(self, audio_data: np.ndarray, filename: str) -> None:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data (np.ndarray): Audio data to save
            filename (str): Output filename
        """
        wavfile.write(filename, self.sample_rate, audio_data)

    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from a file.
        
        Args:
            filename (str): Input audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        audio_data, sample_rate = librosa.load(filename, sr=self.sample_rate)
        return audio_data, sample_rate

    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract relevant features from audio data.
        
        Args:
            audio_data (np.ndarray): Audio data to process
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        features = {}
        
        # Extract spectral features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        features['mfcc'] = mfccs
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        features['pitch'] = pitches
        
        # Extract energy
        energy = librosa.feature.rms(y=audio_data)
        features['energy'] = energy
        
        # Extract zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        features['zero_crossing_rate'] = zcr
        
        return features

    def detect_pauses(self, audio_data: np.ndarray, threshold: float = 0.01) -> List[Tuple[float, float]]:
        """
        Detect pauses in the audio.
        
        Args:
            audio_data (np.ndarray): Audio data to analyze
            threshold (float): Energy threshold for pause detection
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) for each pause
        """
        energy = librosa.feature.rms(y=audio_data)[0]
        pause_indices = np.where(energy < threshold)[0]
        
        pauses = []
        if len(pause_indices) > 0:
            start_idx = pause_indices[0]
            for i in range(1, len(pause_indices)):
                if pause_indices[i] - pause_indices[i-1] > 1:
                    pauses.append((start_idx/self.sample_rate, pause_indices[i-1]/self.sample_rate))
                    start_idx = pause_indices[i]
            pauses.append((start_idx/self.sample_rate, pause_indices[-1]/self.sample_rate))
        
        return pauses 