import os
import speech_recognition as sr
from typing import Tuple
import json
from datetime import datetime

from audio.processor import AudioProcessor
from nlp.analyzer import NLPAnalyzer
from ml.detector import DepressionAnxietyDetector

class VocalMind:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.nlp_analyzer = NLPAnalyzer()
        self.detector = DepressionAnxietyDetector()
        self.recognizer = sr.Recognizer()
        
        # Create necessary directories
        os.makedirs('data/recordings', exist_ok=True)
        os.makedirs('data/results', exist_ok=True)
        
    def record_and_analyze(self, duration: float = 60.0) -> Tuple[float, float]:
        """
        Record audio and analyze for depression and anxiety.
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            Tuple[float, float]: Depression and anxiety scores
        """
        print(f"Recording for {duration} seconds...")
        audio_data = self.audio_processor.record_audio(duration)
        
        # Save recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_path = f"data/recordings/recording_{timestamp}.wav"
        self.audio_processor.save_audio(audio_data, recording_path)
        
        # Extract audio features
        audio_features = self.audio_processor.extract_features(audio_data)
        
        # Convert speech to text
        with sr.AudioFile(recording_path) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None, None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None, None
        
        # Analyze text
        nlp_features = self.nlp_analyzer.analyze_text(text)
        
        # Get predictions
        depression_score, anxiety_score = self.detector.predict(audio_features, nlp_features)
        
        # Save results
        results = {
            'timestamp': timestamp,
            'text': text,
            'audio_features': audio_features,
            'nlp_features': nlp_features,
            'depression_score': depression_score,
            'anxiety_score': anxiety_score
        }
        
        results_path = f"data/results/analysis_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return depression_score, anxiety_score
    
    def train_models(self, training_data_path: str) -> None:
        """
        Train the models using historical data.
        
        Args:
            training_data_path (str): Path to training data JSON file
        """
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        audio_features_list = [item['audio_features'] for item in training_data]
        nlp_features_list = [item['nlp_features'] for item in training_data]
        depression_labels = [item['depression_score'] for item in training_data]
        anxiety_labels = [item['anxiety_score'] for item in training_data]
        
        self.detector.train(
            audio_features_list,
            nlp_features_list,
            depression_labels,
            anxiety_labels
        )

def main():
    # Initialize VocalMind
    vocal_mind = VocalMind()
    
    # Example usage
    print("Welcome to VocalMind - Depression and Anxiety Detection")
    print("This tool will record your voice and analyze it for signs of depression and anxiety.")
    print("Please speak clearly and naturally for the best results.")
    
    try:
        depression_score, anxiety_score = vocal_mind.record_and_analyze(duration=60.0)
        
        if depression_score is not None and anxiety_score is not None:
            print("\nAnalysis Results:")
            print(f"Depression Score: {depression_score:.2f}")
            print(f"Anxiety Score: {anxiety_score:.2f}")
            
            print("\nImportant Note:")
            print("This tool is designed to assist in the detection of potential signs of depression and anxiety.")
            print("It is not a substitute for professional medical advice, diagnosis, or treatment.")
            print("If you are experiencing symptoms of depression or anxiety, please consult with a mental health professional.")
        else:
            print("\nAnalysis could not be completed. Please try again.")
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 