from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Tuple
import numpy as np

class NLPAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Initialize depression-related keywords
        self.depression_keywords = {
            'negative_emotions': ['sad', 'hopeless', 'worthless', 'empty', 'lonely'],
            'physical_symptoms': ['tired', 'exhausted', 'sleep', 'appetite'],
            'cognitive': ['concentrate', 'decide', 'remember', 'think'],
            'suicidal': ['suicide', 'die', 'death', 'end']
        }
        
        # Initialize anxiety-related keywords
        self.anxiety_keywords = {
            'worry': ['worry', 'anxious', 'fear', 'panic'],
            'physical': ['heart', 'breath', 'sweat', 'tremble'],
            'avoidance': ['avoid', 'escape', 'run', 'hide'],
            'cognitive': ['thoughts', 'racing', 'mind', 'control']
        }

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for depression and anxiety indicators.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Analysis results including sentiment and keyword matches
        """
        results = {
            'sentiment': self._analyze_sentiment(text),
            'depression_indicators': self._analyze_depression_indicators(text),
            'anxiety_indicators': self._analyze_anxiety_indicators(text),
            'speech_patterns': self._analyze_speech_patterns(text)
        }
        return results

    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze the overall sentiment of the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Sentiment analysis results
        """
        sentiment_result = self.sentiment_analyzer(text[:512])[0]  # Limit text length for API
        return {
            'label': sentiment_result['label'],
            'score': sentiment_result['score']
        }

    def _analyze_depression_indicators(self, text: str) -> Dict:
        """
        Analyze text for depression-related indicators.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Depression indicators and their frequencies
        """
        text = text.lower()
        indicators = {}
        
        for category, keywords in self.depression_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            indicators[category] = count
            
        return indicators

    def _analyze_anxiety_indicators(self, text: str) -> Dict:
        """
        Analyze text for anxiety-related indicators.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Anxiety indicators and their frequencies
        """
        text = text.lower()
        indicators = {}
        
        for category, keywords in self.anxiety_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            indicators[category] = count
            
        return indicators

    def _analyze_speech_patterns(self, text: str) -> Dict:
        """
        Analyze speech patterns and structure.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Speech pattern analysis results
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Calculate basic metrics
        avg_sentence_length = np.mean([len(word_tokenize(sentence)) for sentence in sentences])
        sentence_count = len(sentences)
        word_count = len(words)
        
        # Analyze sentence structure
        stop_words = set(stopwords.words('english'))
        word_frequencies = {}
        for word in words:
            if word.lower() not in stop_words:
                word_frequencies[word.lower()] = word_frequencies.get(word.lower(), 0) + 1
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'word_frequencies': word_frequencies
        } 