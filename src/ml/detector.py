import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import torch
import torch.nn as nn

class DepressionAnxietyDetector:
    def __init__(self):
        self.audio_scaler = StandardScaler()
        self.nlp_scaler = StandardScaler()
        self.audio_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nlp_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.combined_model = self._create_neural_network()
        
    def _create_neural_network(self) -> nn.Module:
        """
        Create a neural network for combined feature analysis.
        
        Returns:
            nn.Module: PyTorch neural network model
        """
        class CombinedModel(nn.Module):
            def __init__(self):
                super(CombinedModel, self).__init__()
                self.audio_encoder = nn.Sequential(
                    nn.Linear(50, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16)
                )
                self.nlp_encoder = nn.Sequential(
                    nn.Linear(40, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(16, 2),
                    nn.Sigmoid()
                )
                
            def forward(self, audio_features, nlp_features):
                audio_encoded = self.audio_encoder(audio_features)
                nlp_encoded = self.nlp_encoder(nlp_features)
                combined = torch.cat([audio_encoded, nlp_encoded], dim=1)
                return self.classifier(combined)
        
        return CombinedModel()

    def prepare_audio_features(self, audio_features: Dict) -> np.ndarray:
        """
        Prepare audio features for model input.
        
        Args:
            audio_features (Dict): Dictionary of audio features
            
        Returns:
            np.ndarray: Processed audio features
        """
        # Extract and combine relevant features
        features = []
        
        # MFCC features
        mfcc_mean = np.mean(audio_features['mfcc'], axis=1)
        features.extend(mfcc_mean)
        
        # Pitch features
        pitch_mean = np.mean(audio_features['pitch'], axis=1)
        features.extend(pitch_mean)
        
        # Energy features
        energy_mean = np.mean(audio_features['energy'], axis=0)
        features.extend(energy_mean)
        
        # Zero-crossing rate
        zcr_mean = np.mean(audio_features['zero_crossing_rate'], axis=0)
        features.extend(zcr_mean)
        
        return np.array(features)

    def prepare_nlp_features(self, nlp_features: Dict) -> np.ndarray:
        """
        Prepare NLP features for model input.
        
        Args:
            nlp_features (Dict): Dictionary of NLP features
            
        Returns:
            np.ndarray: Processed NLP features
        """
        features = []
        
        # Sentiment features
        sentiment_score = nlp_features['sentiment']['score']
        features.append(sentiment_score)
        
        # Depression indicators
        for category, count in nlp_features['depression_indicators'].items():
            features.append(count)
            
        # Anxiety indicators
        for category, count in nlp_features['anxiety_indicators'].items():
            features.append(count)
            
        # Speech pattern features
        speech_patterns = nlp_features['speech_patterns']
        features.extend([
            speech_patterns['avg_sentence_length'],
            speech_patterns['sentence_count'],
            speech_patterns['word_count']
        ])
        
        return np.array(features)

    def predict(self, audio_features: Dict, nlp_features: Dict) -> Tuple[float, float]:
        """
        Predict depression and anxiety scores.
        
        Args:
            audio_features (Dict): Audio feature dictionary
            nlp_features (Dict): NLP feature dictionary
            
        Returns:
            Tuple[float, float]: Depression and anxiety scores
        """
        # Prepare features
        audio_feat = self.prepare_audio_features(audio_features)
        nlp_feat = self.prepare_nlp_features(nlp_features)
        
        # Scale features
        audio_feat_scaled = self.audio_scaler.transform(audio_feat.reshape(1, -1))
        nlp_feat_scaled = self.nlp_scaler.transform(nlp_feat.reshape(1, -1))
        
        # Convert to PyTorch tensors
        audio_tensor = torch.FloatTensor(audio_feat_scaled)
        nlp_tensor = torch.FloatTensor(nlp_feat_scaled)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.combined_model(audio_tensor, nlp_tensor)
            
        return predictions[0][0].item(), predictions[0][1].item()

    def train(self, audio_features_list: List[Dict], nlp_features_list: List[Dict], 
              depression_labels: List[float], anxiety_labels: List[float]) -> None:
        """
        Train the models on the provided data.
        
        Args:
            audio_features_list (List[Dict]): List of audio feature dictionaries
            nlp_features_list (List[Dict]): List of NLP feature dictionaries
            depression_labels (List[float]): Depression scores for training
            anxiety_labels (List[float]): Anxiety scores for training
        """
        # Prepare training data
        X_audio = np.array([self.prepare_audio_features(feat) for feat in audio_features_list])
        X_nlp = np.array([self.prepare_nlp_features(feat) for feat in nlp_features_list])
        
        # Scale features
        X_audio_scaled = self.audio_scaler.fit_transform(X_audio)
        X_nlp_scaled = self.nlp_scaler.fit_transform(X_nlp)
        
        # Train individual models
        self.audio_model.fit(X_audio_scaled, depression_labels)
        self.nlp_model.fit(X_nlp_scaled, anxiety_labels)
        
        # Train combined model
        self._train_combined_model(X_audio_scaled, X_nlp_scaled, depression_labels, anxiety_labels)

    def _train_combined_model(self, X_audio: np.ndarray, X_nlp: np.ndarray,
                            depression_labels: List[float], anxiety_labels: List[float]) -> None:
        """
        Train the combined neural network model.
        
        Args:
            X_audio (np.ndarray): Scaled audio features
            X_nlp (np.ndarray): Scaled NLP features
            depression_labels (List[float]): Depression scores
            anxiety_labels (List[float]): Anxiety scores
        """
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.combined_model.parameters())
        
        # Convert data to PyTorch tensors
        X_audio_tensor = torch.FloatTensor(X_audio)
        X_nlp_tensor = torch.FloatTensor(X_nlp)
        y_tensor = torch.FloatTensor([[d, a] for d, a in zip(depression_labels, anxiety_labels)])
        
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.combined_model(X_audio_tensor, X_nlp_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step() 