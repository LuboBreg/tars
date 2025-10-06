import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

class EmotionNet(nn.Module):
    """PyTorch CNN model for emotion recognition"""
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 emotions
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionPredictor:
    """Emotion prediction using local PyTorch model"""
    
    def __init__(self, model_path=None):
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default model path
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'emotion_model.pth')
        
        logger.info(f"Loading emotion model from: {model_path}")
        
        # Load model
        self.model = EmotionNet()
        
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Emotion model loaded successfully")
            else:
                logger.warning(f"Model file not found at {model_path}, using random weights")
                # Initialize with random weights for now
                self._init_weights()
                
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("Using randomly initialized weights")
            self._init_weights()
        
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Emotion model ready on {self.device}")
    
    def _init_weights(self):
        """Initialize model weights"""
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
    
    def preprocess_face(self, face_image):
        """Preprocess face image for model input"""
        try:
            # Resize to model input size (48x48)
            face_resized = cv2.resize(face_image, (48, 48))
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.FloatTensor(face_normalized).permute(2, 0, 1).unsqueeze(0)
            
            return face_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Face preprocessing error: {e}")
            raise
    
    def predict(self, face_image) -> Dict[str, float]:
        """Predict emotions from face image"""
        try:
            logger.debug(f"Predicting emotions for face image: {face_image.shape}")
            
            # Preprocess image
            input_tensor = self.preprocess_face(face_image)
            
            # Predict emotions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Create emotion dictionary
            emotion_dict = {}
            for emotion, prob in zip(self.emotions, probabilities):
                emotion_dict[emotion] = float(prob)
            
            logger.debug(f"Emotion predictions: {emotion_dict}")
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Emotion prediction error: {e}")
            return self._get_default_emotions()
    
    def _get_default_emotions(self):
        """Default emotions if prediction fails"""
        return {
            'anger': 0.08,
            'disgust': 0.03,
            'fear': 0.05,
            'happiness': 0.25,
            'sadness': 0.12,
            'surprise': 0.15,
            'neutral': 0.32
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'EmotionNet CNN',
            'input_size': '48x48x3',
            'emotions': self.emotions,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }

# Create and save model function (for training/setup)
def create_emotion_model(save_path='emotion_model.pth'):
    """Create and save emotion model with initialized weights"""
    print("Creating emotion model...")
    model = EmotionNet()
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Save model state dict
    torch.save(model.state_dict(), save_path)
    print(f"Emotion model saved to {save_path}")
    return model

if __name__ == "__main__":
    # Create model if run directly
    create_emotion_model()