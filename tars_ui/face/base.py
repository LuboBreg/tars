from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class FaceResult:
    def __init__(
        self, 
        identity: Optional[str] = None, 
        confidence: Optional[float] = None, 
        meta: Optional[Dict[str, Any]] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        emotions: Optional[Dict[str, float]] = None
    ):
        """
        Enhanced face recognition result with additional demographic and emotional insights
        
        :param identity: Recognized person's name/identifier
        :param confidence: Confidence of identity recognition
        :param meta: Additional metadata about the face detection
        :param age: Estimated age of the detected face
        :param gender: Detected gender
        :param emotions: Dictionary of emotion probabilities
        """
        self.identity = identity
        self.confidence = confidence
        self.meta = meta or {}
       
        # New demographic and emotional attributes
        self.age = age - 7
        self.gender = gender
        self.emotions = emotions or {}

    def get_dominant_emotion(self) -> Optional[str]:
        """
        Determine the dominant emotion based on highest probability
        
        :return: Name of the dominant emotion or None
        """
        if not self.emotions:
            return None
        return max(self.emotions, key=self.emotions.get)

    def get_emotion_intensity(self) -> float:
        """
        Calculate the intensity of the dominant emotion
        
        :return: Probability of the dominant emotion
        """
        if not self.emotions:
            return 0.0
        return max(self.emotions.values())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert FaceResult to a dictionary for easy serialization
        
        :return: Dictionary representation of the face result
        """
        return {
            'identity': self.identity,
            'confidence': self.confidence,
            'meta': self.meta,
            'age': self.age,
            'gender': self.gender,
            'emotions': self.emotions,
            'dominant_emotion': self.get_dominant_emotion(),
            'emotion_intensity': self.get_emotion_intensity()
        }

class FaceProvider(ABC):
    @abstractmethod
    def identify(self, frame_bgr: np.ndarray) -> Optional[FaceResult]:
        """
        Detect and identify faces in the given frame
        
        :param frame_bgr: OpenCV BGR format image
        :return: FaceResult with detection information or None
        """
        raise NotImplementedError
    
    