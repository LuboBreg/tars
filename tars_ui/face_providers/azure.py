import os
import cv2
import numpy as np
from typing import Optional

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

from .base import FaceProvider, FaceResult

class AzureFaceProvider(FaceProvider):
    def __init__(self):
        # Azure Face Client Setup
        self.face_key = os.getenv('AZURE_FACE_KEY')
        self.face_endpoint = os.getenv('AZURE_FACE_ENDPOINT')
        
        if not self.face_key or not self.face_endpoint:
            raise ValueError("Azure Face API credentials not configured")
        
        self.face_client = FaceClient(
            self.face_endpoint, 
            CognitiveServicesCredentials(self.face_key)
        )

    def identify(self, frame: np.ndarray) -> Optional[FaceResult]:
        """
        Comprehensive face analysis with age, emotions, and identity detection
        """
        # Convert OpenCV frame to bytes for Azure
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        try:
            # Detect faces with all attributes
            detected_faces = self.face_client.face.detect_with_stream(
                image=image_bytes,
                return_face_attributes=[
                    'age', 
                    'gender', 
                    'emotion', 
                    'qualityForRecognition'
                ],
                recognition_model='recognition_04',
                detection_model='detection_03'
            )
            
            if not detected_faces:
                return None
            
            # Take the most prominent face
            primary_face = detected_faces[0]
            attributes = primary_face.face_attributes
            
            # Emotion extraction
            emotions = {
                'happiness': attributes.emotion.happiness,
                'sadness': attributes.emotion.sadness,
                'anger': attributes.emotion.anger,
                'surprise': attributes.emotion.surprise,
                'neutral': attributes.emotion.neutral
            }
            
            return FaceResult(
                identity=None,  # If you have identity recognition, add it here
                confidence=None,
                age=int(attributes.age),
                gender=attributes.gender,
                emotions=emotions
            )
        
        except Exception as e:
            print(f"Face analysis error: {e}")
            return None
        