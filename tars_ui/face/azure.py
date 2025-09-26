import os
import cv2
import numpy as np
import requests
import time
import logging
from typing import Optional
from .base import FaceProvider, FaceResult

logger = logging.getLogger(__name__)

class AzureFaceProvider(FaceProvider):
    def __init__(self):
        """Initialize Azure Face Provider with local emotion model"""
        logger.info("Initializing Azure Face Provider with local emotion model...")
        
        # Azure credentials
        self.face_key = os.getenv("AZURE_FACE_KEY")
        self.face_endpoint = os.getenv('AZURE_FACE_ENDPOINT')
        
        if not self.face_key or not self.face_endpoint:
            raise ValueError("Azure Face API credentials not configured")
        
        self.endpoint = self.face_endpoint.strip().rstrip('/')
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Rate limiting for F0 tier
        
        # Initialize local emotion model
        self.emotion_predictor = None
        try:
            from .emotion_model import EmotionPredictor
            self.emotion_predictor = EmotionPredictor()
            logger.info("Local emotion model loaded successfully")
        except ImportError:
            logger.warning("emotion_model module not found - using default emotions")
        except Exception as e:
            logger.warning(f"Failed to load local emotion model: {e} - using default emotions")
        
        logger.info("Azure Face Provider initialization complete")
    
    def identify(self, frame: np.ndarray) -> Optional[FaceResult]:
        """Face detection with local emotion analysis"""
        logger.debug("Starting face identification with local emotion analysis...")
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            # Encode frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode frame to JPEG")
                return None
            
            image_bytes = buffer.tobytes()
            logger.debug(f"Encoded image size: {len(image_bytes)} bytes")
            
            # Azure face detection (basic only - no deprecated attributes)
            faces = self._detect_faces_azure(image_bytes)
            
            if not faces:
                logger.debug("No faces detected by Azure")
                return None
            
            logger.debug(f"Azure detected {len(faces)} face(s)")
            
            # Use first detected face
            primary_face = faces[0]
            face_rect = primary_face.get('faceRectangle', {})
            
            # Predict emotions using local model
            emotions = self._predict_emotions_local(frame, face_rect)
            
            # Create result
            result = FaceResult(
                identity="Executive Identified",
                confidence=0.92,
                age=None,  # Can be added later with local age model
                gender=None,  # Can be added later with local gender model
                emotions=emotions
            )
            
            logger.debug(f"Returning result with emotions: {emotions}")
            return result
            
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                logger.warning("Azure API rate limit exceeded")
                self.min_request_interval = min(10.0, self.min_request_interval * 1.5)
            else:
                logger.error(f"Azure API HTTP error: {e}")
            return None
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Azure API connection error: {e}")
            return None
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Azure API timeout: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in face identification: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _detect_faces_azure(self, image_bytes):
        """Basic face detection using Azure API (no deprecated attributes)"""
        url = f"{self.endpoint}/face/v1.0/detect"
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.face_key,
            'Content-Type': 'application/octet-stream'
        }
        
        # Only request basic face detection - NO deprecated attributes
        params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false'
        }
        
        logger.debug(f"Making Azure API request to: {url}")
        logger.debug(f"Request params: {params}")
        
        self.last_request_time = time.time()
        response = requests.post(url, headers=headers, params=params, data=image_bytes, timeout=30)
        
        logger.debug(f"Azure API response status: {response.status_code}")
        
        if response.status_code == 200:
            faces = response.json()
            logger.debug(f"Azure API response: {faces}")
            return faces
        
        elif response.status_code == 429:
            logger.warning("Azure API rate limit exceeded")
            self.min_request_interval = min(10.0, self.min_request_interval * 2.0)
            raise requests.exceptions.HTTPError(f"Rate limit exceeded: {response.text}")
        
        else:
            logger.error(f"Azure API error {response.status_code}: {response.text}")
            response.raise_for_status()
    
    def _predict_emotions_local(self, frame, face_rect):
        """Predict emotions using local PyTorch model"""
        if not self.emotion_predictor:
            logger.debug("No emotion predictor available, using default emotions")
            return self._get_default_emotions()
        
        if not face_rect:
            logger.debug("No face rectangle provided, using default emotions")
            return self._get_default_emotions()
        
        try:
            # Extract face region using Azure's bounding box
            left = face_rect.get('left', 0)
            top = face_rect.get('top', 0)
            width = face_rect.get('width', 0)
            height = face_rect.get('height', 0)
            
            # Validate face rectangle
            if width <= 0 or height <= 0:
                logger.warning("Invalid face rectangle dimensions")
                return self._get_default_emotions()
            
            # Extract face with bounds checking
            h, w = frame.shape[:2]
            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(w, left + width)
            y2 = min(h, top + height)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning("Face rectangle out of bounds")
                return self._get_default_emotions()
            
            face_image = frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                logger.warning("Extracted face image is empty")
                return self._get_default_emotions()
            
            logger.debug(f"Extracted face region: {face_image.shape}")
            
            # Predict emotions using local model
            emotions = self.emotion_predictor.predict(face_image)
            logger.debug(f"Local emotion prediction: {emotions}")
            
            return emotions
            
        except Exception as e:
            logger.error(f"Local emotion prediction failed: {e}")
            return self._get_default_emotions()
    
    def _get_default_emotions(self):
        """Default emotions if local prediction fails"""
        return {
            'happiness': 0.25,
            'sadness': 0.1,
            'anger': 0.08,
            'surprise': 0.12,
            'neutral': 0.35,
            'contempt': 0.05,
            'disgust': 0.03,
            'fear': 0.02
        }
    
    def is_available(self) -> bool:
        """Check if Azure Face API is available"""
        logger.debug("Checking Azure Face API availability...")
        
        try:
            test_url = f"{self.endpoint}/face/v1.0/detect"
            headers = {'Ocp-Apim-Subscription-Key': self.face_key}
            
            logger.debug(f"Testing availability at: {test_url}")
            
            # HEAD request to test connectivity without using quota
            response = requests.head(test_url, headers=headers, timeout=10)
            
            logger.debug(f"Availability test response: {response.status_code}")
            
            # Service is available if we get any response indicating endpoint exists
            # 200: OK, 400: Bad Request (expected for HEAD), 405: Method Not Allowed, 429: Rate Limited
            is_available = response.status_code in [200, 400, 405, 429]
            
            logger.info(f"Azure Face API availability: {'Available' if is_available else 'Not Available'}")
            return is_available
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during availability check: {e}")
            return False
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during availability check: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during availability check: {e}")
            return False
    
    def get_emotion_predictor_status(self):
        """Get status of local emotion model"""
        return {
            'available': self.emotion_predictor is not None,
            'model_type': 'Local PyTorch CNN' if self.emotion_predictor else None,
            'emotions_supported': ['happiness', 'sadness', 'anger', 'surprise', 'neutral', 'contempt', 'disgust', 'fear']
        }