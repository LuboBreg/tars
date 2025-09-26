import cv2
import numpy as np
import logging
import torch  # Make sure PyTorch is installed (pip install torch)
from typing import Optional
from .base import FaceProvider, FaceResult

logger = logging.getLogger(__name__)

class LocalFaceProvider(FaceProvider):
    """Local face detection using OpenCV + PyTorch emotion model"""
    
    def __init__(self):
        logger.info("Initializing Local Face Provider...")
        
        # Initialize OpenCV face detection cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load frontal face cascade classifier")
        if self.profile_cascade.empty():
            logger.warning("Failed to load profile face cascade classifier - will use frontal only")
            self.profile_cascade = None
        
        # Initialize local emotion model
        self.emotion_predictor = None
        try:
            from .emotions_model import EmotionPredictor
            self.emotion_predictor = EmotionPredictor()
            logger.info("Local emotion model loaded successfully")
        except ImportError:
            logger.warning("emotion_model module not found - using default emotions")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e} - using default emotions")
        
        logger.info("Local Face Provider initialization complete")
   
    def identify(self, frame: np.ndarray) -> Optional[FaceResult]:
        """Detect faces using OpenCV and analyze emotions with local model"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale face detection for better accuracy
            faces = self._detect_faces_multiscale(gray)
            
            if len(faces) == 0:
                return None
            
            # Use the best/largest face
            best_face = self._select_best_face(faces, frame.shape)
            x, y, w, h = best_face
            
            # Extract face region from the original color frame
            face_image = frame[y:y+h, x:x+w]
            
            # Predict emotions using local model
            emotions = self._predict_emotions(face_image)
            
            # Create result
            result = FaceResult(
                identity="Executive Identified (Local)",
                confidence=0.88,
                age=None,
                gender=None,
                emotions=emotions
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Local face detection error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _detect_faces_multiscale(self, gray_frame):
        """Multi-scale face detection for better accuracy"""
        all_faces = []
        
        faces1 = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.05, 
            minNeighbors=6, 
            minSize=(60, 60), 
            maxSize=(400, 400)
        )
        
        faces2 = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50),
            maxSize=(350, 350)
        )
        
        if len(faces1) > 0:
            all_faces.extend(faces1)
        if len(faces2) > 0:
            all_faces.extend(faces2)
        
        if self.profile_cascade is not None:
            profiles = self.profile_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60)
            )
            if len(profiles) > 0:
                all_faces.extend(profiles)
        
        if len(all_faces) > 1:
            all_faces = self._remove_duplicate_faces(all_faces)
        
        return all_faces
    
    def _remove_duplicate_faces(self, faces):
        """Remove overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        for face in faces:
            is_duplicate = False
            for unique_face in unique_faces:
                if self._faces_overlap(face, unique_face, threshold=0.5):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _faces_overlap(self, face1, face2, threshold=0.5):
        """Check if two face rectangles overlap significantly"""
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        ix = max(x1, x2)
        iy = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix or iy2 <= iy:
            return False
        
        intersection_area = (ix2 - ix) * (iy2 - iy)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        overlap_ratio = intersection_area / union_area
        return overlap_ratio > threshold
    
    def _select_best_face(self, faces, frame_shape):
        """Select the best face from multiple detections"""
        if len(faces) == 1:
            return faces[0]
        
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2
        
        best_face = faces[0]
        best_score = 0
        
        for face in faces:
            x, y, face_w, face_h = face
            
            area = face_w * face_h
            aspect_ratio = face_w / face_h
            
            face_center_x = x + face_w // 2
            face_center_y = y + face_h // 2
            distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            
            size_score = min(area / (w * h), 0.25) * 4
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.5
            center_score = 1.0 - (distance / (w * 0.5))
            
            total_score = size_score * 0.4 + aspect_score * 0.3 + center_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def _predict_emotions(self, face_image):
        """Predict emotions using local model after preprocessing the face image"""
        if not self.emotion_predictor:
            logger.debug("No emotion predictor available, using default emotions")
            return self._get_default_emotions()
        
        try:
            logger.debug("--- Starting Emotion Prediction Deep Dive ---")
            
            if face_image.size == 0:
                logger.warning("Empty face image provided for emotion prediction")
                return self._get_default_emotions()
            
            logger.debug(f"Step 1: Raw face crop received. Shape: {face_image.shape}, Dtype: {face_image.dtype}")

            # --- PREPROCESSING ---
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            logger.debug(f"Step 2: Converted to grayscale. Shape: {gray_face.shape}")

            resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
            logger.debug(f"Step 3: Resized image. Shape: {resized_face.shape}")
            
            # --- PYTORCH TENSOR CONVERSION ---
            face_tensor = torch.from_numpy(resized_face).float() / 255.0
            logger.debug(f"Step 4: Converted to tensor. Shape: {face_tensor.shape}, Dtype: {face_tensor.dtype}")
            
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
            logger.debug(f"Step 5: Unsqueezed tensor for model. Shape: {face_tensor.shape}")
            logger.debug(f"Final Tensor Stats -> Min: {face_tensor.min():.4f}, Max: {face_tensor.max():.4f}, Mean: {face_tensor.mean():.4f}")

            # --- PREDICTION ---
            logger.debug("Step 6: Sending final tensor to emotion_predictor.predict()")
            emotions = self.emotion_predictor.predict(face_tensor)
            
            logger.info(f"SUCCESS: Raw prediction received from model: {emotions}")

            dominant_emotion = max(emotions, key=emotions.get)
            logger.info(f"DOMINANT EMOTION: {dominant_emotion} ({emotions[dominant_emotion]:.2%})")
            logger.debug("--- Emotion Prediction Finished ---")
            
            return emotions
            
        except Exception as e:
            logger.error(f"CRITICAL FAILURE in emotion prediction pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_emotions()
    
    def _get_default_emotions(self):
        """Default emotions if prediction fails"""
        return {
            'happiness': 0.25,
            'sadness': 0.12,
            'anger': 0.08,
            'surprise': 0.15,
            'neutral': 0.30,
            'contempt': 0.05,
            'disgust': 0.03,
            'fear': 0.02
        }
    
    def is_available(self) -> bool:
        """Local provider is always available if properly initialized"""
        return not self.face_cascade.empty()
    
    def get_status(self):
        """Get status information"""
        return {
            'provider': 'Local OpenCV + PyTorch',
            'face_detection': 'OpenCV Haar Cascades',
            'emotion_model': 'Local PyTorch CNN' if self.emotion_predictor else 'Default values',
            'profile_detection': 'Available' if self.profile_cascade else 'Unavailable',
            'always_available': True,
            'no_api_limits': True
        }