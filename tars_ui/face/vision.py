"""
Computer Vision module for T.A.R.S Executive AI System
Contains background removal, face segmentation, and image processing utilities
"""

import cv2
import numpy as np


class BackgroundRemover:
    """High-quality face segmentation with distinct iPhone-style portrait effects"""
    
    def __init__(self):
        # Face detection with multiple cascade classifiers for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Segmentation state
        self.frame_count = 0
        self.mask_history = []
        self.max_history = 5
        
        # Portrait mode styles
        self.portrait_style = "natural"
        
    def set_portrait_style(self, style):
        """Set iPhone-style portrait mode effect"""
        self.portrait_style = style
        
    def remove_background(self, frame):
        """High-quality face segmentation with distinct portrait effects"""
        try:
            h, w = frame.shape[:2]
            
            # Multi-scale face detection for better accuracy
            mask = self._high_quality_face_detection(frame)
            
            # Temporal smoothing with edge preservation
            self.mask_history.append(mask.copy())
            if len(self.mask_history) > self.max_history:
                self.mask_history.pop(0)
            
            if len(self.mask_history) >= 3:
                mask = self._advanced_temporal_smoothing(self.mask_history)
            
            # Create distinct portrait result based on selected style
            result = self._create_high_quality_portrait(frame, mask)
            
            self.frame_count += 1
            return result
            
        except Exception as e:
            print(f"Background removal error: {e}")
            return frame
    
    def _high_quality_face_detection(self, frame):
        """Advanced face detection with multiple methods"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detection with different parameters
        faces1 = self.face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(60, 60), maxSize=(400, 400))
        faces2 = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50), maxSize=(350, 350))
        faces3 = self.face_cascade.detectMultiScale(gray, 1.15, 3, minSize=(70, 70), maxSize=(300, 300))
        
        # Combine detections
        all_faces = []
        if len(faces1) > 0:
            all_faces.extend(faces1)
        if len(faces2) > 0:
            all_faces.extend(faces2)
        if len(faces3) > 0:
            all_faces.extend(faces3)
        
        if len(all_faces) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Select best face from all detections
        best_face = self._select_optimal_face(all_faces, w, h)
        
        # Create high-quality mask
        mask = self._create_advanced_face_mask(frame, best_face)
        
        return mask
    
    def _select_optimal_face(self, faces, frame_width, frame_height):
        """Select the most suitable face from multiple detections"""
        center_x, center_y = frame_width // 2, frame_height // 2
        best_face = faces[0]
        best_score = 0
        
        for face in faces:
            x, y, w, h = face
            
            # Calculate multiple quality metrics
            area = w * h
            aspect_ratio = w / h
            
            # Distance from center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            
            # Quality score: prefer larger, well-proportioned, centered faces
            size_score = min(area / (frame_width * frame_height), 0.25) * 4  # Normalize to 0-1
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.5  # Prefer square-ish faces
            center_score = 1.0 - (distance / (frame_width * 0.5))  # Distance from center
            
            total_score = size_score * 0.4 + aspect_score * 0.3 + center_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def _create_advanced_face_mask(self, frame, face_rect):
        """Create high-quality face mask with proper proportions"""
        h, w = frame.shape[:2]
        x, y, fw, fh = face_rect
        
        # Face center
        face_center_x = x + fw // 2
        face_center_y = y + fh // 2
        
        # Create more generous but still precise oval
        ellipse_width = int(fw * 0.65)   # More generous width
        ellipse_height = int(fh * 0.85)  # Include more of the head
        
        # Multi-layer mask for better quality
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Core face area (guaranteed inclusion)
        core_width = int(ellipse_width * 0.7)
        core_height = int(ellipse_height * 0.7)
        cv2.ellipse(mask, (face_center_x, face_center_y), 
                   (core_width, core_height), 0, 0, 360, 255, -1)
        
        # Extended area with skin color validation
        extended_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(extended_mask, (face_center_x, face_center_y), 
                   (ellipse_width, ellipse_height), 0, 0, 360, 255, -1)
        
        # Advanced skin detection
        skin_mask = self._advanced_skin_detection(frame, face_rect)
        
        # Combine core area with validated extended area
        validated_extended = cv2.bitwise_and(extended_mask, skin_mask)
        final_mask = cv2.bitwise_or(mask, validated_extended)
        
        # Edge refinement with bilateral filtering
        final_mask = cv2.bilateralFilter(final_mask, 9, 75, 75)
        
        # Gradient edges for natural falloff
        final_mask = cv2.GaussianBlur(final_mask, (7, 7), 2)
        
        return final_mask
    
    def _advanced_skin_detection(self, frame, face_rect):
        """Multi-method skin detection for higher accuracy"""
        h, w = frame.shape[:2]
        x, y, fw, fh = face_rect
        
        # Sample from multiple face regions for better color estimation
        sample_regions = [
            (x + fw//2, y + fh//2, fw//8),      # Center
            (x + fw//4, y + fh//3, fw//10),     # Left cheek
            (x + 3*fw//4, y + fh//3, fw//10),   # Right cheek
            (x + fw//2, y + 2*fh//3, fw//12),   # Chin area
        ]
        
        all_samples = []
        for sx, sy, sr in sample_regions:
            x1, y1 = max(0, sx - sr), max(0, sy - sr)
            x2, y2 = min(w, sx + sr), min(h, sy + sr)
            if x2 > x1 and y2 > y1:
                sample = frame[y1:y2, x1:x2]
                all_samples.extend(sample.reshape(-1, 3))
        
        if not all_samples:
            return np.zeros((h, w), dtype=np.uint8)
        
        all_samples = np.array(all_samples)
        mean_color = np.mean(all_samples, axis=0)
        std_color = np.std(all_samples, axis=0)
        
        # Color distance with standard deviation weighting
        frame_float = frame.astype(np.float32)
        color_diff = frame_float - mean_color
        weighted_distances = np.sqrt(np.sum((color_diff / (std_color + 1)) ** 2, axis=2))
        
        # Adaptive threshold based on color distribution
        threshold = np.percentile(weighted_distances, 25)  # More permissive
        skin_mask = (weighted_distances < threshold * 1.5).astype(np.uint8) * 255
        
        # Additional HSV-based skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([0, 15, 30])
        hsv_upper = np.array([25, 180, 255])
        hsv_skin = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # Combine both methods
        combined_skin = cv2.bitwise_or(skin_mask, hsv_skin)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_CLOSE, kernel)
        combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_OPEN, kernel)
        
        return combined_skin
    
    def _create_high_quality_portrait(self, frame, mask):
        """Create distinct portrait effects with high quality processing"""
        h, w = frame.shape[:2]
        
        if self.portrait_style == "black_white":
            # Professional B&W portrait processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better tonal range
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Professional B&W adjustment
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            
            # Subtle sharpening for crisp details
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
            enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
            
            # Convert to 3-channel
            stylized_face = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        elif self.portrait_style == "high_contrast":
            # Dramatic high contrast B&W
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aggressive contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
            enhanced = clahe.apply(gray)
            
            # High contrast curve
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=-40)
            
            # Threshold for dramatic effect
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced = cv2.addWeighted(enhanced, 0.7, thresh, 0.3, 0)
            
            # Strong sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.3)
            enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)
            
            stylized_face = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        else:  # "natural" - enhanced color portrait
            # Natural portrait with enhanced colors
            # Convert to LAB for better color manipulation
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance luminance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced_lab = cv2.merge([l, a, b])
            stylized_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Subtle warmth and brightness
            stylized_face = cv2.convertScaleAbs(stylized_face, alpha=1.1, beta=15)
            
            # Apply subtle grayish-white tint
            tint = np.array([245, 245, 240], dtype=np.uint8)
            stylized_face = cv2.addWeighted(stylized_face, 0.85, 
                                          np.full_like(stylized_face, tint), 0.15, 0)
        
        # Create result with proper alpha blending
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Smooth alpha blending
        alpha_normalized = mask.astype(np.float32) / 255.0
        alpha_3d = np.stack([alpha_normalized] * 3, axis=2)
        
        result[:, :, :3] = (stylized_face * alpha_3d).astype(np.uint8)
        result[:, :, 3] = mask
        
        return result
    
    def _advanced_temporal_smoothing(self, mask_history):
        """Advanced temporal smoothing with edge preservation"""
        if len(mask_history) < 3:
            return mask_history[-1]
        
        # Use multiple frames with decreasing weights
        weights = [0.5, 0.3, 0.2]  # Current, previous, older
        smoothed = np.zeros_like(mask_history[-1], dtype=np.float32)
        
        for i, weight in enumerate(weights):
            if i < len(mask_history):
                smoothed += mask_history[-(i+1)].astype(np.float32) * weight
        
        # Edge preservation
        current_edges = cv2.Canny(mask_history[-1], 50, 150)
        edge_mask = current_edges > 0
        
        result = smoothed.astype(np.uint8)
        result[edge_mask] = mask_history[-1][edge_mask]  # Preserve current edges
        
        return result
    
    def reset(self):
        """Reset all state"""
        self.frame_count = 0
        self.mask_history = []


class VisionProcessor:
    """High-level vision processing manager"""
    
    def __init__(self):
        self.background_remover = BackgroundRemover()
        
    def process_frame(self, frame, enable_portrait=False, portrait_style="natural"):
        """Process a frame with optional portrait mode"""
        if enable_portrait:
            self.background_remover.set_portrait_style(portrait_style)
            return self.background_remover.remove_background(frame)
        return frame
    
    def reset(self):
        """Reset all processors"""
        self.background_remover.reset()


# Utility functions for face analysis integration
def get_dominant_emotion(emotions):
    """Determine the dominant emotion from emotion dictionary"""
    return max(emotions, key=emotions.get) if emotions else 'neutral'


def get_emotion_color(emotion):
    """Get color based on detected emotion - returns RGB tuple"""
    emotion_colors = {
        'happiness': (134, 188, 37),   # Deloitte Green
        'neutral': (168, 170, 173),    # Silver
        'surprise': (255, 165, 0),     # Orange
        'sadness': (100, 149, 237),    # Cornflower Blue
        'anger': (220, 20, 60)         # Crimson
    }
    return emotion_colors.get(emotion, (168, 170, 173))  # Default to silver