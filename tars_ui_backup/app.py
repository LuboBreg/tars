import os, sys, time, threading
import numpy as np, cv2
from typing import Optional
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QTextEdit, QLineEdit, 
                              QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QGraphicsDropShadowEffect,
                              QScrollArea, QSizePolicy)
from PySide6.QtGui import QFont, QImage, QPixmap, QColor, QPalette, QPainter, QPen, QBrush, QLinearGradient, QPolygon, QRadialGradient
from PySide6.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve, QPoint, QSize

from tars_ui.face_providers.base import FaceProvider, FaceResult
from tars_ui.face_providers.azure import AzureFaceProvider
from tars_ui.face_providers.incoresoft import IncoresoftFaceProvider
from tars_ui.chat.azure_openai_client import ChatClient
from tars_ui.voice.azure_speech import Speech


# Deloitte Brand Colors
DELOITTE_GREEN = QColor(134, 188, 37)      # #86BC25 - Deloitte signature green
DELOITTE_DARK_GREEN = QColor(100, 140, 28) # Darker green for depth
DELOITTE_BLACK = QColor(18, 20, 24)        # #121418 - Deep professional black
DELOITTE_CHARCOAL = QColor(32, 35, 42)     # #20232A - Charcoal for panels
DELOITTE_SILVER = QColor(168, 170, 173)    # #A8AAAD - Professional silver
DELOITTE_WHITE = QColor(247, 248, 249)     # #F7F8F9 - Clean white

def qimage_from_bgr(frame):
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

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

class PremiumHexagonalDisplay(QLabel):
    """Ultra-professional hexagonal display with Deloitte styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self._frame = None
        self._glow_phase = 0
        
        # Premium shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(DELOITTE_GREEN)
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        
    def set_frame(self, frame):
        self._frame = frame
        self.update()
        
    def advance_glow(self):
        self._glow_phase = (self._glow_phase + 0.02) % (2 * 3.14159)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        try:
            rect = self.rect()
            center_x, center_y = rect.width() // 2, rect.height() // 2
            size = min(rect.width(), rect.height()) // 2 - 30
            
            # Create hexagon points
            points = []
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                x = center_x + size * np.cos(angle)
                y = center_y + size * np.sin(angle)
                points.append(QPoint(int(x), int(y)))
            
            # Multi-layer glow effect with animation
            glow_intensity = (np.sin(self._glow_phase) * 0.3 + 0.7)
            
            for layer in range(15, 0, -1):
                alpha = int(25 * glow_intensity / layer)
                color = QColor(DELOITTE_GREEN)
                color.setAlpha(alpha)
                
                pen = QPen(color)
                pen.setWidth(layer)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                
                # Draw hexagon with rounded corners
                polygon = QPolygon(points)
                painter.drawPolygon(polygon)
            
            # Inner professional border
            pen = QPen(DELOITTE_GREEN)
            pen.setWidth(3)
            painter.setPen(pen)
            inner_polygon = QPolygon([QPoint(p.x() * 0.95 + center_x * 0.05, p.y() * 0.95 + center_y * 0.05) for p in points])
            painter.drawPolygon(inner_polygon)
            
            # Premium background gradient
            gradient = QRadialGradient(center_x, center_y, size)
            gradient.setColorAt(0, QColor(DELOITTE_BLACK.red() + 20, DELOITTE_BLACK.green() + 20, DELOITTE_BLACK.blue() + 20))
            gradient.setColorAt(1, DELOITTE_BLACK)
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(inner_polygon)
            
            # Draw frame with professional clipping
            if self._frame is not None:
                painter.setClipRegion(inner_polygon)
                
                h, w = self._frame.shape[:2]
                channels = self._frame.shape[2] if len(self._frame.shape) > 2 else 1
                aspect = w / h
                margin = 20
                
                if aspect > 1:
                    new_w = (size * 2) - margin
                    new_h = int(new_w / aspect)
                else:
                    new_h = (size * 2) - margin
                    new_w = int(new_h * aspect)
                
                frame_resized = cv2.resize(self._frame, (new_w, new_h))
                
                # Handle both 3-channel (BGR) and 4-channel (BGRA) frames
                if channels == 4:
                    # Convert BGRA to RGBA for Qt
                    rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGRA2RGBA)
                    qimg = QImage(rgba.data, new_w, new_h, new_w * 4, QImage.Format.Format_RGBA8888)
                else:
                    # Standard BGR to RGB conversion
                    qimg = qimage_from_bgr(frame_resized)
                
                x_offset = center_x - new_w // 2
                y_offset = center_y - new_h // 2
                painter.drawImage(x_offset, y_offset, qimg)
                
                # Professional overlay grid
                painter.setClipping(False)
                painter.setPen(QPen(QColor(DELOITTE_GREEN.red(), DELOITTE_GREEN.green(), DELOITTE_GREEN.blue(), 30), 1))
                grid_size = 25
                for i in range(0, rect.width(), grid_size):
                    painter.drawLine(i, 0, i, rect.height())
                for i in range(0, rect.height(), grid_size):
                    painter.drawLine(0, i, rect.width(), i)
        
        finally:
            painter.end()

class EliteLightColumn(QLabel):
    """Premium animated light column with Deloitte styling"""
    def __init__(self, column_id=0, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 450)
        self._phase = column_id * 0.5  # Offset each column
        self._intensity = 1.0
        self._active = True
        self._column_id = column_id
        
        # Premium shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(DELOITTE_GREEN)
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
    def set_intensity(self, intensity):
        self._intensity = max(0.0, min(1.0, intensity))
        self.update()
        
    def set_active(self, active):
        self._active = active
        self.update()
        
    def advance_animation(self):
        self._phase += 0.08 + (self._column_id * 0.01)
        self.update()
        
    def paintEvent(self, event):
        if not self._active:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        try:
            rect = self.rect()
            segments = 18
            segment_height = rect.height() // segments
            margin = 15
            
            for i in range(segments):
                # Advanced wave algorithm
                wave1 = np.sin(self._phase + i * 0.4) * 0.4
                wave2 = np.sin(self._phase * 1.3 + i * 0.2) * 0.3
                wave3 = np.sin(self._phase * 0.7 + i * 0.6) * 0.3
                combined_wave = (wave1 + wave2 + wave3) * 0.5 + 0.5
                
                brightness = int(combined_wave * 255 * self._intensity)
                
                # Professional color palette
                green_intensity = int(DELOITTE_GREEN.green() * brightness / 255)
                color = QColor(
                    int(DELOITTE_GREEN.red() * brightness / 255 * 0.8),
                    green_intensity,
                    int(DELOITTE_GREEN.blue() * brightness / 255 * 0.6),
                    brightness
                )
                
                y_pos = i * segment_height
                segment_rect = QRect(margin, y_pos + 3, rect.width() - 2*margin, segment_height - 6)
                
                # Multi-layer glow effect
                for glow in range(8, 0, -1):
                    glow_color = QColor(color)
                    glow_color.setAlpha(brightness // (glow + 2))
                    expanded_rect = segment_rect.adjusted(-glow//2, -glow//2, glow//2, glow//2)
                    painter.fillRect(expanded_rect, glow_color)
                
                # Premium gradient fill
                gradient = QLinearGradient(segment_rect.left(), 0, segment_rect.right(), 0)
                gradient.setColorAt(0, QColor(color.red()//3, color.green()//3, color.blue()//3, color.alpha()//2))
                gradient.setColorAt(0.5, color)
                gradient.setColorAt(1, QColor(color.red()//3, color.green()//3, color.blue()//3, color.alpha()//2))
                painter.fillRect(segment_rect, QBrush(gradient))
                
                # Professional highlight
                if brightness > 200:
                    highlight_rect = QRect(segment_rect.x() + 5, segment_rect.y() + 1, segment_rect.width() - 10, 2)
                    painter.fillRect(highlight_rect, QColor(255, 255, 255, brightness//3))
        
        finally:
            painter.end()

class ExecutiveButton(QPushButton):
    """Apple-style round flat button"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(80, 80)  # Perfect circle
        
        font = QFont("SF Pro Display", 11, QFont.Weight.Medium)
        self.setFont(font)
        
        # No shadow for flat design
        self.setGraphicsEffect(None)
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 255);
                border: none;
                border-radius: 40px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: rgba({DELOITTE_CHARCOAL.red() + 20}, {DELOITTE_CHARCOAL.green() + 20}, {DELOITTE_CHARCOAL.blue() + 20}, 255);
            }}
            QPushButton:pressed {{
                background-color: rgba({DELOITTE_CHARCOAL.red() - 10}, {DELOITTE_CHARCOAL.green() - 10}, {DELOITTE_CHARCOAL.blue() - 10}, 255);
            }}
            QPushButton:checked {{
                background-color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
                color: white;
            }}
            QPushButton:checked:hover {{
                background-color: rgba({DELOITTE_GREEN.red() + 15}, {DELOITTE_GREEN.green() + 15}, {DELOITTE_GREEN.blue() + 15}, 255);
            }}
        """)

class AppleButton(QPushButton):
    """Apple-style round button with icon and text below"""
    def __init__(self, icon_text, label_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 120)  # Taller for icon + label
        
        # Create layout for icon and text
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Icon button (circular)
        self.icon_btn = QPushButton(icon_text)
        self.icon_btn.setFixedSize(80, 80)
        self.icon_btn.setCheckable(True)
        
        font = QFont("Open Sans Light", 24, QFont.Weight.Light)
        self.icon_btn.setFont(font)
        
        self.icon_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 40px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            }}
            QPushButton:hover {{
                background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 100);
            }}
            QPushButton:pressed {{
                background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 150);
            }}
            QPushButton:checked {{
                background-color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
                color: white;
            }}
            QPushButton:checked:hover {{
                background-color: rgba({DELOITTE_GREEN.red() + 15}, {DELOITTE_GREEN.green() + 15}, {DELOITTE_GREEN.blue() + 15}, 255);
            }}
        """)
        
        # Label below
        self.label = QLabel(label_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_font = QFont("Open Sans Light", 10, QFont.Weight.Medium)
        self.label.setFont(label_font)
        self.label.setStyleSheet(f"""
            QLabel {{
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 200);
            }}
        """)
        
        layout.addWidget(self.icon_btn, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
        
        # Connect signals
        self.icon_btn.clicked.connect(self.clicked.emit)
        self.icon_btn.toggled.connect(self.toggled.emit)
    
    def setCheckable(self, checkable):
        self.icon_btn.setCheckable(checkable)
    
    def setChecked(self, checked):
        self.icon_btn.setChecked(checked)
    
    def isChecked(self):
        return self.icon_btn.isChecked()
    
    def setText(self, text):
        self.label.setText(text)


class FuturisticChatDisplay(QWidget):
    """Futuristic chat interface with typing animation and limited messages"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 250)  # Reduced height since fewer messages
        self.messages = []  # Store last 3 messages
        self.max_messages = 3  # Changed from 5 to 3
        self.typing_speed = 80  # milliseconds per character
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove all margins
        layout.setSpacing(20)  # Increased spacing for bigger text
        
        # Message containers (newest first)
        self.message_labels = []
        for i in range(self.max_messages):
            label = QLabel()
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignTop)
            label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    border: none;
                    padding: 0px;
                    color: rgba(134, 188, 37, 255);
                    font-family: 'Courier New', 'Monaco', monospace;
                    font-size: 16px;
                    font-weight: bold;
                    line-height: 1.5;
                }
            """)
            self.message_labels.append(label)
            layout.addWidget(label)
        
        layout.addStretch()  # Push messages to top
        self.setLayout(layout)
        
        # Completely transparent background, no frame
        self.setStyleSheet("""
            FuturisticChatDisplay {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        # Typing animation timer
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self._continue_typing)
        self.current_typing = None
        
    def add_message(self, sender: str, message: str):
        """Add new message and trigger typing animation"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Create formatted message with green colors
        if sender == "T.A.R.S":
            color = "rgba(0, 255, 0, 255)"  # Bright green
            prefix = f"[{timestamp}] T.A.R.S: "
        elif sender == "EXECUTIVE":
            color = "rgba(50, 255, 50, 220)"  # Slightly different green
            prefix = f"[{timestamp}] YOU: "
        else:
            color = "rgba(134, 188, 37, 200)"  # Deloitte green
            prefix = f"[{timestamp}] {sender}: "
        
        full_message = prefix + message
        
        # Add to messages list (newest first)
        self.messages.insert(0, (full_message, color))
        
        # Keep only last 3 messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[:self.max_messages]
        
        # Update display
        self._update_display()
        
    def _update_display(self):
        """Update all message labels"""
        # Clear all labels first
        for label in self.message_labels:
            label.setText("")
        
        # Set static messages (all except the newest)
        for i in range(1, len(self.messages)):
            if i < len(self.message_labels):
                message_text, color = self.messages[i]
                self.message_labels[i].setStyleSheet(f"""
                    QLabel {{
                        background: transparent;
                        border: none;
                        padding: 0px;
                        color: {color};
                        font-family: 'Courier New', 'Monaco', monospace;
                        font-size: 16px;
                        font-weight: bold;
                        line-height: 1.5;
                    }}
                """)
                self.message_labels[i].setText(message_text)
        
        # Start typing animation for newest message
        if self.messages:
            self._start_typing_animation(0)
    
    def _start_typing_animation(self, label_index):
        """Start typing animation for specified message"""
        if label_index >= len(self.messages) or label_index >= len(self.message_labels):
            return
            
        message_text, color = self.messages[label_index]
        
        # Setup typing animation
        self.current_typing = {
            'label_index': label_index,
            'full_text': message_text,
            'current_text': '',
            'char_index': 0,
            'color': color
        }
        
        # Start animation
        self.typing_timer.start(self.typing_speed)
    
    def _continue_typing(self):
        """Continue typing animation"""
        if not self.current_typing:
            self.typing_timer.stop()
            return
        
        typing = self.current_typing
        
        # Add next character
        if typing['char_index'] < len(typing['full_text']):
            typing['current_text'] += typing['full_text'][typing['char_index']]
            typing['char_index'] += 1
            
            # Update label
            label = self.message_labels[typing['label_index']]
            label.setStyleSheet(f"""
                QLabel {{
                    background: transparent;
                    border: none;
                    padding: 0px;
                    color: {typing['color']};
                    font-family: 'Courier New', 'Monaco', monospace;
                    font-size: 16px;
                    font-weight: bold;
                    line-height: 1.5;
                }}
            """)
            
            # Add blinking cursor
            display_text = typing['current_text'] + "_"
            label.setText(display_text)
        else:
            # Typing complete, remove cursor
            label = self.message_labels[typing['label_index']]
            label.setText(typing['full_text'])
            
            # Stop animation
            self.typing_timer.stop()
            self.current_typing = None
    
    def append(self, html_message):
        """Compatibility method for existing code"""
        # Extract plain text from HTML (simple approach)
        import re
        plain_text = re.sub('<[^<]+?>', '', html_message)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        
        # Determine sender from text
        if "T.A.R.S" in plain_text:
            sender = "T.A.R.S"
            message = plain_text.split(":", 1)[-1].strip()
        else:
            sender = "SYSTEM"
            message = plain_text
            
        self.add_message(sender, message)

class ExecutiveInput(QLineEdit):
    """Premium executive input field"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        
        font = QFont("Open Sans Light", 13)
        self.setFont(font)
        
        # Premium shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            QLineEdit {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 230), 
                    stop:1 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 230));
                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 100);
                border-radius: 12px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                padding: 12px 20px;
                selection-background-color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 80);
                font-weight: 500;
            }}
            QLineEdit:focus {{
                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 200);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba({DELOITTE_CHARCOAL.red() + 10}, {DELOITTE_CHARCOAL.green() + 10}, {DELOITTE_CHARCOAL.blue() + 10}, 240), 
                    stop:1 rgba({DELOITTE_BLACK.red() + 15}, {DELOITTE_BLACK.green() + 15}, {DELOITTE_BLACK.blue() + 15}, 240));
            }}
        """)

class ExecutiveStatusPanel(QLabel):
    """Premium executive status display"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        
        font = QFont("Open Sans Light", 14, QFont.Weight.Medium)
        self.setFont(font)
        
        # Premium shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240), 
                    stop:0.5 rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 240),
                    stop:1 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240));
                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 150);
                border-radius: 12px;
                color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
                padding: 18px 24px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
        """)

class PremiumTarsUI(QWidget):
    def __init__(self, provider: Optional[FaceProvider] = None, chat: Optional[ChatClient] = None, 
                 camera_index: int = 0, speech: Optional[Speech] = None):
        super().__init__()
        self.setWindowTitle("T.A.R.S - Deloitte Executive AI System")
        self.resize(1800, 1000)
        
        # Professional executive theme
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 255), 
                    stop:0.3 rgba({DELOITTE_CHARCOAL.red()//2}, {DELOITTE_CHARCOAL.green()//2}, {DELOITTE_CHARCOAL.blue()//2}, 255),
                    stop:0.7 rgba({DELOITTE_CHARCOAL.red()//2}, {DELOITTE_CHARCOAL.green()//2}, {DELOITTE_CHARCOAL.blue()//2}, 255),
                    stop:1 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 255));
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            }}
            QLabel {{
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            }}
        """)
        
        self.setup_premium_ui()
        self.initialize_systems(provider, chat, speech, camera_index)
        self.setup_timers_and_threads()

    def setup_premium_ui(self):
        # Executive header with Deloitte branding
        header = QLabel("T.A.R.S")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont("Open Sans Light", 32, QFont.Weight.Bold)
        header.setFont(header_font)
        
        subtitle = QLabel("DELOITTE EXECUTIVE AI SYSTEM")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Open Sans Light", 14, QFont.Weight.Medium)
        subtitle.setFont(subtitle_font)
        
        header.setStyleSheet(f"""
            color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
            padding: 15px;
            letter-spacing: 3px;
        """)
        subtitle.setStyleSheet(f"""
            color: rgba({DELOITTE_SILVER.red()}, {DELOITTE_SILVER.green()}, {DELOITTE_SILVER.blue()}, 255);
            padding-bottom: 30px;
            letter-spacing: 2px;
        """)

        # Left panel - Video Display (2/3 width)
        self.video_display = PremiumHexagonalDisplay()
        
        self.identity_panel = ExecutiveStatusPanel()
        self.identity_panel.setText("VISUAL INTELLIGENCE: STANDBY")
        
        video_layout = QVBoxLayout()
        video_title = QLabel("VISUAL INTELLIGENCE")
        video_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_title.setFont(QFont("Open Sans Light", 16, QFont.Weight.DemiBold))
        video_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_display, 1)
        video_layout.addWidget(self.identity_panel)
        
        # Right panel - Controls and Chat (1/3 width)
        right_layout = QVBoxLayout()
        
        # Control buttons section
        controls_title = QLabel("SYSTEM CONTROLS")
        controls_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_title.setFont(QFont("Open Sans Light", 16, QFont.Weight.DemiBold))
        controls_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        # Apple style round buttons
        self.face_id_btn = AppleButton("👁", "Face ID")
        self.face_id_btn.setCheckable(True)
        self.face_id_btn.clicked.connect(self.toggle_face_identification)
        
        self.voice_command_btn = AppleButton("🎙", "Voice")
        self.voice_command_btn.setCheckable(True)
        self.voice_command_btn.setChecked(True)  # Wake words always active
        self.voice_command_btn.clicked.connect(self.toggle_wake_word_system)
        
        self.background_remove_btn = AppleButton("🎭", "Portrait")
        self.background_remove_btn.setCheckable(True)
        self.background_remove_btn.clicked.connect(self.toggle_background_removal)
        
        # Portrait style buttons (initially hidden)
        self.natural_btn = AppleButton("🌅", "Natural")
        self.natural_btn.setCheckable(True)
        self.natural_btn.setChecked(True)
        self.natural_btn.clicked.connect(lambda: self.set_portrait_style("natural"))
        
        self.bw_btn = AppleButton("⚫", "B&W")
        self.bw_btn.setCheckable(True)
        self.bw_btn.clicked.connect(lambda: self.set_portrait_style("black_white"))
        
        self.contrast_btn = AppleButton("⚡", "Contrast")
        self.contrast_btn.setCheckable(True)
        self.contrast_btn.clicked.connect(lambda: self.set_portrait_style("high_contrast"))
        
        # Initially hide portrait style buttons
        self.natural_btn.setVisible(False)
        self.bw_btn.setVisible(False)
        self.contrast_btn.setVisible(False)
        
        # Button layout - horizontal arrangement
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        buttons_layout.addWidget(self.face_id_btn)
        buttons_layout.addWidget(self.voice_command_btn)
        buttons_layout.addWidget(self.background_remove_btn)
        
        # Portrait style layout (hidden initially)
        self.portrait_style_layout = QHBoxLayout()
        self.portrait_style_layout.setSpacing(15)
        self.portrait_style_layout.addWidget(self.natural_btn)
        self.portrait_style_layout.addWidget(self.bw_btn)
        self.portrait_style_layout.addWidget(self.contrast_btn)
        
        # Chat section
        chat_title = QLabel("EXECUTIVE COMMUNICATION")
        chat_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chat_title.setFont(QFont("Open Sans Light", 16, QFont.Weight.DemiBold))
        chat_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        self.executive_chat = FuturisticChatDisplay()
        self.executive_chat.append(f"<div style='color: rgb({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}); font-weight: bold; font-size: 14px;'>T.A.R.S EXECUTIVE SYSTEM:</div><div style='margin-top: 8px; color: rgb({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}); line-height: 1.5;'>All systems operational. Deloitte Executive AI ready for engagement.</div>")
        
        self.command_input = ExecutiveInput()
        self.command_input.setPlaceholderText("Enter executive command or query...")
        self.command_input.returnPressed.connect(self.process_command)
        
        self.transmit_btn = AppleButton("⚡", "Execute")
        self.transmit_btn.clicked.connect(self.process_command)
        
        # Input layout
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.command_input, 1)
        input_layout.addWidget(self.transmit_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Assemble right panel
        right_layout.addWidget(controls_title)
        right_layout.addLayout(buttons_layout)
        right_layout.addLayout(self.portrait_style_layout)
        right_layout.addSpacing(20)
        right_layout.addWidget(chat_title)
        right_layout.addWidget(self.executive_chat, 1)
        right_layout.addLayout(input_layout)
        
        # Main layout - 2/3 left, 1/3 right
        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(30)
        main_content_layout.addLayout(video_layout, 2)  # 2/3 width
        main_content_layout.addLayout(right_layout, 1)  # 1/3 width
        
        # Final assembly
        main_layout = QVBoxLayout()
        
        # Header section
        header_layout = QVBoxLayout()
        header_layout.addWidget(header)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)
        
        # Content section
        main_layout.addLayout(main_content_layout, 1)
        main_layout.setContentsMargins(25, 25, 25, 25)
        self.setLayout(main_layout)

    def initialize_systems(self, provider, chat, speech, camera_index):
        # Initialize AI backends
        self.chat = chat or ChatClient()
        self.speech = speech or Speech()
        
        provider_name = os.getenv("TARS_FACE_PROVIDER", "azure").lower()
        if provider is None:
            if provider_name == "azure":
                provider = AzureFaceProvider()
            elif provider_name == "incoresoft":
                provider = IncoresoftFaceProvider()
            else:
                raise SystemExit(f"Unknown TARS_FACE_PROVIDER: {provider_name}")
        self.provider = provider

        # Premium camera system
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera system unavailable. Please verify permissions and device availability.")
        
        # Face identification state
        self.face_identification_active = False
        
        # Background removal state
        self.background_removal_active = False
        self.background_remover = BackgroundRemover()

    def setup_timers_and_threads(self):
        # High-performance video processing
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_visual_intelligence)
        self.video_timer.start(30)  # 33 FPS for premium experience
        
        # Hexagon glow animation
        self.glow_timer = QTimer()
        self.glow_timer.timeout.connect(self.animate_hexagon_glow)
        self.glow_timer.start(50)  # 20 FPS for ambient glow
        
        # Executive-grade face recognition variables
        self._last_identity = None
        self._identity_lock = threading.Lock()
        self._voice_active = False
        self._face_recognition_thread = None
        
        # Start continuous wake word listening
        self.start_wake_word_listening()

    def start_wake_word_listening(self):
        """Start continuous wake word listening in background"""
        self.wake_word_listening = True
        threading.Thread(target=self._wake_word_listener, daemon=True).start()
        self.append_executive_message("SYSTEM", "Wake word system activated. Available commands:")
        self.append_executive_message("SYSTEM", "• 'Activate Face' - Turn on face identification")
        self.append_executive_message("SYSTEM", "• 'Portrait Mode' - Enable portrait background removal")
        self.append_executive_message("SYSTEM", "• 'Natural Mode' / 'Black White' / 'High Contrast' - Switch portrait styles")
        self.append_executive_message("SYSTEM", "• 'Hey TARS' - Start voice conversation")
    
    def _wake_word_listener(self):
        """Continuous listening for wake words"""
        while self.wake_word_listening:
            try:
                # Listen for wake words with shorter timeout
                audio_input = self.speech.recognize_once(timeout=2).lower().strip()
                
                if audio_input:
                    # Check for wake word matches
                    for wake_word, action in self.wake_words.items():
                        if wake_word in audio_input:
                            self.append_executive_message("WAKE WORD", f"Detected: '{wake_word}'")
                            action()
                            break
                    
            except Exception:
                # Silence exceptions for continuous listening
                time.sleep(0.1)
                continue
                
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    # Wake word action methods
    def activate_face_id(self):
        """Activate face identification via wake word"""
        if not self.face_identification_active:
            self.face_id_btn.setChecked(True)
            self.toggle_face_identification()
    
    def deactivate_face_id(self):
        """Deactivate face identification via wake word"""
        if self.face_identification_active:
            self.face_id_btn.setChecked(False)
            self.toggle_face_identification()
    
    def activate_portrait_mode(self):
        """Activate portrait mode via wake word"""
        if not self.background_removal_active:
            self.background_remove_btn.setChecked(True)
            self.toggle_background_removal()
    
    def deactivate_portrait_mode(self):
        """Deactivate portrait mode via wake word"""
        if self.background_removal_active:
            self.background_remove_btn.setChecked(False)
            self.toggle_background_removal()
    
    def start_voice_chat(self):
        """Start voice chat conversation"""
        self.append_executive_message("SYSTEM", "Voice chat activated - listening for your command...")
        threading.Thread(target=self._process_voice_chat, daemon=True).start()
    
    def _process_voice_chat(self):
        """Process voice chat with longer listening time"""
        try:
            # Listen for longer command/question
            voice_input = self.speech.recognize_once(timeout=10).strip()
            
            if voice_input:
                self.command_input.setText(voice_input)
                self.process_command()
            else:
                self.append_executive_message("SYSTEM", "No voice input detected")
                
        except Exception as e:
            self.append_executive_message("SYSTEM", f"Voice chat error: {e}")

    def toggle_face_identification(self):
        """Toggle face identification service on/off"""
        self.face_identification_active = self.face_id_btn.isChecked()
        
        if self.face_identification_active:
            self.append_executive_message("SYSTEM", "Face identification service activated")
            self.identity_panel.setText("FACE ID: SCANNING...")
            # Start face recognition thread
            if self._face_recognition_thread is None or not self._face_recognition_thread.is_alive():
                self._face_recognition_thread = threading.Thread(target=self._executive_face_recognition, daemon=True)
                self._face_recognition_thread.start()
        else:
            self.append_executive_message("SYSTEM", "Face identification service deactivated")
            self.identity_panel.setText("FACE ID: STANDBY")
            # Clear identity data
            with self._identity_lock:
                self._last_identity = None

    def toggle_background_removal(self):
        """Toggle background removal on/off"""
        self.background_removal_active = self.background_remove_btn.isChecked()
        
        if self.background_removal_active:
            self.append_executive_message("SYSTEM", "Portrait mode activated - iPhone-style face isolation")
            self.background_remover.reset()
            # Show portrait style options
            self.natural_btn.setVisible(True)
            self.bw_btn.setVisible(True)
            self.contrast_btn.setVisible(True)
        else:
            self.append_executive_message("SYSTEM", "Portrait mode deactivated")
            # Hide portrait style options
            self.natural_btn.setVisible(False)
            self.bw_btn.setVisible(False)
            self.contrast_btn.setVisible(False)

    def set_portrait_style(self, style):
        """Set iPhone-style portrait mode effect"""
        # Update button states
        self.natural_btn.setChecked(style == "natural")
        self.bw_btn.setChecked(style == "black_white")
        self.contrast_btn.setChecked(style == "high_contrast")
        
        # Update background remover style
        self.background_remover.set_portrait_style(style)
        
        # Notify user
        style_names = {
            "natural": "Natural grayish-white portrait mode",
            "black_white": "Black & White portrait mode (iPhone style)",
            "high_contrast": "High contrast black & white mode"
        }
        self.append_executive_message("SYSTEM", f"Portrait style: {style_names.get(style, style)}")

    def toggle_wake_word_system(self):
        """Toggle wake word listening system on/off"""
        if self.voice_command_btn.isChecked():
            if not self.wake_word_listening:
                self.start_wake_word_listening()
        else:
            self.wake_word_listening = False
            self.append_executive_message("SYSTEM", "Wake word system deactivated")

    def animate_hexagon_glow(self):
        self.video_display.advance_glow()

    def update_visual_intelligence(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize for better performance (smaller processing size)
            processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # Store original sized frame for face recognition (better quality)
            self._current_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply background removal if active
            if self.background_removal_active:
                # Process background removal on smaller frame for performance
                bg_removed_frame = self.background_remover.remove_background(processed_frame)
                
                # Resize back to display size
                if bg_removed_frame.shape[2] == 4:
                    display_frame = cv2.resize(bg_removed_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = cv2.resize(bg_removed_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = self._current_frame
            
            # Update display
            self.video_display.set_frame(display_frame)
            
            # Face recognition status (only update every few frames for performance)
            if self.face_identification_active and hasattr(self, '_frame_counter'):
                self._frame_counter = getattr(self, '_frame_counter', 0) + 1
                
                # Only update face recognition display every 5 frames
                if self._frame_counter % 5 == 0:
                    with self._identity_lock:
                        identity_result = self._last_identity
                    
                    if identity_result and identity_result.identity:
                        confidence = f" ({identity_result.confidence:.1%})" if identity_result.confidence is not None else ""
                        status_text = f"EXECUTIVE IDENTIFIED: {identity_result.identity.upper()}{confidence}"
                        self.identity_panel.setStyleSheet(f"""
                            QLabel {{
                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba({DELOITTE_GREEN.red()//3}, {DELOITTE_GREEN.green()//3}, {DELOITTE_GREEN.blue()//3}, 240), 
                                    stop:0.5 rgba({DELOITTE_GREEN.red()//2}, {DELOITTE_GREEN.green()//2}, {DELOITTE_GREEN.blue()//2}, 240),
                                    stop:1 rgba({DELOITTE_GREEN.red()//3}, {DELOITTE_GREEN.green()//3}, {DELOITTE_GREEN.blue()//3}, 240));
                                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 200);
                                border-radius: 12px;
                                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                                padding: 18px 24px;
                                font-weight: 600;
                                letter-spacing: 0.3px;
                            }}
                        """)
                        self.identity_panel.setText(status_text)
                    elif identity_result:
                        status_text = "FACE ID: SCANNING..."
                        self.identity_panel.setStyleSheet(f"""
                            QLabel {{
                                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240), 
                                    stop:0.5 rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 240),
                                    stop:1 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240));
                                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 150);
                                border-radius: 12px;
                                color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
                                padding: 18px 24px;
                                font-weight: 600;
                                letter-spacing: 0.3px;
                            }}
                        """)
                        self.identity_panel.setText(status_text)
            elif not self.face_identification_active:
                # Initialize frame counter
                self._frame_counter = 0

    def _executive_face_recognition(self):
        """Face recognition thread - only runs when face identification is active"""
        while self.face_identification_active:
            frame = getattr(self, "_current_frame", None)
            if frame is not None:
                try:
                    recognition_result = self.provider.identify(frame)
                except Exception:
                    recognition_result = None
                with self._identity_lock:
                    self._last_identity = recognition_result
            time.sleep(0.4)  # Executive-grade recognition frequency

    def _voice_command_pressed(self):
        if self._voice_active:
            return
        self._voice_active = True
        self.append_executive_message("SYSTEM", "Voice command system activated...")
        threading.Thread(target=self._process_voice_command, daemon=True).start()

    def _voice_command_released(self):
        pass  # Voice recognition handles its own termination

    def _process_voice_command(self):
        try:
            voice_input = self.speech.recognize_once().strip()
        except Exception as e:
            voice_input = ""
        
        self._voice_active = False
        
        if voice_input:
            self.command_input.setText(voice_input)
            self.process_command()
        else:
            self.append_executive_message("SYSTEM", "No voice input detected")

    def process_command(self):
        command_text = self.command_input.text().strip()
        if not command_text:
            return
        
        self.append_executive_message("EXECUTIVE", command_text)
        self.command_input.clear()
        threading.Thread(target=self._executive_ai_processing, args=(command_text,), daemon=True).start()

    def append_executive_message(self, sender: str, message: str):
        """Add message to futuristic chat display"""
        self.executive_chat.add_message(sender, message)


    def _executive_ai_processing(self, command_text: str):
        try:
            executive_messages = [
                {
                    "role": "system", 
                    "content": """You are T.A.R.S, an elite AI system designed for Deloitte executives. You are:
                    - Highly intelligent and strategic
                    - Professional and concise 
                    - Focused on business insights and executive decision support
                    - Capable of wit when appropriate, but always professional
                    - Knowledgeable about consulting, technology, and business strategy
                    
                    Respond as the sophisticated executive AI assistant you are, providing valuable insights and analysis."""
                },
                {"role": "user", "content": command_text}
            ]
            
            ai_response = self.chat.chat(executive_messages)
        except Exception as e:
            ai_response = f"Executive AI system encountered an error: {e}"
        
        self.append_executive_message("T.A.R.S", ai_response)
        
        # Executive text-to-speech
        try:
            self.speech.speak(ai_response)
        except Exception:
            pass
        
    
    # face recognition methods 
    
    def _update_emotion_based_ui(self, face_analysis):
        """
        Update UI based on emotion detection - only when face identification is active
        """
        if not face_analysis or not self.face_identification_active:
            return

        # Create detailed status text
        status_text = (
            f"EXECUTIVE PROFILE:\n"
            f"Age: {face_analysis.age or 'N/A'} years\n"
            f"Gender: {face_analysis.gender or 'N/A'}\n"
            f"Dominant Emotion: {self._get_dominant_emotion(face_analysis.emotions).upper()} "
            f"({max(face_analysis.emotions.values()):.1%})"
        )
        
        # Emotion-based color coding
        dominant_emotion = self._get_dominant_emotion(face_analysis.emotions)
        emotion_color = self._get_emotion_color(dominant_emotion)
        
        # Update identity panel with emotion-based styling
        self.identity_panel.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba({emotion_color.red()//3}, {emotion_color.green()//3}, {emotion_color.blue()//3}, 240), 
                    stop:0.5 rgba({emotion_color.red()//2}, {emotion_color.green()//2}, {emotion_color.blue()//2}, 240),
                    stop:1 rgba({emotion_color.red()//3}, {emotion_color.green()//3}, {emotion_color.blue()//3}, 240));
                border: 2px solid rgba({emotion_color.red()}, {emotion_color.green()}, {emotion_color.blue()}, 200);
                border-radius: 12px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                padding: 18px 24px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
        """)
        
        self.identity_panel.setText(status_text)
        
        # Trigger emotion-based interactions
        self._handle_emotion_based_interaction(face_analysis)

    def _get_dominant_emotion(self, emotions):
        """
        Determine the dominant emotion
        """
        return max(emotions, key=emotions.get) if emotions else 'neutral'

    def _get_emotion_color(self, emotion):
        """
        Get color based on detected emotion
        """
        emotion_colors = {
            'happiness': DELOITTE_GREEN,
            'neutral': DELOITTE_SILVER,
            'surprise': QColor(255, 165, 0),  # Orange
            'sadness': QColor(100, 149, 237),  # Cornflower Blue
            'anger': QColor(220, 20, 60)  # Crimson
        }
        return emotion_colors.get(emotion, DELOITTE_SILVER)

    def _handle_emotion_based_interaction(self, face_analysis):
        """
        Trigger conversational responses based on detected emotions - only when face identification is active
        """
        if not face_analysis or not face_analysis.emotions or not self.face_identification_active:
            return

        dominant_emotion = self._get_dominant_emotion(face_analysis.emotions)
        emotion_intensity = max(face_analysis.emotions.values())

        # Emotion-specific interactions
        interactions = {
            'happiness': [
                "I detect you're in high spirits! Would you like to discuss some exciting business opportunities?",
                "Your positive energy is contagious! How can I support your goals today?"
            ],
            'anger': [
                "I sense some tension. Would you like to take a moment to discuss what's troubling you?",
                "Strong emotions can be a catalyst for change. How can I help you navigate this?"
            ],
            'sadness': [
                "Your mood seems reflective. Is there anything I can help you with?",
                "Sometimes a different perspective can help. Would you like to explore some solutions?"
            ],
            'surprise': [
                "Interesting reaction! What caught you off guard?",
                "Surprises often lead to breakthrough insights. Care to elaborate?"
            ],
            'neutral': [
                "Ready for our next strategic discussion.",
                "How can I assist you in moving your objectives forward?"
            ]
        }

        # Select interaction based on emotion and intensity
        if emotion_intensity > 0.5:
            possible_responses = interactions.get(dominant_emotion, interactions['neutral'])
            response = possible_responses[hash(str(face_analysis)) % len(possible_responses)]
            self.append_executive_message("T.A.R.S", response)


# Create alias for backward compatibility
TarsUI = PremiumTarsUI

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("T.A.R.S Executive System")
    
    # Set executive application style
    app.setStyleSheet(f"""
        QToolTip {{
            background-color: rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240);
            color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            border: 1px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 150);
            padding: 8px;
            border      -radius: 6px;
            font-family: 'Open Sans Light';
        }}
    """)
    
    ui = PremiumTarsUI(provider=None, chat=None, speech=None)
    ui.show()
    
    sys.exit(app.exec())