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
from tars_ui.voice.audio_voice import AudioManager, create_audio_system_callback


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
        
        font = QFont("Arial", 24, QFont.Weight.Light)
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
        label_font = QFont("Arial", 10, QFont.Weight.Medium)
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
        self.setMinimumSize(400, 250)  
        self.messages = []  
        self.max_messages = 3  
        self.typing_speed = 80  
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
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
        
        layout.addStretch()
        self.setLayout(layout)
        
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
        
        #