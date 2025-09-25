"""
UI Components for T.A.R.S Executive AI System
Contains all custom UI widgets and components
"""

import numpy as np
import cv2
from PySide6.QtWidgets import (QLabel, QTextEdit, QLineEdit, QPushButton, 
                              QVBoxLayout, QHBoxLayout, QGraphicsDropShadowEffect)
from PySide6.QtGui import (QFont, QImage, QPixmap, QColor, QPalette, QPainter, 
                          QPen, QBrush, QLinearGradient, QPolygon, QRadialGradient)
from PySide6.QtCore import Qt, QRect, QPoint

# Import colors - you'll need to adjust this import based on your structure
# from .constants import DELOITTE_GREEN, DELOITTE_DARK_GREEN, etc.
# For now, keeping the colors inline - move these to your constants file

# Deloitte Brand Colors
DELOITTE_GREEN = QColor(134, 188, 37)      # #86BC25 - Deloitte signature green
DELOITTE_DARK_GREEN = QColor(100, 140, 28) # Darker green for depth
DELOITTE_BLACK = QColor(18, 20, 24)        # #121418 - Deep professional black
DELOITTE_CHARCOAL = QColor(32, 35, 42)     # #20232A - Charcoal for panels
DELOITTE_SILVER = QColor(168, 170, 173)    # #A8AAAD - Professional silver
DELOITTE_WHITE = QColor(247, 248, 249)     # #F7F8F9 - Clean white


def qimage_from_bgr(frame):
    """Convert BGR frame to QImage"""
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


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
        
        font = QFont("Arial", 11, QFont.Weight.Medium)
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
    """Apple-style round button with flat icon and text below - no white backgrounds"""
    def __init__(self, icon_text, label_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 120)  # Taller for icon + label
        
        # Create layout for icon and text
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)  # Increased spacing
        
        # Icon button (circular) - completely transparent by default
        self.icon_btn = QPushButton(icon_text)
        self.icon_btn.setFixedSize(80, 80)
        self.icon_btn.setCheckable(True)
        
        # Clean, minimal font - larger for better visibility
        font = QFont("Arial", 32, QFont.Weight.Light)  # Increased size
        self.icon_btn.setFont(font)
        
        # Completely transparent button with subtle hover states
        self.icon_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 40px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 180);
            }}
            QPushButton:hover {{
                background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 50);
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            }}
            QPushButton:pressed {{
                background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 80);
            }}
            QPushButton:checked {{
                background-color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255);
                color: white;
            }}
            QPushButton:checked:hover {{
                background-color: rgba({DELOITTE_GREEN.red() + 20}, {DELOITTE_GREEN.green() + 20}, {DELOITTE_GREEN.blue() + 20}, 255);
            }}
        """)
        
        # Label below with transparent background
        self.label = QLabel(label_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_font = QFont("Arial", 11, QFont.Weight.Medium)
        self.label.setFont(label_font)
        self.label.setStyleSheet(f"""
            QLabel {{
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 140);
                background: transparent;
                border: none;
            }}
        """)
        
        layout.addWidget(self.icon_btn, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
        
        # Ensure container has absolutely no background
        self.setStyleSheet("""
            AppleButton {
                background: transparent;
                border: none;
            }
        """)
        
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


class PremiumChatDisplay(QTextEdit):
    """Executive-grade chat interface"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        
        font = QFont("Arial", 12)
        font.setStyleHint(QFont.StyleHint.System)
        self.setFont(font)
        
        # Professional shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            QTextEdit {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 250), 
                    stop:0.3 rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 250),
                    stop:1 rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 250));
                border: 2px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 120);
                border-radius: 15px;
                color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                padding: 20px;
                selection-background-color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 100);
                line-height: 1.4;
            }}
            QScrollBar:vertical {{
                background: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 200);
                border: 1px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 80);
                border-radius: 8px;
                width: 16px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 150),
                    stop:1 rgba({DELOITTE_DARK_GREEN.red()}, {DELOITTE_DARK_GREEN.green()}, {DELOITTE_DARK_GREEN.blue()}, 150));
                border-radius: 7px;
                min-height: 25px;
                margin: 1px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 200);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)


class ExecutiveInput(QLineEdit):
    """Premium executive input field"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        
        font = QFont("Arial", 13)
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
        
        font = QFont("Arial", 14, QFont.Weight.Medium)
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