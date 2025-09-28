# Add this to the top of your main app file (before any imports)
import multiprocessing
import os
import sys

# Prevent multiprocessing conflicts
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Limit TensorFlow resources to prevent conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import time, threading
import numpy as np, cv2
from typing import Optional
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QTextEdit, QLineEdit, 
                              QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QGraphicsDropShadowEffect,
                              QScrollArea, QSizePolicy)
from PySide6.QtGui import QFont, QImage, QPixmap, QColor, QPalette, QPainter, QPen, QBrush, QLinearGradient, QPolygon, QRadialGradient
from PySide6.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve, QPoint, QSize

from .face.base import FaceProvider, FaceResult
from .face.azure import AzureFaceProvider
from .chat.azure_openai_client import ChatClient
from .voice.azure_speech import Speech
from PySide6.QtGui import QFontDatabase

# Import UI Components
from .ui_components import (
    PremiumHexagonalDisplay, EliteLightColumn, ExecutiveButton,
    AppleButton, PremiumChatDisplay, ExecutiveInput, ExecutiveStatusPanel
)

# Import Vision Module
from .face.vision import BackgroundRemover

# Import Face Recognition Module
from .face.face_recognition import FaceRecognitionManager, FaceRecognitionUIHelper

# Import Audio/Voice Module
from .voice.audio_voice import AudioManager, create_audio_system_callback
from .constants import DELOITTE_GREEN, DELOITTE_BLACK, DELOITTE_DARK_GREEN, DELOITTE_CHARCOAL, DELOITTE_CHARCOAL, DELOITTE_SILVER, DELOITTE_WHITE


class PremiumTarsUI(QWidget):
    def __init__(self, provider: Optional[FaceProvider] = None, chat: Optional[ChatClient] = None, 
                 camera_index: int = 0, speech: Optional[Speech] = None):
        super().__init__()
        self.setWindowTitle("T.A.R.S - Deloitte Executive AI System")
        self.resize(1800, 1000)
        
        # Add cleanup tracking
        self._cleanup_done = False
        self._audio_initialized = False
        self._threads_active = []
        self._last_voice_trigger = 0
        
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
        header_font = QFont("Arial", 32, QFont.Weight.Bold)
        header.setFont(header_font)
        
        subtitle = QLabel("DELOITTE EXECUTIVE AI SYSTEM")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Arial", 14, QFont.Weight.Medium)
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
        video_title.setFont(QFont("Arial", 16, QFont.Weight.DemiBold))
        video_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_display, 1)
        video_layout.addWidget(self.identity_panel)
        
        # Right panel - Controls and Chat (1/3 width)
        right_layout = QVBoxLayout()
        
        # Control buttons section
        controls_title = QLabel("SYSTEM CONTROLS")
        controls_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_title.setFont(QFont("Arial", 16, QFont.Weight.DemiBold))
        controls_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        # Apple style round buttons
        self.face_id_btn = AppleButton("⊙", "Face ID")
        self.face_id_btn.setCheckable(True)
        self.face_id_btn.setChecked(True)
        self.face_id_btn.clicked.connect(self.toggle_face_identification)

        self.voice_command_btn = AppleButton("●", "Voice")
        self.voice_command_btn.setCheckable(True)
        self.voice_command_btn.setChecked(True)
        self.voice_command_btn.clicked.connect(self.toggle_wake_word_system)
        
        self.background_remove_btn = AppleButton("◐", "Portrait")
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
        
        # Toggle for auto voice
        self.auto_voice_btn = AppleButton("🗣", "Auto Voice")
        self.auto_voice_btn.setCheckable(True)
        self.auto_voice_btn.setChecked(True)

        # Chat section
        chat_title = QLabel("EXECUTIVE COMMUNICATION")
        chat_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chat_title.setFont(QFont("Arial", 16, QFont.Weight.DemiBold))
        chat_title.setStyleSheet(f"color: rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 255); padding: 15px;")
        
        self.executive_chat = PremiumChatDisplay()
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
        right_layout.addWidget(self.auto_voice_btn)
        right_layout.addSpacing(20)
        right_layout.addWidget(chat_title)
        right_layout.addWidget(self.executive_chat, 1)
        right_layout.addLayout(input_layout)
        
        # Main layout - 2/3 left, 1/3 right
        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(30)
        main_content_layout.addLayout(video_layout, 2)
        main_content_layout.addLayout(right_layout, 1)
        
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
        
        # Provider selection
        provider_name = os.getenv("TARS_FACE_PROVIDER", "opencv").lower()
        if provider is None:
            if provider_name == "opencv":
                from .face.deepface_provider import DeepFaceProvider
                provider = DeepFaceProvider() 
            elif provider_name == "local":
                from .face.local import LocalFaceProvider
                provider = LocalFaceProvider()
            elif provider_name == "azure":
                provider = AzureFaceProvider()
            else:
                raise SystemExit(f"Unknown TARS_FACE_PROVIDER: {provider_name}")
        self.provider = provider

        # Premium camera system
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera system unavailable. Please verify permissions and device availability.")
        
        # Initialize background removal system
        self.background_removal_active = False
        self.background_remover = BackgroundRemover()
        
        # Initialize audio system with proper error handling
        try:
            audio_callback = create_audio_system_callback(self)
            self.audio_manager = AudioManager(self.speech, self.chat, audio_callback)
            self._audio_initialized = True
            print("Audio system initialized successfully")
        except Exception as e:
            print(f"Audio system initialization failed: {e}")
            self.audio_manager = None
            self._audio_initialized = False
        
        # Initialize face recognition with safer threading
        try:
            face_callback = self._create_face_recognition_callback()
            self.face_manager = FaceRecognitionManager(self.provider, face_callback)
            print("Face recognition initialized successfully")
        except Exception as e:
            print(f"Face recognition initialization failed: {e}")
            self.face_manager = None

    def _create_face_recognition_callback(self):
        """Create callback for face recognition system"""
        def face_callback(action, *args):
            if action == 'update_identity_panel':
                text, style_info = args
                if 'color' in style_info:
                    stylesheet = FaceRecognitionUIHelper.create_identity_panel_style(
                        style_info['color'], style_info.get('success', False)
                    )
                    self.identity_panel.setStyleSheet(stylesheet)
                self.identity_panel.setText(text)
                
                # Trigger voice conversation when face is identified
                if "EXECUTIVE IDENTIFIED" in text.upper() or "IDENTIFIED" in text.upper():
                    self._trigger_voice_greeting()
                    
            elif action == 'add_message':
                sender, message = args
                self.append_executive_message(sender, message)
        return face_callback
    
    def _trigger_voice_greeting(self):
        """Safer voice greeting trigger"""
        if not hasattr(self, 'auto_voice_btn') or not self.auto_voice_btn.isChecked():
            return
        
        if not self._audio_initialized or not self.audio_manager:
            return

        # Avoid multiple rapid triggers
        current_time = time.time()
        if current_time - self._last_voice_trigger < 30:
            return
        
        self._last_voice_trigger = current_time

        try:
            self.append_executive_message("SYSTEM", "Executive detected - initiating voice conversation...")
            self.audio_manager.trigger_auto_voice_greeting()
        except Exception as e:
            print(f"Voice greeting trigger failed: {e}")
            self.append_executive_message("SYSTEM", f"Voice greeting error: {e}")

    def setup_timers_and_threads(self):
        # High-performance video processing
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_visual_intelligence)
        self.video_timer.start(30)
        
        # Hexagon glow animation
        self.glow_timer = QTimer()
        self.glow_timer.timeout.connect(self.animate_hexagon_glow)
        self.glow_timer.start(50)
        
        # Frame counter for face recognition optimization
        self._frame_counter = 0
        
        # Start audio system safely
        if self._audio_initialized and self.audio_manager:
            try:
                self.audio_manager.start_wake_words()
                print("Wake word system started")
            except Exception as e:
                print(f"Failed to start wake word system: {e}")
        
        # Start face recognition safely
        if hasattr(self, 'face_manager') and self.face_manager:
            try:
                self.face_manager.start_recognition()
                self.identity_panel.setText("FACE ID: SCANNING...")
                self.append_executive_message("SYSTEM", "Face identification service activated")
            except Exception as e:
                print(f"Failed to start face recognition: {e}")

    def toggle_face_identification(self):
        """Safer face ID toggle"""
        if not hasattr(self, 'face_manager') or not self.face_manager:
            self.append_executive_message("SYSTEM", "Face recognition system not available")
            return
        
        try:
            if self.face_id_btn.isChecked():
                self.face_manager.start_recognition()
                self.append_executive_message("SYSTEM", "Face identification service activated")
                self.identity_panel.setText("FACE ID: SCANNING...")
            else:
                self.face_manager.stop_recognition()
                self.append_executive_message("SYSTEM", "Face identification service deactivated")
                self.identity_panel.setText("FACE ID: STANDBY")
        except Exception as e:
            print(f"Face ID toggle error: {e}")
            self.append_executive_message("SYSTEM", f"Face ID error: {e}")

    def toggle_background_removal(self):
        """Toggle background removal on/off"""
        self.background_removal_active = self.background_remove_btn.isChecked()
        
        if self.background_removal_active:
            self.append_executive_message("SYSTEM", "Portrait mode activated - iPhone-style face isolation")
            self.background_remover.reset()
            self.natural_btn.setVisible(True)
            self.bw_btn.setVisible(True)
            self.contrast_btn.setVisible(True)
        else:
            self.append_executive_message("SYSTEM", "Portrait mode deactivated")
            self.natural_btn.setVisible(False)
            self.bw_btn.setVisible(False)
            self.contrast_btn.setVisible(False)

    def set_portrait_style(self, style):
        """Set iPhone-style portrait mode effect"""
        self.natural_btn.setChecked(style == "natural")
        self.bw_btn.setChecked(style == "black_white")
        self.contrast_btn.setChecked(style == "high_contrast")
        
        self.background_remover.set_portrait_style(style)
        
        style_names = {
            "natural": "Natural grayish-white portrait mode",
            "black_white": "Black & White portrait mode (iPhone style)",
            "high_contrast": "High contrast black & white mode"
        }
        self.append_executive_message("SYSTEM", f"Portrait style: {style_names.get(style, style)}")

    def toggle_wake_word_system(self):
        """Safer wake word toggle"""
        if not self._audio_initialized or not self.audio_manager:
            self.append_executive_message("SYSTEM", "Audio system not available")
            return
        
        try:
            if self.voice_command_btn.isChecked():
                if not self.audio_manager.is_wake_word_active():
                    self.audio_manager.start_wake_words()
                    self.append_executive_message("SYSTEM", "Wake word system activated")
            else:
                self.audio_manager.stop_wake_words()
                self.append_executive_message("SYSTEM", "Wake word system deactivated")
        except Exception as e:
            print(f"Wake word toggle error: {e}")
            self.append_executive_message("SYSTEM", f"Wake word error: {e}")

    def animate_hexagon_glow(self):
        self.video_display.advance_glow()

    def update_visual_intelligence(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize for better performance
            processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # Store original sized frame for face recognition
            self._current_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LANCZOS4)
            
            # Update face recognition manager with current frame
            if hasattr(self, 'face_manager') and self.face_manager:
                try:
                    self.face_manager.update_frame(self._current_frame)
                except Exception as e:
                    if self._frame_counter % 100 == 0:
                        print(f"Face recognition update error: {e}")
            
            # Apply background removal if active
            if self.background_removal_active:
                try:
                    bg_removed_frame = self.background_remover.remove_background(processed_frame)
                    display_frame = cv2.resize(bg_removed_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    if self._frame_counter % 100 == 0:
                        print(f"Background removal error: {e}")
                    display_frame = self._current_frame
            else:
                display_frame = self._current_frame
            
            # Update display
            try:
                self.video_display.set_frame(display_frame)
            except Exception as e:
                if self._frame_counter % 100 == 0:
                    print(f"Video display error: {e}")
           
            self._frame_counter += 1

    def process_command(self):
        """Safer command processing"""
        command_text = self.command_input.text().strip()
        if not command_text:
            return
        
        self.append_executive_message("EXECUTIVE", command_text)
        self.command_input.clear()
        
        # Use audio manager if available, otherwise fallback
        if self._audio_initialized and self.audio_manager:
            try:
                self.audio_manager.process_voice_command(command_text)
                return
            except Exception as e:
                print(f"Audio manager command processing failed: {e}")
        
        # Fallback to direct processing
        self._safe_thread_start(self._executive_ai_processing, (command_text,))

    def _safe_thread_start(self, target, args=(), daemon=True):
        """Start threads safely with tracking"""
        try:
            thread = threading.Thread(target=target, args=args, daemon=daemon)
            thread.start()
            if not daemon:
                self._threads_active.append(thread)
            return thread
        except Exception as e:
            print(f"Failed to start thread: {e}")
            return None

    def _executive_ai_processing(self, command_text: str):
        """Process AI command (kept for direct command processing)"""
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
        
        try:
            self.speech.speak(ai_response)
        except Exception:
            pass

    def append_executive_message(self, sender: str, message: str):
        timestamp = time.strftime("%H:%M:%S")
        
        if sender == "T.A.R.S":
            color = f"rgb({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()})"
        elif sender == "EXECUTIVE":
            color = f"rgb({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()})"
        else:
            color = f"rgb({DELOITTE_SILVER.red()}, {DELOITTE_SILVER.green()}, {DELOITTE_SILVER.blue()})"
        
        formatted_message = f"""
        <div style='margin-bottom: 15px; padding: 12px; border-left: 3px solid rgb({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}); background-color: rgba({DELOITTE_CHARCOAL.red()}, {DELOITTE_CHARCOAL.green()}, {DELOITTE_CHARCOAL.blue()}, 50);'>
            <div style='color: rgb({DELOITTE_SILVER.red()}, {DELOITTE_SILVER.green()}, {DELOITTE_SILVER.blue()}); font-size: 11px; margin-bottom: 5px;'>[{timestamp}]</div>
            <div style='color: {color}; font-weight: bold; font-size: 13px; margin-bottom: 5px;'>{sender}:</div>
            <div style='color: rgb({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}); line-height: 1.5;'>{message}</div>
        </div>
        """
        self.executive_chat.append(formatted_message)

    def closeEvent(self, event):
        """Proper cleanup on application close"""
        if self._cleanup_done:
            event.accept()
            return
        
        print("Starting application cleanup...")
        self._cleanup_done = True
        
        try:
            if hasattr(self, 'video_timer'):
                self.video_timer.stop()
            if hasattr(self, 'glow_timer'):
                self.glow_timer.stop()
            
            if self._audio_initialized and hasattr(self, 'audio_manager') and self.audio_manager:
                try:
                    self.audio_manager.end_current_conversation()
                    self.audio_manager.stop_wake_words()
                    print("Audio system stopped")
                except Exception as e:
                    print(f"Audio cleanup error: {e}")
            
            if hasattr(self, 'face_manager') and self.face_manager:
                try:
                    self.face_manager.stop_recognition()
                    print("Face recognition stopped")
                except Exception as e:
                    print(f"Face recognition cleanup error: {e}")
            
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                    print("Camera released")
                except Exception as e:
                    print(f"Camera cleanup error: {e}")
            
            for thread in self._threads_active:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(0.5)
        event.accept()


# Create alias for backward compatibility
TarsUI = PremiumTarsUI

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("T.A.R.S Executive System")
    
    app.setStyleSheet(f"""
        QToolTip {{
            background-color: rgba({DELOITTE_BLACK.red()}, {DELOITTE_BLACK.green()}, {DELOITTE_BLACK.blue()}, 240);
            color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
            border: 1px solid rgba({DELOITTE_GREEN.red()}, {DELOITTE_GREEN.green()}, {DELOITTE_GREEN.blue()}, 150);
            padding: 8px;
            border-radius: 6px;
            font-family: 'Arial';
        }}
    """)
    
    try:
        ui = PremiumTarsUI(provider=None, chat=None, speech=None)
        ui.show()
        
        import atexit
        atexit.register(lambda: print("Application exiting..."))
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()