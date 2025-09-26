import os, sys, time, threading
import numpy as np, cv2
from typing import Optional
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QTextEdit, QLineEdit, 
                              QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QGraphicsDropShadowEffect,
                              QScrollArea, QSizePolicy)
from PySide6.QtGui import (QFont, QImage, QPixmap, QColor, QPalette, QPainter, QPen, 
                          QBrush, QLinearGradient, QPolygon, QRadialGradient, QTextCursor)
from PySide6.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve, QPoint, QSize

from .face.base import FaceProvider, FaceResult
from .face.azure import AzureFaceProvider
# from .face.incoresoft import IncoresoftFaceProvider
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


# Deloitte Brand Colors - These should be moved to a constants.py file
#DELOITTE_GREEN = QColor(134, 188, 37)      # #86BC25 - Deloitte signature green
#DELOITTE_DARK_GREEN = QColor(100, 140, 28) # Darker green for depth
#DELOITTE_BLACK = QColor(18, 20, 24)        # #121418 - Deep professional black
#DELOITTE_CHARCOAL = QColor(32, 35, 42)     # #20232A - Charcoal for panels
#DELOITTE_SILVER = QColor(168, 170, 173)    # #A8AAAD - Professional silver
#ELOITTE_WHITE = QColor(247, 248, 249)     # #F7F8F9 - Clean white


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
        self.face_id_btn = AppleButton("⊙", "Face ID")  # Circle with center dot
        self.face_id_btn.setCheckable(True)
        self.face_id_btn.setChecked(True)
        self.face_id_btn.clicked.connect(self.toggle_face_identification)

        self.voice_command_btn = AppleButton("●", "Voice")  # Solid circle
        self.voice_command_btn.setCheckable(True)
        self.voice_command_btn.setChecked(True)  # Wake words always active
        self.voice_command_btn.clicked.connect(self.toggle_wake_word_system)
        
        self.background_remove_btn = AppleButton("◐", "Portrait")  # Half circle
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
        self.auto_voice_btn.setChecked(True)  # Default enabled

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
        
        # In initialize_systems method, replace the provider selection with:
        provider_name = os.getenv("TARS_FACE_PROVIDER", "opencv").lower()  # Default to deepface
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
        
        # Initialize audio system with callback
        audio_callback = create_audio_system_callback(self)
        self.audio_manager = AudioManager(self.speech, self.chat, audio_callback)
        
        # Initialize face recognition with callback
        face_callback = self._create_face_recognition_callback()
        self.face_manager = FaceRecognitionManager(self.provider, face_callback)

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
        """Trigger voice conversation when face is detected"""
        # Check if auto voice is enabled
        if hasattr(self, 'auto_voice_btn') and not self.auto_voice_btn.isChecked():
            return

        # Avoid multiple rapid triggers
        current_time = time.time()
        if hasattr(self, '_last_voice_trigger'):
            if current_time - self._last_voice_trigger < 30:  # Wait 30 seconds between triggers
                return
        
        self._last_voice_trigger = current_time
        
        # Start voice conversation
        self.append_executive_message("SYSTEM", "Executive detected - initiating voice conversation...")
        
        # Use your existing audio manager to start voice chat
        if hasattr(self, 'audio_manager'):
            threading.Thread(target=self._auto_voice_conversation, daemon=True).start()
        else:
            # Fallback to direct speech processing
            threading.Thread(target=self._process_auto_voice_chat, daemon=True).start()

    def _auto_voice_conversation(self):
        """Automatically start voice conversation when face is detected"""
        try:
            # Give a greeting first
            greeting_messages = [
                "Hello! I detected your presence. How can I assist you today?",
                "Good to see you! What can I help you with?",
                "Welcome! I'm ready to assist. What would you like to discuss?"
            ]
            
            greeting = np.random.choice(greeting_messages)
            self.append_executive_message("T.A.R.S", greeting)
            
            # Speak the greeting
            try:
                self.speech.speak(greeting)
            except:
                pass
            
            # Wait a moment, then listen for response
            time.sleep(2)
            self.append_executive_message("SYSTEM", "Listening for your response...")
            
            # Listen for user response
            voice_input = self.speech.recognize_once().strip()
            
            if voice_input:
                # Process the voice input through your existing chat system
                self.command_input.setText(voice_input)
                self.process_command()
            else:
                self.append_executive_message("SYSTEM", "No voice input detected - voice conversation ended")
                
        except Exception as e:
            self.append_executive_message("SYSTEM", f"Auto voice conversation error: {e}")

    def setup_timers_and_threads(self):
        # High-performance video processing
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_visual_intelligence)
        self.video_timer.start(30)  # 33 FPS for premium experience
        
        # Hexagon glow animation
        self.glow_timer = QTimer()
        self.glow_timer.timeout.connect(self.animate_hexagon_glow)
        self.glow_timer.start(50)  # 20 FPS for ambient glow
        
        # Frame counter for face recognition optimization
        self._frame_counter = 0
        
        # Start continuous wake word listening
        self.audio_manager.start_wake_words()
    
        # Start face recognition by default
        if hasattr(self, 'face_manager'):
            self.face_manager.start_recognition()
            self.identity_panel.setText("FACE ID: SCANNING...")
            self.append_executive_message("SYSTEM", "Face identification service activated automatically")

    def toggle_face_identification(self):
        """Toggle face identification service on/off"""
        if self.face_id_btn.isChecked():
            if hasattr(self, 'face_manager'):
                self.face_manager.start_recognition()
                self.append_executive_message("SYSTEM", "Face identification service activated")
                self.identity_panel.setText("FACE ID: SCANNING...")
        else:
            if hasattr(self, 'face_manager'):
                self.face_manager.stop_recognition()
                self.append_executive_message("SYSTEM", "Face identification service deactivated")
                self.identity_panel.setText("FACE ID: STANDBY")

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
            if not self.audio_manager.is_wake_word_active():
                self.audio_manager.start_wake_words()
        else:
            self.audio_manager.stop_wake_words()

    def animate_hexagon_glow(self):
        self.video_display.advance_glow()

    def update_visual_intelligence(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize for better performance (smaller processing size)
            processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # Store original sized frame for face recognition (better quality)
            self._current_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LANCZOS4)
            
            # Update face recognition manager with current frame
            self.face_manager.update_frame(self._current_frame)
            
            # Apply background removal if active
            if self.background_removal_active:
                # Process background removal on smaller frame for performance
                bg_removed_frame = self.background_remover.remove_background(processed_frame)
                
                # Resize back to display size
                display_frame = cv2.resize(bg_removed_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = self._current_frame
            
            # Update display
            self.video_display.set_frame(display_frame)
           
            # Update frame counter
            self._frame_counter += 1

    def process_command(self):
        command_text = self.command_input.text().strip()
        if not command_text:
            return
        
        self.append_executive_message("EXECUTIVE", command_text)
        self.command_input.clear()
        
        # Process through audio manager for consistent handling
        threading.Thread(target=self._executive_ai_processing, args=(command_text,), daemon=True).start()

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
        
        # Executive text-to-speech
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
        try:
            cursor = self.executive_chat.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.insertHtml(formatted_message)
            self.executive_chat.moveCursor(QTextCursor.Start)
        except Exception:
            self.executive_chat.append(formatted_message)


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
            border-radius: 6px;
            font-family: 'Arial';
        }}
    """)
    
    ui = PremiumTarsUI(provider=None, chat=None, speech=None)
    ui.show()
    
    sys.exit(app.exec())