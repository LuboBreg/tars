"""
Face Recognition module for T.A.R.S Executive AI System
Contains face identification, emotion analysis, and related UI updates
"""

import time
import threading
import logging
from PySide6.QtGui import QColor

# Set up logging
logger = logging.getLogger(__name__)

# Import colors from constants - you should create a constants.py file
try:
    from ..constants import (
        DELOITTE_GREEN, DELOITTE_WHITE, DELOITTE_BLACK, 
        DELOITTE_CHARCOAL, DELOITTE_SILVER
    )
except ImportError:
    # Fallback colors if constants file doesn't exist
    DELOITTE_GREEN = QColor(134, 188, 37)
    DELOITTE_WHITE = QColor(247, 248, 249)
    DELOITTE_BLACK = QColor(18, 20, 24)
    DELOITTE_CHARCOAL = QColor(32, 35, 42)
    DELOITTE_SILVER = QColor(168, 170, 173)


class FaceRecognitionManager:
    """Manages face recognition operations and UI updates"""
    
    def __init__(self, face_provider, ui_callback=None):
        """
        Initialize face recognition manager
        
        Args:
            face_provider: Face provider instance (Azure, Incoresoft, etc.)
            ui_callback: Optional callback function for UI updates
        """
        self.provider = face_provider
        self.ui_callback = ui_callback
        
        # Recognition state
        self.active = False
        self.current_frame = None
        self.last_identity = None
        self.identity_lock = threading.Lock()
        self.recognition_thread = None
        
        # Recognition settings
        self.recognition_interval = 4.0  # seconds between recognition attempts
        self.max_retries = 3  # Maximum retries for recognition failures
        
    def start_recognition(self):
        """Start face recognition service"""
        if self.active:
            logger.warning("Face recognition already active")
            return
            
        self.active = True
        self._notify_ui("SYSTEM", "Face identification service activated")
        self._update_identity_panel("FACE ID: SCANNING...")
        
        # Start recognition thread
        if self.recognition_thread is None or not self.recognition_thread.is_alive():
            self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
            self.recognition_thread.start()
            logger.info("Face recognition thread started")
    
    def stop_recognition(self):
        """Stop face recognition service"""
        self.active = False
        self._notify_ui("SYSTEM", "Face identification service deactivated")
        self._update_identity_panel("FACE ID: STANDBY")
        
        # Clear identity data
        with self.identity_lock:
            self.last_identity = None
        
        logger.info("Face recognition service stopped")
    
    def update_frame(self, frame):
        """Update current frame for recognition"""
        if frame is not None and frame.size > 0:
            self.current_frame = frame
        else:
            logger.warning("Invalid frame provided for recognition")
    
    def get_current_identity(self):
        """Get current identity result (thread-safe)"""
        with self.identity_lock:
            return self.last_identity
    
    def _recognition_loop(self):
        """Main recognition loop - runs in separate thread"""
        retry_count = 0
        
        while self.active:
            frame = self.current_frame
            if frame is not None:
                try:
                    recognition_result = self.provider.identify(frame)
                    
                    with self.identity_lock:
                        self.last_identity = recognition_result
                    
                    # Handle recognition result
                    if recognition_result:
                        self._handle_recognition_result(recognition_result)
                        retry_count = 0  # Reset retry count on success
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Face recognition error (attempt {retry_count}): {e}")
                    
                    with self.identity_lock:
                        self.last_identity = None
                    
                    # If max retries reached, notify user
                    if retry_count >= self.max_retries:
                        self._notify_ui("SYSTEM", f"Face recognition temporarily unavailable after {self.max_retries} attempts")
                        retry_count = 0  # Reset for next cycle
                        
            time.sleep(self.recognition_interval)
    
    def _handle_recognition_result(self, result):
        """Process recognition result and update UI"""
        try:
            if hasattr(result, 'identity') and result.identity:
                # Successful identification
                confidence = f" ({result.confidence:.1%})" if hasattr(result, 'confidence') and result.confidence is not None else ""
                status_text = f"EXECUTIVE IDENTIFIED: {result.identity.upper()}{confidence}"
                
                # Update UI with success styling
                self._update_identity_panel(status_text, success=True)
                
                # Handle emotion-based interactions if available
                if hasattr(result, 'emotions') and result.emotions:
                    self._handle_emotion_analysis(result)
                    
            else:
                # Face detected but not identified
                self._update_identity_panel("FACE ID: SCANNING...")
                
        except Exception as e:
            logger.error(f"Error handling recognition result: {e}")
            self._update_identity_panel("FACE ID: ERROR")
    
    def _handle_emotion_analysis(self, face_analysis):
        """Handle emotion-based UI updates and interactions"""
        try:
            if not face_analysis or not hasattr(face_analysis, 'emotions') or not face_analysis.emotions:
                return

            # Create detailed status text
            status_text = self._create_detailed_status(face_analysis)
            
            # Get emotion-based styling
            dominant_emotion = self._get_dominant_emotion(face_analysis.emotions)
            emotion_color = self._get_emotion_color(dominant_emotion)
            
            # Update identity panel with emotion-based styling
            self._update_identity_panel(status_text, emotion_color=emotion_color)
            
            # Trigger emotion-based interactions
            self._trigger_emotion_interaction(face_analysis)
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
    
    def _create_detailed_status(self, face_analysis):
        """Create detailed status text from face analysis"""
        try:
            age_text = f"{face_analysis.age}" if hasattr(face_analysis, 'age') and face_analysis.age else 'N/A'
            gender_text = face_analysis.gender if hasattr(face_analysis, 'gender') and face_analysis.gender else 'N/A'
            
            dominant_emotion = self._get_dominant_emotion(face_analysis.emotions)
            emotion_confidence = max(face_analysis.emotions.values()) if face_analysis.emotions else 0
            
            return (
                f"EXECUTIVE PROFILE:\n"
                f"Age: {age_text} years\n"
                f"Gender: {gender_text}\n"
                f"Dominant Emotion: {dominant_emotion.upper()} ({emotion_confidence:.1%})"
            )
        except Exception as e:
            logger.error(f"Error creating detailed status: {e}")
            return "FACE ID: ANALYSIS ERROR"
    
    def _trigger_emotion_interaction(self, face_analysis):
        """Trigger conversational responses based on detected emotions"""
        try:
            if not face_analysis.emotions:
                return

            dominant_emotion = self._get_dominant_emotion(face_analysis.emotions)
            emotion_intensity = max(face_analysis.emotions.values())

            # Only trigger interactions for strong emotions
            if emotion_intensity <= 0.5:
                return

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

            possible_responses = interactions.get(dominant_emotion, interactions['neutral'])
            response = possible_responses[hash(str(face_analysis)) % len(possible_responses)]
            self._notify_ui("T.A.R.S", response)
            
        except Exception as e:
            logger.error(f"Error in emotion interaction: {e}")
    
    def _get_dominant_emotion(self, emotions):
        """Determine the dominant emotion"""
        try:
            return max(emotions, key=emotions.get) if emotions else 'neutral'
        except (ValueError, TypeError):
            logger.warning("Invalid emotions data")
            return 'neutral'

    def _get_emotion_color(self, emotion):
        """Get color based on detected emotion"""
        emotion_colors = {
            'happiness': DELOITTE_GREEN,
            'neutral': DELOITTE_SILVER,
            'surprise': QColor(255, 165, 0),  # Orange
            'sadness': QColor(100, 149, 237),  # Cornflower Blue
            'anger': QColor(220, 20, 60)  # Crimson
        }
        return emotion_colors.get(emotion, DELOITTE_SILVER)
    
    def _update_identity_panel(self, text, success=False, emotion_color=None):
        """Update identity panel through callback"""
        if not self.ui_callback:
            return
            
        try:
            # Determine styling
            if emotion_color:
                color = emotion_color
            elif success:
                color = DELOITTE_GREEN
            else:
                color = DELOITTE_GREEN
                
            # Create style information
            style_info = {
                'color': color,
                'success': success
            }
            
            # Call UI callback
            self.ui_callback('update_identity_panel', text, style_info)
            
        except Exception as e:
            logger.error(f"Error updating identity panel: {e}")
    
    def _notify_ui(self, sender, message):
        """Send message to UI through callback"""
        if not self.ui_callback:
            return
            
        try:
            self.ui_callback('add_message', sender, message)
        except Exception as e:
            logger.error(f"Error notifying UI: {e}")


class FaceRecognitionUIHelper:
    """Helper class for face recognition UI updates"""
    
    @staticmethod
    def create_identity_panel_style(color, success=False):
        """Create stylesheet for identity panel based on color and state"""
        try:
            return f"""
                QLabel {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba({color.red()//3}, {color.green()//3}, {color.blue()//3}, 240), 
                        stop:0.5 rgba({color.red()//2}, {color.green()//2}, {color.blue()//2}, 240),
                        stop:1 rgba({color.red()//3}, {color.green()//3}, {color.blue()//3}, 240));
                    border: 2px solid rgba({color.red()}, {color.green()}, {color.blue()}, 200);
                    border-radius: 12px;
                    color: rgba({DELOITTE_WHITE.red()}, {DELOITTE_WHITE.green()}, {DELOITTE_WHITE.blue()}, 255);
                    padding: 18px 24px;
                    font-weight: 600;
                    letter-spacing: 0.3px;
                }}
            """
        except Exception as e:
            logger.error(f"Error creating identity panel style: {e}")
            return FaceRecognitionUIHelper.create_default_identity_panel_style()
    
    @staticmethod
    def create_default_identity_panel_style():
        """Create default identity panel style"""
        return f"""
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
        """
    