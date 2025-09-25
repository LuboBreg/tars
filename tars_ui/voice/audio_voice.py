"""
Audio and Voice Processing module for T.A.R.S Executive AI System
Contains wake word detection, voice commands, and audio processing functionality
"""

import time
import threading


def debug_voice_system():
    """Debug voice recognition system"""
    print("=== VOICE SYSTEM DEBUG ===")
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        mic = sr.Microphone()
        print(f"Available microphones: {sr.Microphone.list_microphone_names()}")
        
        print("Testing microphone for 3 seconds...")
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=3)
            print("Audio captured successfully")
            
        text = r.recognize_google(audio)
        print(f"Recognized: '{text}'")
        
    except Exception as e:
        print(f"Voice system error: {e}")
    
    print("=== END DEBUG ===")


class WakeWordManager:
    """Manages wake word detection and voice command processing"""
    
    def __init__(self, speech_client, ui_callback=None):
        """
        Initialize wake word manager
        
        Args:
            speech_client: Speech client instance for recognition and TTS
            ui_callback: Optional callback function for UI updates
        """
        self.speech = speech_client
        self.ui_callback = ui_callback
        
        # Wake word state
        self.active = False
        self.listening_thread = None
        
        # Wake word mappings
        self.wake_words = {
            'activate face': self._activate_face_id,
            'deactivate face': self._deactivate_face_id,
            'turn on face': self._activate_face_id,
            'turn off face': self._deactivate_face_id,
            'portrait mode': self._activate_portrait_mode,
            'disable portrait': self._deactivate_portrait_mode,
            'turn on portrait': self._activate_portrait_mode,
            'turn off portrait': self._deactivate_portrait_mode,
            'natural mode': lambda: self._set_portrait_style("natural"),
            'black white': lambda: self._set_portrait_style("black_white"),
            'high contrast': lambda: self._set_portrait_style("high_contrast"),
            'hey tars': self._start_voice_chat,
            'hello tars': self._start_voice_chat,
        }
    
    def start_listening(self):
        """Start continuous wake word listening"""
        if self.active:
            return
            
        self.active = True
        self.listening_thread = threading.Thread(target=self._wake_word_listener, daemon=True)
        self.listening_thread.start()
        
        self._notify_ui("SYSTEM", "Wake word system activated. Available commands:")
        self._notify_ui("SYSTEM", "• 'Activate Face' - Turn on face identification")
        self._notify_ui("SYSTEM", "• 'Portrait Mode' - Enable portrait background removal")
        self._notify_ui("SYSTEM", "• 'Natural Mode' / 'Black White' / 'High Contrast' - Switch portrait styles")
        self._notify_ui("SYSTEM", "• 'Hey TARS' - Start voice conversation")
    
    def stop_listening(self):
        """Stop wake word listening"""
        self.active = False
        self._notify_ui("SYSTEM", "Wake word system deactivated")
    
    def _wake_word_listener(self):
        """Continuous listening for wake words"""
        while self.active:
            try:
                # Listen for wake words with shorter timeout
                audio_input = self.speech.recognize_once(timeout=2).lower().strip()
                
                if audio_input:
                    # Check for wake word matches
                    for wake_word, action in self.wake_words.items():
                        if wake_word in audio_input:
                            self._notify_ui("WAKE WORD", f"Detected: '{wake_word}'")
                            action()
                            break
                
            except Exception:
                # Silence exceptions for continuous listening
                time.sleep(0.1)
                continue
                
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    def _activate_face_id(self):
        """Activate face identification via wake word"""
        self._ui_action('activate_face_id')
    
    def _deactivate_face_id(self):
        """Deactivate face identification via wake word"""
        self._ui_action('deactivate_face_id')
    
    def _activate_portrait_mode(self):
        """Activate portrait mode via wake word"""
        self._ui_action('activate_portrait_mode')
    
    def _deactivate_portrait_mode(self):
        """Deactivate portrait mode via wake word"""
        self._ui_action('deactivate_portrait_mode')
    
    def _set_portrait_style(self, style):
        """Set portrait style via wake word"""
        self._ui_action('set_portrait_style', style)
    
    def _start_voice_chat(self):
        """Start voice chat conversation"""
        self._notify_ui("SYSTEM", "Voice chat activated - listening for your command...")
        threading.Thread(target=self._process_voice_chat, daemon=True).start()
    
    def _process_voice_chat(self):
        """Process voice chat with longer listening time"""
        try:
            # Listen for longer command/question
            voice_input = self.speech.recognize_once(timeout=10).strip()
            
            if voice_input:
                self._ui_action('process_voice_command', voice_input)
            else:
                self._notify_ui("SYSTEM", "No voice input detected")
                
        except Exception as e:
            self._notify_ui("SYSTEM", f"Voice chat error: {e}")
    
    def _ui_action(self, action, *args):
        """Execute UI action through callback"""
        if self.ui_callback:
            self.ui_callback(action, *args)
    
    def _notify_ui(self, sender, message):
        """Send message to UI through callback"""
        if self.ui_callback:
            self.ui_callback('add_message', sender, message)


class VoiceCommandProcessor:
    """Processes voice commands and manages voice interactions"""
    
    def __init__(self, speech_client, chat_client, ui_callback=None):
        """
        Initialize voice command processor
        
        Args:
            speech_client: Speech client for recognition and TTS
            chat_client: Chat client for AI responses
            ui_callback: Optional callback function for UI updates
        """
        self.speech = speech_client
        self.chat = chat_client
        self.ui_callback = ui_callback
        
        # Voice state
        self.voice_active = False
    
    def process_voice_command(self, command_text=None):
        """Process a voice command - either provided text or capture from microphone"""
        if self.voice_active:
            return
            
        self.voice_active = True
        
        try:
            if command_text is None:
                # Capture voice input
                self._notify_ui("SYSTEM", "Voice command system activated...")
                voice_input = self.speech.recognize_once().strip()
            else:
                # Use provided text
                voice_input = command_text
                
            if voice_input:
                # Update UI with the command
                self._ui_action('set_command_input', voice_input)
                
                # Process the command through AI
                self._process_ai_command(voice_input)
            else:
                self._notify_ui("SYSTEM", "No voice input detected")
                
        except Exception as e:
            self._notify_ui("SYSTEM", f"Voice command error: {e}")
        finally:
            self.voice_active = False
    
    def _process_ai_command(self, command_text):
        """Process command through AI system"""
        self._notify_ui("EXECUTIVE", command_text)
        
        # Process in separate thread to avoid blocking
        threading.Thread(target=self._ai_processing_thread, args=(command_text,), daemon=True).start()
    
    def _ai_processing_thread(self, command_text):
        """AI processing thread"""
        try:
            # Create executive AI messages
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
        
        # Send response to UI
        self._notify_ui("T.A.R.S", ai_response)
        
        # Executive text-to-speech
        try:
            self.speech.speak(ai_response)
        except Exception:
            pass  # Silently handle TTS errors
    
    def _ui_action(self, action, *args):
        """Execute UI action through callback"""
        if self.ui_callback:
            self.ui_callback(action, *args)
    
    def _notify_ui(self, sender, message):
        """Send message to UI through callback"""
        if self.ui_callback:
            self.ui_callback('add_message', sender, message)


class AudioManager:
    """High-level audio management combining wake words and voice commands"""
    
    def __init__(self, speech_client, chat_client=None, ui_callback=None):
        """
        Initialize audio manager
        
        Args:
            speech_client: Speech client for recognition and TTS
            chat_client: Optional chat client for AI responses
            ui_callback: Optional callback function for UI updates
        """
        self.wake_word_manager = WakeWordManager(speech_client, ui_callback)
        self.voice_processor = VoiceCommandProcessor(speech_client, chat_client, ui_callback)
        
        # Unified callback for UI interactions
        self.ui_callback = ui_callback
    
    def start_wake_words(self):
        """Start wake word listening"""
        self.wake_word_manager.start_listening()
    
    def stop_wake_words(self):
        """Stop wake word listening"""
        self.wake_word_manager.stop_listening()
    
    def process_voice_command(self, command_text=None):
        """Process voice command"""
        self.voice_processor.process_voice_command(command_text)
    
    def is_wake_word_active(self):
        """Check if wake word system is active"""
        return self.wake_word_manager.active
    
    def is_voice_active(self):
        """Check if voice processing is active"""
        return self.voice_processor.voice_active
    
    def add_wake_word(self, wake_word, action_callback):
        """Add a custom wake word and action"""
        self.wake_word_manager.wake_words[wake_word.lower()] = action_callback
    
    def remove_wake_word(self, wake_word):
        """Remove a wake word"""
        if wake_word.lower() in self.wake_word_manager.wake_words:
            del self.wake_word_manager.wake_words[wake_word.lower()]


# Utility functions for audio system integration
def create_audio_system_callback(ui_instance):
    """
    Create a callback function that connects audio system to UI
    
    Args:
        ui_instance: The main UI instance
        
    Returns:
        Callback function for audio system
    """
    def audio_callback(action, *args):
        if action == 'add_message':
            sender, message = args
            ui_instance.append_executive_message(sender, message)
        elif action == 'set_command_input':
            command_text = args[0]
            ui_instance.command_input.setText(command_text)
        elif action == 'process_voice_command':
            command_text = args[0]
            ui_instance.command_input.setText(command_text)
            ui_instance.process_command()
        elif action == 'activate_face_id':
            if not ui_instance.face_identification_active:
                ui_instance.face_id_btn.setChecked(True)
                ui_instance.toggle_face_identification()
        elif action == 'deactivate_face_id':
            if ui_instance.face_identification_active:
                ui_instance.face_id_btn.setChecked(False)
                ui_instance.toggle_face_identification()
        elif action == 'activate_portrait_mode':
            if not ui_instance.background_removal_active:
                ui_instance.background_remove_btn.setChecked(True)
                ui_instance.toggle_background_removal()
        elif action == 'deactivate_portrait_mode':
            if ui_instance.background_removal_active:
                ui_instance.background_remove_btn.setChecked(False)
                ui_instance.toggle_background_removal()
        elif action == 'set_portrait_style':
            style = args[0]
            ui_instance.set_portrait_style(style)
    
    return audio_callback