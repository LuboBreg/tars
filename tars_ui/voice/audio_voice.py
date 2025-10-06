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


"""
Complete AudioManager class with interrupt detection functionality
This is a drop-in replacement for your existing AudioManager class
"""

import time
import threading
import random

"""
Fixed AudioManager with proper conversation management and reliable interrupt detection
This version prevents overlapping conversations and uses a simpler interrupt approach
"""

import time
import threading
import random

class AudioManager:
    """High-level audio management with proper conversation control and interrupts"""
    
    def __init__(self, speech_client, chat_client=None, ui_callback=None):
        """Initialize audio manager"""
        # Initialize sub-components
        self.wake_word_manager = WakeWordManager(speech_client, ui_callback)
        self.voice_processor = VoiceCommandProcessor(speech_client, chat_client, ui_callback)
        
        # Unified callback for UI interactions
        self.ui_callback = ui_callback
        
        # Conversation state management
        self.auto_voice_enabled = True
        self.last_auto_trigger = 0
        self.auto_trigger_cooldown = 45  # Increased to prevent overlaps
        self.max_conversation_turns = 5
        self.current_conversation_active = False
        self.conversation_history = []
        
        # Interrupt detection - simplified approach
        self.conversation_interrupted = False
        self.interrupt_phrases = ['stop', 'pause', 'interrupt', 'halt', 'cancel', 'end', 'quiet']
        
        # Speech state tracking
        self.currently_speaking = False
        self.speech_lock = threading.Lock()
        
        # Debug flag
        self.debug_mode = True

    def _debug_log(self, message):
        """Debug logging"""
        if self.debug_mode and self.ui_callback:
            self.ui_callback('add_message', "DEBUG", f"[AUDIO] {message}")

    def trigger_auto_voice_greeting(self):
        """Trigger automatic voice conversation when face is detected - OVERLAP PREVENTION"""
        # Prevent overlapping conversations
        if self.current_conversation_active:
            self._debug_log("Skipping auto voice - conversation already active")
            return
            
        if not self.auto_voice_enabled:
            self._debug_log("Auto voice disabled")
            return
            
        # Rate limiting with longer cooldown to prevent rapid triggers
        current_time = time.time()
        if current_time - self.last_auto_trigger < self.auto_trigger_cooldown:
            self._debug_log(f"Rate limited - {self.auto_trigger_cooldown - (current_time - self.last_auto_trigger):.1f}s remaining")
            return
            
        # Check if currently speaking
        with self.speech_lock:
            if self.currently_speaking:
                self._debug_log("Skipping auto voice - currently speaking")
                return
        
        self.last_auto_trigger = current_time
        self._debug_log("Triggering auto voice conversation")
        
        # Start new conversation
        self.current_conversation_active = True
        self.conversation_history = []
        self.conversation_interrupted = False
        
        # Start conversation in separate thread
        threading.Thread(target=self._multi_turn_conversation_thread, daemon=True).start()

    def _multi_turn_conversation_thread(self):
        """Handle multi-turn conversation with simple interrupt detection"""
        try:
            self._debug_log("Starting conversation thread")
            
            # Mark conversation as active
            self.current_conversation_active = True
            self.conversation_interrupted = False
            
            # Initial greeting
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", "Executive detected - starting conversation...")
            
            greetings = [
                "Hello! I can see you. How can I assist you today?",
                "Good to see you! What can I help you with?",
                "Welcome! I'm ready to assist. What would you like to discuss?",
                "Executive detected. How may I support your objectives today?"
            ]
            
            greeting = random.choice(greetings)
            self.conversation_history.append({"role": "assistant", "content": greeting})
            
            # Display greeting
            if self.ui_callback:
                self.ui_callback('add_message', "T.A.R.S", greeting)
            
            # Speak greeting with interrupt check
            if not self._speak_with_simple_interrupt_check(greeting):
                self._debug_log("Conversation interrupted during greeting")
                return
            
            # Main conversation loop
            turn_count = 0
            while (turn_count < self.max_conversation_turns and 
                   self.current_conversation_active and 
                   not self.conversation_interrupted):
                
                self._debug_log(f"Starting turn {turn_count + 1}")
                
                try:
                    # Brief pause
                    time.sleep(1.5)
                    
                    if self.conversation_interrupted:
                        break
                        
                    if self.ui_callback:
                        self.ui_callback('add_message', "SYSTEM", f"Listening... (Turn {turn_count + 1}/{self.max_conversation_turns})")
                    
                    # Listen for user input with timeout
                    voice_input = self._listen_for_user_input()
                    
                    if self.conversation_interrupted:
                        break
                        
                    if not voice_input:
                        if self.ui_callback:
                            self.ui_callback('add_message', "SYSTEM", "No input detected. Ending conversation.")
                        break
                    
                    # Check for stop phrases immediately
                    if self._check_for_stop_phrases(voice_input):
                        if self.ui_callback:
                            self.ui_callback('add_message', "SYSTEM", f"Conversation ended by user: '{voice_input}'")
                        break
                    
                    # Add to conversation history
                    self.conversation_history.append({"role": "user", "content": voice_input})
                    
                    # Display user input
                    if self.ui_callback:
                        self.ui_callback('add_message', "EXECUTIVE", voice_input)
                    
                    # Generate AI response
                    ai_response = self._generate_contextual_response(self.conversation_history)
                    
                    if self.conversation_interrupted:
                        break
                    
                    # Add AI response to history
                    self.conversation_history.append({"role": "assistant", "content": ai_response})
                    
                    # Display AI response
                    if self.ui_callback:
                        self.ui_callback('add_message', "T.A.R.S", ai_response)
                    
                    # Speak AI response with interrupt check
                    if not self._speak_with_simple_interrupt_check(ai_response):
                        break
                    
                    turn_count += 1
                    
                except Exception as e:
                    self._debug_log(f"Error in conversation turn: {e}")
                    if self.ui_callback:
                        self.ui_callback('add_message', "SYSTEM", f"Voice error: {e}")
                    break
            
            # End conversation
            if self.conversation_interrupted:
                end_message = "Conversation interrupted by user."
            elif turn_count >= self.max_conversation_turns:
                end_message = "Conversation limit reached."
            else:
                end_message = "Conversation ended."
                
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", end_message)
                
        except Exception as e:
            self._debug_log(f"Error in conversation thread: {e}")
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", f"Conversation error: {e}")
        finally:
            # Always reset conversation state
            self.current_conversation_active = False
            self.conversation_interrupted = False
            with self.speech_lock:
                self.currently_speaking = False
            self._debug_log("Conversation thread ended")

    def _speak_with_simple_interrupt_check(self, text):
        """Speak text with simple interrupt detection - chunks approach"""
        self._debug_log(f"Speaking: {text[:50]}...")
        
        try:
            with self.speech_lock:
                self.currently_speaking = True
            
            # Split text into smaller chunks for interrupt checking
            sentences = text.replace('.', '.|').replace('!', '!|').replace('?', '?|').split('|')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i, sentence in enumerate(sentences):
                # Check for interrupt before each sentence
                if self.conversation_interrupted:
                    self._debug_log("Interrupt detected - stopping speech")
                    return False
                
                # Quick interrupt check using a very short listen
                if i > 0:  # Don't check before first sentence
                    if self._quick_interrupt_check():
                        self._debug_log("Quick interrupt detected")
                        return False
                
                # Speak this sentence
                try:
                    self.voice_processor.speech.speak(sentence)
                except Exception as e:
                    self._debug_log(f"Speech error: {e}")
                
                # Small pause between sentences
                if i < len(sentences) - 1:
                    time.sleep(0.3)
            
            return True
            
        except Exception as e:
            self._debug_log(f"Error in speak_with_interrupt_check: {e}")
            return True
        finally:
            with self.speech_lock:
                self.currently_speaking = False

    def _quick_interrupt_check(self):
        """Quick check for interrupt without blocking"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            mic = sr.Microphone()
            
            with mic as source:
                # Very quick listen - 0.2 seconds max
                try:
                    r.adjust_for_ambient_noise(source, duration=0.05)
                    audio = r.listen(source, timeout=0.1, phrase_time_limit=0.3)
                    
                    # Try to recognize quickly
                    text = r.recognize_google(audio, show_all=False).lower().strip()
                    self._debug_log(f"Quick check heard: '{text}'")
                    
                    # Check for interrupt phrases
                    if self._check_for_stop_phrases(text):
                        self.conversation_interrupted = True
                        if self.ui_callback:
                            self.ui_callback('add_message', "SYSTEM", f"Interrupted by: '{text}'")
                        return True
                        
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    # No speech or unclear speech - not an interrupt
                    pass
                except sr.RequestError as e:
                    self._debug_log(f"Recognition error in quick check: {e}")
            
            return False
            
        except Exception as e:
            self._debug_log(f"Error in quick interrupt check: {e}")
            return False

    def _listen_for_user_input(self):
        """Listen for user input with interrupt awareness"""
        if self.conversation_interrupted:
            return None
            
        try:
            self._debug_log("Listening for user input...")
            
            # Use existing speech recognition with timeout
            voice_input = self.voice_processor.speech.recognize_once(timeout=8).strip()
            
            if voice_input:
                self._debug_log(f"User said: '{voice_input}'")
                return voice_input
            else:
                self._debug_log("No voice input detected")
                return None
                
        except Exception as e:
            self._debug_log(f"Error listening for input: {e}")
            return None

    def _check_for_stop_phrases(self, text):
        """Check if text contains stop/interrupt phrases"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        for phrase in self.interrupt_phrases:
            if phrase in text_lower:
                self._debug_log(f"Stop phrase '{phrase}' detected in '{text_lower}'")
                return True
        
        return False

    def _generate_contextual_response(self, conversation_history):
        """Generate AI response using conversation context"""
        try:
            messages = [
                {
                    "role": "system", 
                    "content": """You are T.A.R.S, an elite AI system designed for Deloitte executives. You are:
                    - Highly intelligent and strategic
                    - Professional and concise (keep responses under 100 words)
                    - Focused on business insights and executive decision support
                    - Capable of wit when appropriate, but always professional
                    - Knowledgeable about consulting, technology, and business strategy
                    
                    Keep responses conversational but professional. Be concise to allow for natural conversation flow."""
                }
            ]
            
            # Add recent conversation history
            recent_history = conversation_history[-6:]
            messages.extend(recent_history)
            
            # Generate response
            ai_response = self.voice_processor.chat.chat(messages)
            return ai_response
            
        except Exception as e:
            return f"I encountered an error: {e}"

    def end_current_conversation(self):
        """Manually end the current conversation"""
        self._debug_log("Manually ending conversation")
        self.conversation_interrupted = True
        self.current_conversation_active = False
        with self.speech_lock:
            self.currently_speaking = False

    def is_conversation_active(self):
        """Check if a conversation is currently active"""
        return self.current_conversation_active

    def set_auto_voice_enabled(self, enabled):
        """Enable/disable auto voice conversations"""
        self.auto_voice_enabled = enabled
        self._debug_log(f"Auto voice {'enabled' if enabled else 'disabled'}")

    def get_conversation_status(self):
        """Get detailed conversation status"""
        return {
            'active': self.current_conversation_active,
            'interrupted': self.conversation_interrupted,
            'speaking': self.currently_speaking,
            'auto_enabled': self.auto_voice_enabled,
            'turns_completed': len([msg for msg in self.conversation_history if msg['role'] == 'user']),
            'max_turns': self.max_conversation_turns,
            'cooldown_remaining': max(0, self.auto_trigger_cooldown - (time.time() - self.last_auto_trigger))
        }

    # Keep all your existing methods unchanged
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

    def set_max_conversation_turns(self, turns):
        """Set maximum number of conversation turns"""
        self.max_conversation_turns = max(1, min(10, turns))

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def set_debug_mode(self, enabled):
        """Enable/disable debug logging"""
        self.debug_mode = enabled
        

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