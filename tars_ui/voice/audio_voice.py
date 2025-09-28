"""
Audio and Voice Processing module for T.A.R.S Executive AI System
Enhanced version with Azerbaijani language responses and natural conversation flow
"""

import time
import threading
import random


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
            
        # Support both English and Azerbaijani recognition
        try:
            text = r.recognize_google(audio, language='az-AZ')
            print(f"Recognized (Azerbaijani): '{text}'")
        except:
            text = r.recognize_google(audio)
            print(f"Recognized (English): '{text}'")
        
    except Exception as e:
        print(f"Voice system error: {e}")
    
    print("=== END DEBUG ===")


class WakeWordManager:
    """Manages wake word detection and voice command processing"""
    
    def __init__(self, speech_client, ui_callback=None):
        self.speech = speech_client
        self.ui_callback = ui_callback
        
        # Wake word state
        self.active = False
        self.listening_thread = None
        
        # Wake word mappings (support both English and Azerbaijani)
        self.wake_words = {
            # English wake words
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
            
            # Azerbaijani wake words
            'üz tanıma aktiv et': self._activate_face_id,
            'üz tanıma söndür': self._deactivate_face_id,
            'portret rejimi': self._activate_portrait_mode,
            'portret söndür': self._deactivate_portrait_mode,
            'salam tars': self._start_voice_chat,
            'hey tars': self._start_voice_chat,
        }
    
    def start_listening(self):
        """Start continuous wake word listening"""
        if self.active:
            return
            
        self.active = True
        self.listening_thread = threading.Thread(target=self._wake_word_listener, daemon=True)
        self.listening_thread.start()
        
        self._notify_ui("SYSTEM", "Wake word system activated.")
    
    def stop_listening(self):
        """Stop wake word listening"""
        self.active = False
        self._notify_ui("SYSTEM", "Wake word system deactivated")
    
    def _wake_word_listener(self):
        """Continuous listening for wake words"""
        while self.active:
            try:
                # Get audio input using default recognition
                audio_input = self.speech.recognize_once()
                
                if audio_input and len(audio_input.strip()) > 0:
                    audio_input = audio_input.lower().strip()
                    self._notify_ui("DEBUG", f"Recognition: '{audio_input}'")
                    
                    # Check for wake words
                    for wake_word, action in self.wake_words.items():
                        if wake_word.lower() in audio_input:
                            self._notify_ui("WAKE WORD", f"Detected: '{wake_word}' in '{audio_input}'")
                            try:
                                action()
                            except Exception as e:
                                self._notify_ui("ERROR", f"Wake word action failed: {e}")
                            break
                    else:
                        self._notify_ui("DEBUG", f"No wake words found in: '{audio_input}'")
                else:
                    self._notify_ui("DEBUG", "No audio input received")
                
            except Exception as e:
                self._notify_ui("DEBUG", f"Wake word listener error: {e}")
                time.sleep(0.5)
                continue
                
            time.sleep(0.2)
    
    def _activate_face_id(self):
        self._ui_action('activate_face_id')
    
    def _deactivate_face_id(self):
        self._ui_action('deactivate_face_id')
    
    def _activate_portrait_mode(self):
        self._ui_action('activate_portrait_mode')
    
    def _deactivate_portrait_mode(self):
        self._ui_action('deactivate_portrait_mode')
    
    def _set_portrait_style(self, style):
        self._ui_action('set_portrait_style', style)
    
    def _start_voice_chat(self):
        self._notify_ui("SYSTEM", "Voice chat activated - listening for your command...")
        threading.Thread(target=self._process_voice_chat, daemon=True).start()
    
    def _process_voice_chat(self):
        try:
            voice_input = self.speech.recognize_once()
            
            if voice_input and voice_input.strip():
                voice_input = voice_input.strip()
                self._ui_action('process_voice_command', voice_input)
            else:
                self._notify_ui("SYSTEM", "No voice input detected")
                
        except Exception as e:
            self._notify_ui("SYSTEM", f"Voice chat error: {e}")
    
    def _ui_action(self, action, *args):
        if self.ui_callback:
            self.ui_callback(action, *args)
    
    def _notify_ui(self, sender, message):
        if self.ui_callback:
            self.ui_callback('add_message', sender, message)


class VoiceCommandProcessor:
    """Processes voice commands and manages voice interactions with Azerbaijani responses"""
    
    def __init__(self, speech_client, chat_client, ui_callback=None):
        self.speech = speech_client
        self.chat = chat_client
        self.ui_callback = ui_callback
        self.voice_active = False
    
    def process_voice_command(self, command_text=None):
        """Process a voice command"""
        if self.voice_active:
            return
            
        self.voice_active = True
        
        try:
            if command_text is None:
                self._notify_ui("SYSTEM", "Voice command system activated...")
                voice_input = self.speech.recognize_once()
                
                if voice_input and voice_input.strip():
                    voice_input = voice_input.strip()
                else:
                    voice_input = None
            else:
                voice_input = command_text
                
            if voice_input:
                self._ui_action('set_command_input', voice_input)
                self._process_ai_command(voice_input)
            else:
                self._notify_ui("SYSTEM", "No voice input detected")
                
        except Exception as e:
            self._notify_ui("SYSTEM", f"Voice command error: {e}")
        finally:
            self.voice_active = False
    
    def _process_ai_command(self, command_text):
        self._notify_ui("EXECUTIVE", command_text)
        threading.Thread(target=self._ai_processing_thread, args=(command_text,), daemon=True).start()
    
    def _ai_processing_thread(self, command_text):
        try:
            # Detect user's language
            user_language = self._detect_user_language(command_text)
            
            if user_language == "az":
                language_instruction = "CRITICAL: Always respond in Azerbaijani language only!"
                error_message = "Sistem xətası baş verdi: "
            else:
                language_instruction = "CRITICAL: Always respond in English language only!"
                error_message = "Executive AI system encountered an error: "
            
            executive_messages = [
                {
                    "role": "system", 
                    "content": f"""You are T.A.R.S, an elite AI system for Deloitte executives. 
                    
                    {language_instruction}
                    - Keep responses extremely brief - maximum 25 words or 2 sentences
                    - Be direct, precise, and actionable 
                    - No elaboration unless specifically asked
                    - Think like a busy executive who needs quick, clear answers
                    - Use formal but friendly tone
                    - Match the language the user is using ({'Azerbaijani' if user_language == 'az' else 'English'})
                    """
                },
                {"role": "user", "content": command_text}
            ]
            
            ai_response = self.chat.chat(executive_messages)
            
        except Exception as e:
            ai_response = f"{error_message}{e}"
        
        self._notify_ui("T.A.R.S", ai_response)
        
        try:
            # Use appropriate TTS language
            if user_language == "az":
                try:
                    self.speech.speak(ai_response, language='az')
                except:
                    self.speech.speak(ai_response)
            else:
                self.speech.speak(ai_response)
        except Exception:
            pass
    
    def _detect_user_language(self, text):
        """Detect if user is speaking in Azerbaijani or English based on keywords"""
        if not text:
            return "en"  # Default to English
        
        text_lower = text.lower()
        
        # Azerbaijani indicators
        azerbaijani_words = [
            'salam', 'necə', 'nə', 'mən', 'sən', 'bu', 'o', 'var', 'yox', 'gəl', 'get', 
            'yaxşı', 'pis', 'böyük', 'kiçik', 'çox', 'az', 'hamı', 'heç', 'bəli', 'xeyr',
            'işləmək', 'etmək', 'olmaq', 'bilmək', 'istəmək', 'görmək', 'demək', 'vermək',
            'kömək', 'məsələ', 'layihə', 'işi', 'vaxt', 'gün', 'həftə', 'ay', 'il'
        ]
        
        # English indicators  
        english_words = [
            'hello', 'how', 'what', 'when', 'where', 'why', 'who', 'the', 'and', 'or',
            'yes', 'no', 'good', 'bad', 'big', 'small', 'work', 'project', 'help',
            'time', 'day', 'week', 'month', 'year', 'can', 'will', 'should', 'need'
        ]
        
        azerbaijani_count = sum(1 for word in azerbaijani_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Return detected language
        if azerbaijani_count > english_count:
            return "az"
        else:
            return "en"
    
    def _ui_action(self, action, *args):
        if self.ui_callback:
            self.ui_callback(action, *args)
    
    def _notify_ui(self, sender, message):
        if self.ui_callback:
            self.ui_callback('add_message', sender, message)


class AudioManager:
    """Enhanced audio management with Azerbaijani responses and natural conversation flow"""
    
    def __init__(self, speech_client, chat_client=None, ui_callback=None):
        # Initialize sub-components
        self.wake_word_manager = WakeWordManager(speech_client, ui_callback)
        self.voice_processor = VoiceCommandProcessor(speech_client, chat_client, ui_callback)
        self.ui_callback = ui_callback
        
        # Conversation state management
        self.conversation_active = False
        self.conversation_paused = False
        self.auto_voice_enabled = True
        self.last_trigger_time = 0
        self.cooldown_seconds = 45  # Prevent overlapping conversations
        
        # Natural conversation settings - support up to 3 user messages
        self.max_exchanges = 3  # Up to 3 back-and-forth exchanges
        self.current_exchange = 0
        self.conversation_history = []
        self.max_history_length = 8  # Keep last 8 messages (4 exchanges) for context
        
        # Interrupt control (support both languages)
        self.stop_phrases = [
            # English
            'stop', 'end', 'halt', 'cancel', 'enough', 'bye', 'goodbye',
            # Azerbaijani
            'dayandır', 'kifayət', 'bitir', 'sağol', 'əlvida', 'çıx'
        ]
        self.conversation_should_end = False
        
        # Face recognition control
        self.face_recognition_was_active = False
        
        # Threading control
        self.conversation_lock = threading.Lock()

    def trigger_auto_voice_greeting(self):
        """Trigger natural conversation when face is detected"""
        # Prevent overlapping conversations
        if not self.conversation_lock.acquire(blocking=False):
            self._log("Conversation already in progress - skipping trigger")
            return
        
        try:
            if self.conversation_active:
                self._log("Conversation already active")
                return
            
            if not self.auto_voice_enabled:
                self._log("Auto voice disabled")
                return
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_trigger_time < self.cooldown_seconds:
                remaining = self.cooldown_seconds - (current_time - self.last_trigger_time)
                self._log(f"Cooldown active ({remaining:.1f}s remaining)")
                return
            
            self.last_trigger_time = current_time
            self._log("Starting natural conversation")
            
            # Initialize conversation state
            self.conversation_active = True
            self.conversation_should_end = False
            self.current_exchange = 0
            self.conversation_history = []
            
            # Pause face recognition during conversation
            self._pause_face_recognition()
            
            # Start conversation in separate thread
            threading.Thread(target=self._natural_conversation_worker, daemon=True).start()
            
        except Exception as e:
            self._log(f"Error starting conversation: {e}")
            self.conversation_lock.release()

    def _pause_face_recognition(self):
        """Temporarily pause face recognition during conversation"""
        try:
            if self.ui_callback:
                # Check if face recognition is currently active
                self.ui_callback('check_face_recognition_status')
                # Temporarily disable it
                self.ui_callback('pause_face_recognition')
                self.face_recognition_was_active = True
                self._log("Face recognition paused for conversation")
        except Exception as e:
            self._log(f"Error pausing face recognition: {e}")

    def _resume_face_recognition(self):
        """Resume face recognition after conversation ends"""
        try:
            if self.face_recognition_was_active and self.ui_callback:
                self.ui_callback('resume_face_recognition')
                self._log("Face recognition resumed")
                self.face_recognition_was_active = False
        except Exception as e:
            self._log(f"Error resuming face recognition: {e}")

    def _natural_conversation_worker(self):
        """Main conversation worker with natural flow in user's preferred language"""
        try:
            self._log("Natural conversation started")
            
            # Mixed language greetings - system will detect and adapt
            greetings = [
                "Salam! Səni görürəm! Nə ilə kömək edə bilərəm?",
                "Hello! I can see you. What can I help you with?",
                "Salam! Sizə necə kömək edə bilərəm?",
                "Hi there! How can I assist you today?",
                "Xoş gəlmisiniz! Nə haqqında danışmaq istəyirsiniz?",
                "Welcome! What would you like to discuss?"
            ]
            
            greeting = random.choice(greetings)
            self.conversation_history.append({"role": "assistant", "content": greeting})
            
            self._send_message("T.A.R.S", greeting)
            self._speak_naturally(greeting)
            
            if self.conversation_should_end:
                return
            
            # Natural conversation loop - up to 3 exchanges
            for exchange in range(self.max_exchanges):
                self.current_exchange = exchange + 1
                
                if self.conversation_should_end:
                    break
                
                self._send_message("SYSTEM", f"Listening... ({self.current_exchange}/{self.max_exchanges})")
                
                # Listen for user input with natural timing
                user_input = self._listen_naturally()
                
                if self.conversation_should_end or not user_input:
                    if not user_input:
                        self._send_message("SYSTEM", "I didn't catch that. Feel free to speak up anytime!")
                    break
                
                # Process user input
                self.conversation_history.append({"role": "user", "content": user_input})
                
                # Trim conversation history to keep only recent context (last 3 user messages)
                self._trim_conversation_history()
                
                self._send_message("EXECUTIVE", user_input)
                
                # Check for conversation end
                if self._should_end_conversation(user_input):
                    farewell = self._get_natural_farewell(user_input)
                    self._send_message("T.A.R.S", farewell)
                    self._speak_naturally(farewell)
                    break
                
                # Generate contextual response
                ai_response = self._generate_natural_response()
                if not ai_response or self.conversation_should_end:
                    break
                
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                self._send_message("T.A.R.S", ai_response)
                
                # Speak response naturally
                self._speak_naturally(ai_response)
                
                if self.conversation_should_end:
                    break
                
                # Natural pause between exchanges
                time.sleep(0.8)
            
            # End conversation naturally
            if not self.conversation_should_end and self.current_exchange >= self.max_exchanges:
                # Detect language for final message
                final_language = "en"
                if self.conversation_history:
                    for msg in reversed(self.conversation_history):
                        if msg["role"] == "user":
                            final_language = self._detect_user_language(msg["content"])
                            break
                
                if final_language == "az":
                    final_message = "Söhbətə görə təşəkkür edirəm! Başqa bir şeyə ehtiyacınız varsa, buradayam."
                else:
                    final_message = "Thanks for chatting! I'll be here if you need anything else."
                    
                self._send_message("T.A.R.S", final_message)
                self._speak_naturally(final_message)
            
            self._send_message("SYSTEM", "Conversation completed")
            
        except Exception as e:
            self._log(f"Conversation error: {e}")
            self._send_message("SYSTEM", f"Conversation error: {e}")
            
        finally:
            # Always cleanup
            self._cleanup_conversation()

    def _listen_naturally(self):
        """Listen for user input with natural timing and interruption detection"""
        try:
            # Give user time to start speaking
            time.sleep(0.5)
            
            user_input = self.voice_processor.speech.recognize_once()
            
            if user_input and user_input.strip():
                user_input = user_input.strip()
                self._log(f"User said: '{user_input}'")
                return user_input
            
            return None
            
        except Exception as e:
            self._log(f"Listen error: {e}")
            return None

    def _speak_naturally(self, text):
        """Speak with natural pacing and interruption detection"""
        try:
            self._log(f"Speaking: {text[:50]}...")
            
            # Check for interruption before speaking
            if self.conversation_should_end:
                return False
            
            # Split into natural phrases for smoother delivery
            phrases = self._split_into_phrases(text)
            
            for i, phrase in enumerate(phrases):
                if self.conversation_should_end:
                    self._log("Speech interrupted")
                    return False
                
                # Speak phrase with Azerbaijani TTS if available
                try:
                    try:
                        self.voice_processor.speech.speak(phrase.strip(), language='az')
                    except:
                        self.voice_processor.speech.speak(phrase.strip())
                except Exception as e:
                    self._log(f"TTS error: {e}")
                
                # Natural pause between phrases (except last one)
                if i < len(phrases) - 1:
                    time.sleep(0.3)
            
            return True
            
        except Exception as e:
            self._log(f"Speak error: {e}")
            return True

    def _split_into_phrases(self, text):
        """Split text into natural speaking phrases"""
        # Split on natural pause points (both English and Azerbaijani)
        for delimiter in ['. ', '! ', '? ', ', ', ' - ', ': ']:
            text = text.replace(delimiter, delimiter + '|SPLIT|')
        
        phrases = [phrase.replace('|SPLIT|', '').strip() for phrase in text.split('|SPLIT|')]
        return [phrase for phrase in phrases if phrase]

    def _should_end_conversation(self, user_input):
        """Check if user wants to end conversation"""
        user_lower = user_input.lower().strip()
        
        for phrase in self.stop_phrases:
            if phrase in user_lower:
                self._log(f"End phrase detected: '{phrase}'")
                return True
        
        # Also check for very short responses that might indicate disengagement
        if len(user_input.strip()) <= 2:
            self._log("Very short response - might indicate disengagement")
            return True
        
        return False

    def _detect_user_language(self, text):
        """Detect if user is speaking in Azerbaijani or English based on keywords"""
        if not text:
            return "en"  # Default to English
        
        text_lower = text.lower()
        
        # Azerbaijani indicators
        azerbaijani_words = [
            'salam', 'necə', 'nə', 'mən', 'sən', 'bu', 'o', 'var', 'yox', 'gəl', 'get', 
            'yaxşı', 'pis', 'böyük', 'kiçik', 'çox', 'az', 'hamı', 'heç', 'bəli', 'xeyr',
            'işləmək', 'etmək', 'olmaq', 'bilmək', 'istəmək', 'görmək', 'demək', 'vermək',
            'kömək', 'məsələ', 'layihə', 'işi', 'vaxt', 'gün', 'həftə', 'ay', 'il'
        ]
        
        # English indicators  
        english_words = [
            'hello', 'how', 'what', 'when', 'where', 'why', 'who', 'the', 'and', 'or',
            'yes', 'no', 'good', 'bad', 'big', 'small', 'work', 'project', 'help',
            'time', 'day', 'week', 'month', 'year', 'can', 'will', 'should', 'need'
        ]
        
        azerbaijani_count = sum(1 for word in azerbaijani_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Return detected language
        if azerbaijani_count > english_count:
            return "az"
        else:
            return "en"

    def _generate_natural_response(self):
        """Generate contextually appropriate response in user's language with conversation context"""
        try:
            # Detect user's language from their latest message
            user_language = "en"  # Default
            if self.conversation_history:
                for msg in reversed(self.conversation_history):
                    if msg["role"] == "user":
                        user_language = self._detect_user_language(msg["content"])
                        break
            
            # Set language instruction based on detection
            if user_language == "az":
                language_instruction = "CRITICAL: Always respond in Azerbaijani language only!"
                language_note = "Azerbaijani"
            else:
                language_instruction = "CRITICAL: Always respond in English language only!"
                language_note = "English"
            
            self._log(f"Detected user language: {language_note}")
            
            # Build context-aware system prompt
            context_info = ""
            if len(self.conversation_history) > 1:
                # Get recent conversation for context (last 6 messages = 3 exchanges)
                recent_messages = self.conversation_history[-6:]
                context_info = f"\n\nConversation context:\n"
                for msg in recent_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context_info += f"{role}: {msg['content']}\n"
            
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are T.A.R.S, a professional AI assistant for Deloitte executives.

                    This is exchange {self.current_exchange}/{self.max_exchanges} of a natural conversation.
                    
                    {language_instruction}
                    
                    Guidelines:
                    - Keep responses conversational and brief (20-40 words)
                    - Reference previous context when relevant - user may be continuing a topic
                    - Be helpful but not overwhelming
                    - If this is the final exchange, provide a natural conclusion
                    - Speak like you're having a real conversation with a colleague
                    - Use formal but friendly tone
                    - Pay attention to what the user has said before in this conversation
                    - Match the language the user is using ({language_note})
                    
                    Current exchange: {self.current_exchange}/{self.max_exchanges}
                    {context_info}"""
                }
            ]
            
            # Add only the current user message to avoid token overflow
            if self.conversation_history:
                last_user_message = None
                for msg in reversed(self.conversation_history):
                    if msg["role"] == "user":
                        last_user_message = msg
                        break
                
                if last_user_message:
                    messages.append(last_user_message)
            
            response = self.voice_processor.chat.chat(messages)
            return response
            
        except Exception as e:
            self._log(f"AI response error: {e}")
            # Return error message in detected language
            if user_language == "az":
                return "Bu anda bunu emal etməkdə çətinlik çəkirəm."
            else:
                return "I'm having trouble processing that right now."

    def _get_natural_farewell(self, user_input=None):
        """Get appropriate farewell based on conversation in user's language"""
        # Detect language from user input or conversation history
        user_language = "en"
        if user_input:
            user_language = self._detect_user_language(user_input)
        elif self.conversation_history:
            for msg in reversed(self.conversation_history):
                if msg["role"] == "user":
                    user_language = self._detect_user_language(msg["content"])
                    break
        
        if user_language == "az":
            farewells = [
                "Söhbətə görə təşəkkür edirəm! Başqa bir şeyə ehtiyacınız varsa xəbər verin.",
                "Sizinlə danışmaq çox gözəl idi! Kömək lazım olanda buradayam.",
                "Təşəkkür edirəm! İstənilən vaxt müraciət edə bilərsiniz.",
                "Sizinlə söhbət etmək gözəl idi! Lazım olsa, yanınızdayam."
            ]
        else:
            farewells = [
                "Thanks for chatting! Let me know if you need anything else.",
                "Great talking with you! I'm here whenever you need help.",
                "Thank you! Feel free to reach out anytime.",
                "Good talking with you! I'll be around if you need me."
            ]
        
        return random.choice(farewells)

    def _trim_conversation_history(self):
        """Trim conversation history to keep only the last 3 user messages and their responses"""
        if len(self.conversation_history) <= self.max_history_length:
            return
        
        # Count user messages to ensure we keep only last 3
        user_message_count = 0
        keep_from_index = len(self.conversation_history)
        
        # Go backwards through history to find where to trim
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i]["role"] == "user":
                user_message_count += 1
                if user_message_count > 3:
                    keep_from_index = i + 1
                    break
        
        # Trim the history
        if keep_from_index > 0:
            trimmed_messages = len(self.conversation_history) - keep_from_index
            self.conversation_history = self.conversation_history[keep_from_index:]
            self._log(f"Trimmed {trimmed_messages} old messages from conversation history")

    def _cleanup_conversation(self):
        """Clean up after conversation ends"""
        self.conversation_active = False
        self.conversation_should_end = False
        self.current_exchange = 0
        
        # Resume face recognition
        self._resume_face_recognition()
        
        # Release conversation lock
        self.conversation_lock.release()
        
        self._log("Conversation cleanup completed")

    def end_current_conversation(self):
        """Manually end current conversation"""
        self._log("Manually ending conversation")
        self.conversation_should_end = True

    def is_conversation_active(self):
        """Check if conversation is active"""
        return self.conversation_active

    def set_auto_voice_enabled(self, enabled):
        """Enable/disable auto voice conversations"""
        self.auto_voice_enabled = enabled
        self._log(f"Auto voice {'enabled' if enabled else 'disabled'}")

    def get_conversation_status(self):
        """Get detailed conversation status"""
        return {
            'active': self.conversation_active,
            'current_exchange': self.current_exchange,
            'max_exchanges': self.max_exchanges,
            'auto_enabled': self.auto_voice_enabled,
            'face_recognition_paused': self.face_recognition_was_active,
            'cooldown_remaining': max(0, self.cooldown_seconds - (time.time() - self.last_trigger_time))
        }

    def _send_message(self, sender, message):
        """Send message to UI"""
        if self.ui_callback:
            self.ui_callback('add_message', sender, message)

    def _log(self, message):
        """Debug logging"""
        if self.ui_callback:
            self.ui_callback('add_message', "DEBUG", f"[CONVERSATION] {message}")

    # Keep existing compatibility methods
    def start_wake_words(self):
        self.wake_word_manager.start_listening()
    
    def stop_wake_words(self):
        self.wake_word_manager.stop_listening()
    
    def process_voice_command(self, command_text=None):
        self.voice_processor.process_voice_command(command_text)
    
    def is_wake_word_active(self):
        return self.wake_word_manager.active
    
    def is_voice_active(self):
        return self.voice_processor.voice_active
    
    def add_wake_word(self, wake_word, action_callback):
        self.wake_word_manager.wake_words[wake_word.lower()] = action_callback
    
    def remove_wake_word(self, wake_word):
        if wake_word.lower() in self.wake_word_manager.wake_words:
            del self.wake_word_manager.wake_words[wake_word.lower()]

    def set_max_conversation_turns(self, turns):
        """Set maximum conversation exchanges"""
        self.max_exchanges = max(1, min(5, turns))

    def clear_conversation_history(self):
        self.conversation_history = []


# Utility functions for audio system integration
def create_audio_system_callback(ui_instance):
    """Create a callback function that connects audio system to UI"""
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
            if hasattr(ui_instance, 'face_id_btn') and not ui_instance.face_id_btn.isChecked():
                ui_instance.face_id_btn.setChecked(True)
                ui_instance.toggle_face_identification()
        elif action == 'deactivate_face_id':
            if hasattr(ui_instance, 'face_id_btn') and ui_instance.face_id_btn.isChecked():
                ui_instance.face_id_btn.setChecked(False)
                ui_instance.toggle_face_identification()
        elif action == 'pause_face_recognition':
            # Temporarily disable face recognition during conversation
            if hasattr(ui_instance, 'face_manager'):
                ui_instance.face_manager.pause_recognition()
        elif action == 'resume_face_recognition':
            # Resume face recognition after conversation
            if hasattr(ui_instance, 'face_manager'):
                ui_instance.face_manager.resume_recognition()
        elif action == 'check_face_recognition_status':
            # Check if face recognition is active
            if hasattr(ui_instance, 'face_id_btn'):
                return ui_instance.face_id_btn.isChecked()
        elif action == 'activate_portrait_mode':
            if hasattr(ui_instance, 'background_remove_btn') and not ui_instance.background_remove_btn.isChecked():
                ui_instance.background_remove_btn.setChecked(True)
                ui_instance.toggle_background_removal()
        elif action == 'deactivate_portrait_mode':
            if hasattr(ui_instance, 'background_remove_btn') and ui_instance.background_remove_btn.isChecked():
                ui_instance.background_remove_btn.setChecked(False)
                ui_instance.toggle_background_removal()
        elif action == 'set_portrait_style':
            style = args[0]
            if hasattr(ui_instance, 'set_portrait_style'):
                ui_instance.set_portrait_style(style)
    
    return audio_callback