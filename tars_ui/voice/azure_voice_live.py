# File: voice/azure_voice_live.py

import os
import asyncio
import json
import websockets
import base64
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import numpy as np

class VoiceLiveEventHandler:
    """Base event handler for Voice Live API"""
    
    def __init__(self):
        self.event_handlers = {}
    
    def on(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def dispatch(self, event_type: str, event_data: Any):
        """Dispatch an event to registered handlers"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event_data))
                else:
                    handler(event_data)
            except Exception as e:
                print(f"Error in event handler for {event_type}: {e}")

class VoiceLiveAPI(VoiceLiveEventHandler):
    """Azure Voice Live API WebSocket client"""
    
    def __init__(self):
        super().__init__()
        self.endpoint = os.getenv("AZURE_VOICE_LIVE_ENDPOINT")
        self.api_key = os.getenv("AZURE_VOICE_LIVE_API_KEY")
        self.model = os.getenv("VOICE_LIVE_MODEL", "gpt-4o")
        self.api_version = os.getenv("AZURE_VOICE_LIVE_API_VERSION", "2025-05-01-preview")
        self.ws = None
        self.connected = False
    
    def is_connected(self):
        return self.connected and self.ws is not None
    
    async def connect(self):
        """Connect to Azure Voice Live API"""
        if self.is_connected():
            raise Exception("Already connected")
        
        if not self.endpoint or not self.api_key:
            raise Exception("AZURE_VOICE_LIVE_ENDPOINT and AZURE_VOICE_LIVE_API_KEY must be set")
        
        url = f"{self.endpoint}/voice-live/realtime?api-version={self.api_version}&deployment={self.model}&api-key={self.api_key}"
        
        try:
            self.ws = await websockets.connect(url, extra_headers={
                'Authorization': f'Bearer {self.api_key}'
            })
            self.connected = True
            print(f"Connected to Azure Voice Live API: {self.endpoint}")
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
        except Exception as e:
            print(f"Failed to connect to Voice Live API: {e}")
            raise
    
    async def _listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                print(f"Received: {event.get('type', 'unknown')}")
                
                if event.get('type') == 'error':
                    print(f"API Error: {event}")
                
                # Dispatch events
                self.dispatch(f"server.{event['type']}", event)
                self.dispatch("server.*", event)
                
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            print(f"Error in message listener: {e}")
            self.connected = False
    
    async def send_event(self, event_type: str, data: Dict = None):
        """Send an event to the API"""
        if not self.is_connected():
            raise Exception("Not connected to Voice Live API")
        
        data = data or {}
        event = {
            "event_id": self._generate_id("evt_"),
            "type": event_type,
            **data
        }
        
        print(f"Sending: {event_type}")
        await self.ws.send(json.dumps(event))
        
        # Dispatch client event
        self.dispatch(f"client.{event_type}", event)
    
    def _generate_id(self, prefix: str):
        """Generate a unique ID"""
        return f"{prefix}{int(datetime.utcnow().timestamp() * 1000)}"
    
    async def disconnect(self):
        """Disconnect from the API"""
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.connected = False
            print("Disconnected from Voice Live API")

class VoiceLiveClient:
    """Main client for Azure Voice Live API integration"""
    
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.api = VoiceLiveAPI()
        self.session_configured = False
        self.conversation_active = False
        
        # Audio buffers
        self.input_audio_buffer = bytearray()
        
        # Session configuration
        self.session_config = {
            "modalities": ["text", "audio"],
            "instructions": self.system_prompt,
            "voice": {
                "name": "en-US-Aria:DragonHDLatestNeural",
                "type": "azure-standard",
                "temperature": 0.8,
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "azure_semantic_vad",
                "threshold": 0.3,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 200,
            },
            "tools": [],
            "tool_choice": "auto",
            "temperature": 0.8,
            "max_response_output_tokens": 1024,
        }
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the API"""
        self.api.on("server.session.created", self._on_session_created)
        self.api.on("server.response.audio.delta", self._on_audio_delta)
        self.api.on("server.response.audio_transcript.delta", self._on_transcript_delta)
        self.api.on("server.input_audio_buffer.speech_started", self._on_speech_started)
        self.api.on("server.input_audio_buffer.speech_stopped", self._on_speech_stopped)
        self.api.on("server.conversation.item.created", self._on_item_created)
        self.api.on("server.response.created", self._on_response_created)
        self.api.on("server.response.done", self._on_response_done)
    
    async def connect(self):
        """Connect to the Voice Live API and set up session"""
        await self.api.connect()
        await self._configure_session()
        await self._wait_for_session()
    
    async def _configure_session(self):
        """Configure the session with our settings"""
        await self.api.send_event("session.update", {
            "session": self.session_config
        })
    
    async def _wait_for_session(self):
        """Wait for session to be created"""
        while not self.session_configured:
            await asyncio.sleep(0.01)
    
    async def _on_session_created(self, event):
        """Handle session created event"""
        print("Session created successfully")
        self.session_configured = True
    
    async def _on_audio_delta(self, event):
        """Handle incoming audio delta"""
        audio_data = event.get('delta', '')
        if audio_data:
            # Convert base64 to audio bytes
            audio_bytes = base64.b64decode(audio_data)
            # Here you would send audio to your audio output system
            # For integration with your app, dispatch to audio callback
            if hasattr(self, 'audio_callback'):
                self.audio_callback('audio_chunk', audio_bytes)
    
    async def _on_transcript_delta(self, event):
        """Handle transcript delta"""
        transcript = event.get('delta', '')
        if transcript and hasattr(self, 'audio_callback'):
            self.audio_callback('transcript_chunk', transcript)
    
    async def _on_speech_started(self, event):
        """Handle speech started event"""
        print("Speech started")
        if hasattr(self, 'audio_callback'):
            self.audio_callback('speech_started')
    
    async def _on_speech_stopped(self, event):
        """Handle speech stopped event"""
        print("Speech stopped")
        if hasattr(self, 'audio_callback'):
            self.audio_callback('speech_stopped')
    
    async def _on_item_created(self, event):
        """Handle conversation item created"""
        item = event.get('item', {})
        if hasattr(self, 'audio_callback'):
            self.audio_callback('item_created', item)
    
    async def _on_response_created(self, event):
        """Handle response created"""
        self.conversation_active = True
        if hasattr(self, 'audio_callback'):
            self.audio_callback('response_started')
    
    async def _on_response_done(self, event):
        """Handle response done"""
        self.conversation_active = False
        if hasattr(self, 'audio_callback'):
            self.audio_callback('response_ended')
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to the API"""
        if not self.api.is_connected():
            return
        
        # Convert audio bytes to base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        await self.api.send_event("input_audio_buffer.append", {
            "audio": audio_b64
        })
        
        # Add to local buffer
        self.input_audio_buffer.extend(audio_data)
    
    async def send_text_message(self, text: str):
        """Send a text message"""
        await self.api.send_event("conversation.item.create", {
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })
        await self.create_response()
    
    async def create_response(self):
        """Create a response from the assistant"""
        await self.api.send_event("response.create")
    
    async def interrupt_response(self):
        """Interrupt the current response"""
        if self.conversation_active:
            await self.api.send_event("response.cancel")
    
    def set_audio_callback(self, callback):
        """Set callback for audio events"""
        self.audio_callback = callback
    
    async def disconnect(self):
        """Disconnect from the API"""
        await self.api.disconnect()
    
    def is_connected(self):
        """Check if connected"""
        return self.api.is_connected()

class VoiceLiveAudioManager:
    """Wrapper to integrate Voice Live API with your existing audio system"""
    
    def __init__(self, speech_client, chat_client, ui_callback=None):
        self.speech_client = speech_client  # Keep for compatibility
        self.chat_client = chat_client
        self.ui_callback = ui_callback
        
        # Voice Live client
        self.voice_live_client = None
        self.conversation_active = False
        self.auto_voice_enabled = True
        
        # Audio handling
        self.current_audio_data = bytearray()
        
    async def initialize_voice_live(self, system_prompt: str):
        """Initialize the Voice Live client"""
        self.voice_live_client = VoiceLiveClient(system_prompt)
        self.voice_live_client.set_audio_callback(self._handle_voice_live_callback)
        
        try:
            await self.voice_live_client.connect()
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", "Azure Voice Live API connected successfully")
            return True
        except Exception as e:
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", f"Failed to connect to Voice Live API: {e}")
            return False
    
    def _handle_voice_live_callback(self, action, *args):
        """Handle callbacks from Voice Live client"""
        if action == 'audio_chunk':
            audio_bytes = args[0]
            # Here you would play the audio through your audio system
            # For now, we'll just log it
            print(f"Received audio chunk: {len(audio_bytes)} bytes")
            
        elif action == 'transcript_chunk':
            transcript = args[0]
            if self.ui_callback:
                self.ui_callback('add_message', "T.A.R.S", transcript)
                
        elif action == 'speech_started':
            if self.ui_callback:
                self.ui_callback('add_message', "DEBUG", "Speech detection started")
                
        elif action == 'speech_stopped':
            if self.ui_callback:
                self.ui_callback('add_message', "DEBUG", "Speech detection stopped")
                
        elif action == 'response_started':
            self.conversation_active = True
            
        elif action == 'response_ended':
            self.conversation_active = False
    
    async def trigger_voice_conversation(self):
        """Trigger a voice conversation"""
        if not self.voice_live_client or not self.voice_live_client.is_connected():
            if self.ui_callback:
                self.ui_callback('add_message', "SYSTEM", "Voice Live API not connected")
            return
        
        # Send a greeting
        greeting = "Hello! How can I assist you today?"
        await self.voice_live_client.send_text_message(greeting)
        
        if self.ui_callback:
            self.ui_callback('add_message', "SYSTEM", "Voice conversation started with Azure Voice Live API")
    
    async def send_audio_data(self, audio_data: bytes):
        """Send audio data to Voice Live API"""
        if self.voice_live_client and self.voice_live_client.is_connected():
            await self.voice_live_client.send_audio_chunk(audio_data)
    
    async def send_text_message(self, text: str):
        """Send text message through Voice Live API"""
        if self.voice_live_client and self.voice_live_client.is_connected():
            await self.voice_live_client.send_text_message(text)
    
    async def interrupt_conversation(self):
        """Interrupt current conversation"""
        if self.voice_live_client and self.conversation_active:
            await self.voice_live_client.interrupt_response()
    
    def is_conversation_active(self):
        """Check if conversation is active"""
        return self.conversation_active
    
    async def disconnect(self):
        """Disconnect from Voice Live API"""
        if self.voice_live_client:
            await self.voice_live_client.disconnect()
    
    # Compatibility methods for your existing audio system
    def start_wake_words(self):
        """Compatibility method"""
        if self.ui_callback:
            self.ui_callback('add_message', "SYSTEM", "Voice Live API handles wake words automatically")
    
    def stop_wake_words(self):
        """Compatibility method"""
        pass
    
    def is_wake_word_active(self):
        """Compatibility method"""
        return True
    
    def process_voice_command(self, command_text=None):
        """Compatibility method"""
        if command_text and self.voice_live_client:
            asyncio.create_task(self.send_text_message(command_text))
    
    def end_current_conversation(self):
        """End current conversation"""
        if self.voice_live_client:
            asyncio.create_task(self.interrupt_conversation())
    
    def set_auto_voice_enabled(self, enabled):
        """Set auto voice enabled"""
        self.auto_voice_enabled = enabled
    
    def trigger_auto_voice_greeting(self):
        """Trigger auto voice greeting (compatibility)"""
        if self.voice_live_client:
            asyncio.create_task(self.trigger_voice_conversation())