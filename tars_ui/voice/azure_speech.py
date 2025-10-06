import os
import azure.cognitiveservices.speech as speechsdk

class Speech:
    def __init__(self, key: str = None, region: str = None, voice: str = None):
        key = key or os.getenv("AZURE_SPEECH_KEY")
        region = region or os.getenv("AZURE_SPEECH_REGION")
        if not key or not region:
            raise RuntimeError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        self.speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        # Optional voice
        voice = voice or os.getenv("AZURE_SPEECH_VOICE")
        if voice:
            self.speech_config.speech_synthesis_voice_name = voice

    def recognize_once(self) -> str:
        audio_cfg = speechsdk.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_cfg)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            return ""
        return ""

    def speak(self, text: str):
        audio_out = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synth = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_out)
        _ = synth.speak_text_async(text).get()

    def stop_speaking(self):
        """Stop current speech synthesis"""
        try:
            # For Azure Speech SDK, you might need to implement this
            # depending on your specific implementation
            self.should_stop_speaking = True
        except Exception:
            pass

