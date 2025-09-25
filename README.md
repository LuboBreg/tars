# TARS-UI (Azure edition, macOS)

TARS-inspired UI with:
- **Azure OpenAI** for chat (Assistants-style minimal wrapper)
- **Azure Speech** for **voice in/out** (push-to-talk mic capture + TTS to speakers)
- **Face recognition** provider (default: **Azure Face**; optional: Incoresoft VEZHA® REST adapter)

> Fan-made UI inspired by *Interstellar*'s TARS (not affiliated).

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configure (`.env` alongside this README or export in shell)
```bash
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini      # your deployment name
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Azure Speech (for mic + TTS) (required for voice features)
AZURE_SPEECH_KEY=your-speech-key
AZURE_SPEECH_REGION=your-region

# Face (default = Azure Face; set keys if used)
AZURE_FACE_ENDPOINT=https://<region>.api.cognitive.microsoft.com
AZURE_FACE_KEY=your-azure-face-key
```

## Run
```bash
tars-ui
```

## Notes
- **Push-to-talk**: hold the mic button to capture a single utterance; release to stop and send to chat.
- **TTS voice**: uses your Azure Speech default voice; change with `AZURE_SPEECH_VOICE` (e.g., `en-US-JennyNeural`).

## Permissions (macOS)
- Grant Terminal/Python access to **Microphone** and **Camera** in *System Settings → Privacy & Security*.
- Close other apps that might use the camera/mic.