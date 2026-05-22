# Audio Toolkit — MCP server

An [MCP](https://modelcontextprotocol.io) server that exposes the four models
fine-tuned/deployed during the HuggingFace Audio Transformers Course as tools
any MCP client (Claude Desktop, Claude Code, …) can call.

| Tool | What it does | Backed by |
|---|---|---|
| `classify_music_genre(audio_path, top_k=3)` | Music genre of a clip (10 GTZAN genres) | [VoicesColeby/distilhubert-finetuned-gtzan](https://huggingface.co/VoicesColeby/distilhubert-finetuned-gtzan) |
| `transcribe_audio(audio_path)` | English speech → text | [VoicesColeby/whisper-tiny-minds14-en-us](https://huggingface.co/VoicesColeby/whisper-tiny-minds14-en-us) |
| `synthesize_dutch_speech(text, out_path)` | Dutch text → spoken `.wav` | facebook/mms-tts-nld |
| `speech_to_speech_translation(audio_path)` | Speech in any language → spoken Dutch | the deployed [🤗 Space](https://huggingface.co/spaces/VoicesColeby/speech-to-speech-translation) (no local model) |

## Install

```bash
pip install -r requirements.txt
```

Models are **lazy-loaded** — the server starts instantly; the first call to
each tool downloads/loads its model (a few hundred MB each, cached after).

> Note: tools decode audio with **librosa**, not the `transformers` audio
> pipelines, because those import `torchcodec` (needs FFmpeg shared libs that
> aren't always installed). librosa uses the soundfile/audioread backend.

## Run

```bash
python server.py
```

This serves over **stdio**, which is what MCP clients connect to.

## Register with a client

**Claude Code:**

```bash
claude mcp add audio-toolkit -- python /absolute/path/to/mcp_audio_toolkit/server.py
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "audio-toolkit": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_audio_toolkit/server.py"]
    }
  }
}
```

Then ask the assistant things like *"classify the genre of ~/clip.wav"*,
*"transcribe ~/voicemail.wav"*, *"say 'goedemorgen' in Dutch"*, or
*"translate ~/english.wav into Dutch speech"*.

## Smoke-test the tool logic (without an MCP client)

The tool functions are importable and runnable directly:

```python
from server import classify_music_genre, transcribe_audio, synthesize_dutch_speech
print(synthesize_dutch_speech("Hallo, dit is een test.", "out.wav"))
print(classify_music_genre("some_song.wav"))
```
