"""Audio Toolkit MCP server.

Exposes the four models fine-tuned/deployed during the HuggingFace Audio
Transformers Course as MCP tools, so any MCP client (Claude Desktop, Claude
Code, etc.) can classify music, transcribe speech, synthesise Dutch speech,
and run speech-to-speech translation.

Design notes:
  - Models are lazy-loaded and cached on first use (the server starts fast;
    the first call to each tool downloads/loads its model).
  - Audio files are decoded with librosa (soundfile/audioread backend), NOT
    the transformers audio pipelines — those import torchcodec, which needs
    FFmpeg shared libs that aren't always present (e.g. this dev box).
  - speech_to_speech_translation calls the deployed Space via gradio_client,
    so it needs no local model at all.

Run:  python server.py        (stdio transport — what MCP clients use)
"""
from __future__ import annotations

import functools
from pathlib import Path

import librosa
import soundfile as sf
import torch
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("audio-toolkit")

HUB_USER = "VoicesColeby"
GENRE_MODEL = f"{HUB_USER}/distilhubert-finetuned-gtzan"
ASR_MODEL = f"{HUB_USER}/whisper-tiny-minds14-en-us"
TTS_MODEL = "facebook/mms-tts-nld"          # pretrained Dutch TTS (light, CPU-fine)
S2S_SPACE = f"{HUB_USER}/speech-to-speech-translation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- lazy model loaders (cached) -------------------------------------------

@functools.lru_cache(maxsize=1)
def _genre():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(GENRE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(GENRE_MODEL).to(DEVICE).eval()
    return fe, model


@functools.lru_cache(maxsize=1)
def _asr():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(ASR_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL).to(DEVICE).eval()
    return proc, model


@functools.lru_cache(maxsize=1)
def _tts():
    from transformers import VitsModel, VitsTokenizer
    tok = VitsTokenizer.from_pretrained(TTS_MODEL)
    model = VitsModel.from_pretrained(TTS_MODEL).to(DEVICE).eval()
    return tok, model


def _load_audio(path: str, target_sr: int):
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"audio file not found: {p}")
    y, _ = librosa.load(str(p), sr=target_sr, mono=True)  # librosa avoids torchcodec
    return y


# --- tools ------------------------------------------------------------------

@mcp.tool()
def classify_music_genre(audio_path: str, top_k: int = 3) -> dict:
    """Classify the music genre of an audio file.

    Uses a distilHuBERT model fine-tuned on GTZAN (10 genres: blues, classical,
    country, disco, hiphop, jazz, metal, pop, reggae, rock).

    Args:
        audio_path: path to a local audio file (wav/mp3/flac/...).
        top_k: how many ranked genre guesses to return.
    Returns:
        {"top_genre": str, "ranked": [{"genre": str, "score": float}, ...]}
    """
    fe, model = _genre()
    y = _load_audio(audio_path, fe.sampling_rate)
    inputs = fe(y, sampling_rate=fe.sampling_rate, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1)[0]
    order = probs.argsort(descending=True)[: max(1, top_k)]
    ranked = [
        {"genre": model.config.id2label[int(i)], "score": round(float(probs[int(i)]), 4)}
        for i in order
    ]
    return {"top_genre": ranked[0]["genre"], "ranked": ranked}


@mcp.tool()
def transcribe_audio(audio_path: str) -> dict:
    """Transcribe English speech in an audio file to text.

    Uses a Whisper-tiny model fine-tuned on the MINDS-14 (en-US) banking dataset.

    Args:
        audio_path: path to a local audio file.
    Returns:
        {"text": str}
    """
    proc, model = _asr()
    y = _load_audio(audio_path, 16000)
    feats = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=225)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return {"text": text}


@mcp.tool()
def synthesize_dutch_speech(text: str, out_path: str = "tts_out.wav") -> dict:
    """Synthesise Dutch speech from text and write it to a .wav file.

    Uses facebook/mms-tts-nld (Dutch). For the SpeechT5 model fine-tuned during
    the course, see VoicesColeby/speecht5-finetuned-voxpopuli-nl.

    Args:
        text: Dutch text to speak.
        out_path: where to write the .wav.
    Returns:
        {"out_path": str, "sample_rate": int, "seconds": float}
    """
    tok, model = _tts()
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        wav = model(**inputs).waveform[0].detach().cpu().numpy()
    sr = int(model.config.sampling_rate)
    out = str(Path(out_path).expanduser())
    sf.write(out, wav, sr)
    return {"out_path": out, "sample_rate": sr, "seconds": round(len(wav) / sr, 2)}


@mcp.tool()
def speech_to_speech_translation(audio_path: str) -> dict:
    """Translate speech in ANY language into spoken Dutch.

    Calls the deployed Gradio Space (Whisper -> opus-mt EN->NL -> MMS-TTS), so
    this needs no local model. Returns the path to the generated Dutch audio.

    Args:
        audio_path: path to a local audio file in any language.
    Returns:
        {"out_path": str}  # generated Dutch .wav (downloaded by the gradio client)
    """
    from gradio_client import Client, handle_file
    client = Client(S2S_SPACE)
    out = client.predict(handle_file(audio_path), api_name="/predict")
    return {"out_path": out}


if __name__ == "__main__":
    mcp.run()  # stdio transport
