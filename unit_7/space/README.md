---
title: Speech To Speech Translation
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.32.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Speech-to-Speech Translation (→ Dutch)

Cascaded pipeline for the HuggingFace Audio Course Unit 7 hands-on exercise:

1. **Whisper** (`openai/whisper-base`) translates input speech in any language to English text.
2. **opus-mt** (`Helsinki-NLP/opus-mt-en-nl`) translates the English text to Dutch.
3. **MMS-TTS** (`facebook/mms-tts-nld`) synthesises Dutch speech.

Output speech is non-English (Dutch).
