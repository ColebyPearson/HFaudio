# HuggingFace Audio Transformers Course

## Project Overview
This repo contains notebooks and code for completing the HuggingFace Audio Transformers Course.
Course URL: https://huggingface.co/learn/audio-course/chapter0/introduction
GitHub: https://github.com/huggingface/audio-transformers-course

## Certification Requirements
- **Certificate of completion**: Complete 3 of 4 hands-on exercises (80%)
- **Certificate of honors**: Complete all 4 hands-on exercises (100%)

### Hands-on Exercises
1. **Unit 4** - Music Genre Classifier: Fine-tune on `marsyas/gtzan`, achieve >=87% accuracy, push to Hub
2. **Unit 5** - Speech Recognition: Fine-tune `openai/whisper-tiny` on `PolyAI/minds14` en-US, WER < 0.37, push to Hub
3. **Unit 6** - Text-to-Speech: Fine-tune SpeechT5 on chosen dataset, push to Hub tagged `text-to-speech`
4. **Unit 7** - Speech-to-Speech Translation: Build Gradio demo translating to non-English, deploy on HF Spaces

## Structure
- `unit_*/` - Notebooks organized by course unit
- Notebooks are designed to run on Google Colab (free GPU tier)

## Key Libraries
- transformers, datasets, evaluate, accelerate
- librosa, soundfile, torchaudio
- jiwer (WER metric), gradio (demos)
