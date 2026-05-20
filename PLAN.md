# Plan: Tackle the 4 Graded Exercises for HF Audio Course Certificate

## Context

The HuggingFace Audio Transformers Course has 4 hands-on exercises (Units 4-7) that must be completed for certification. Certificate of completion requires 3/4, certificate of honors requires 4/4. All notebooks are already scaffolded in the repo. The exercises build on each other -- Unit 6's TTS model can be reused in Unit 7's demo -- so order matters.

## Prerequisites (Before Starting Any Exercise)

- [ ] HuggingFace account created and API token generated
- [ ] Google Colab access (free tier GPU is sufficient for all exercises)
- [ ] Read through Units 1-3 notebooks to build foundational understanding
- [ ] Run `huggingface-cli login` or use `notebook_login()` in Colab

---

## Recommended Order & Execution Plan

### Exercise 1: Unit 4 -- Music Genre Classifier
**Notebook**: `unit_4/04_music_genre_classifier.ipynb`
**Difficulty**: Easiest -- good warm-up for the fine-tuning workflow
**Colab GPU time**: ~1-2 hours

**What you'll do**: Fine-tune an audio classification model on the GTZAN dataset (1000 tracks, 10 genres).

**Pass criteria**: >= 87% accuracy, model pushed to Hub

**Steps**:
1. Open notebook in Colab, connect to a T4 GPU runtime
2. Run all setup and data loading cells
3. Start with `ntu-spml/distilhubert` as the base model (already configured)
4. Train for 10 epochs and check accuracy
5. If below 87%, try these levers in order:
   - Switch to `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer -- often hits 90%+)
   - Lower learning rate to `3e-5` or `1e-5`
   - Increase epochs to 15-20
6. Once >= 87%, run the `push_to_hub` cell with the required kwargs
7. Verify the model appears on your HF profile

**Key gotcha**: The GTZAN dataset has one corrupted file. If you hit an error during preprocessing, add a try/except filter or use `trust_remote_code=True` (already set).

---

### Exercise 2: Unit 5 -- Speech Recognition (ASR)
**Notebook**: `unit_5/05_speech_recognition.ipynb`
**Difficulty**: Medium -- more moving parts (tokenizer, data collator, WER metric)
**Colab GPU time**: ~1-2 hours

**What you'll do**: Fine-tune Whisper-tiny on the MINDS-14 banking dataset for English ASR.

**Pass criteria**: Normalized WER < 0.37, model pushed to Hub

**Steps**:
1. Open notebook in Colab with T4 GPU
2. Run cells sequentially -- the data split (450 train / rest eval) and `num_proc=1` are already set correctly per the exercise requirements
3. Train for 500 steps (already configured)
4. Check `eval_wer` -- should comfortably clear 0.37 with these settings
5. If WER is too high:
   - Increase `max_steps` to 750-1000
   - Try `learning_rate=5e-6` for more gradual learning
6. Push to Hub with the required kwargs
7. Verify model on your HF profile

**Key gotchas**:
- WER must be reported as a decimal (0.37), NOT a percentage (37%) -- the notebook handles this correctly
- Must use `num_proc=1` in `.map()` -- already enforced in the notebook
- The data collator handles BOS token removal and padding -- don't modify it

---

### Exercise 3: Unit 6 -- Text-to-Speech (TTS)
**Notebook**: `unit_6/06_text_to_speech.ipynb`
**Difficulty**: Hardest -- longest training, speaker embeddings, and TTS-specific data collator
**Colab GPU time**: ~3-5 hours (longest exercise)

**What you'll do**: Fine-tune SpeechT5 on VoxPopuli Dutch for text-to-speech synthesis.

**Pass criteria**: No metric threshold -- just push to Hub tagged `text-to-speech`

**Steps**:
1. Open in Colab with T4 GPU
2. **Choose your language/dataset** -- Dutch VoxPopuli is pre-configured, but you can change it
3. Run the data filtering cell (removes clips > 10s which SpeechT5 can't handle)
4. Preprocessing is slow due to speaker embedding extraction (~30-60 min) -- this is normal
5. Train for 4000 steps with the configured hyperparameters
6. Listen to generated samples in the test cell -- quality won't be perfect but should be intelligible
7. Push to Hub with `tasks: "text-to-speech"` tag

**Key gotchas**:
- SpeechT5 tokenizer only supports English characters by default. For Dutch, some characters may need to be stripped or replaced -- add a text cleaning function if you see tokenization errors
- Speaker embedding computation is CPU-bound and slow. Be patient during preprocessing
- Colab free tier may disconnect during long training -- consider saving checkpoints every 1000 steps (already configured)
- **Important**: If you choose Dutch here, it feeds directly into Unit 7 (you can reuse this model)

**Strategic choice**: Pick Dutch (`nl`) because:
- The course provides a pre-trained Dutch SpeechT5 checkpoint as a fallback for Unit 7
- Good pre-existing translation models (Helsinki-NLP/opus-mt-en-nl)
- It sets up a smooth path for Exercise 4

---

### Exercise 4: Unit 7 -- Speech-to-Speech Translation Demo
**Notebook**: `unit_7/07_speech_to_speech_translation.ipynb`
**Difficulty**: Medium -- no training needed, but requires deploying a HF Space
**Colab GPU time**: None (runs on HF Spaces CPU tier)

**What you'll do**: Build a Gradio app that translates speech from any language into non-English speech.

**Pass criteria**: Public HF Space that outputs non-English audio, verified by the automated assessment

**Steps**:
1. Go to the [template Space](https://huggingface.co/spaces/course-demos/speech-to-speech-translation?duplicate=true) and click **Duplicate**
2. In your duplicated Space, edit `app.py`:
   - **translate() function**: Whisper translates to English. Add a second step using `Helsinki-NLP/opus-mt-en-nl` to translate English text -> Dutch (or your chosen language)
   - **synthesise() function**: Replace the English SpeechT5 with either:
     - Your fine-tuned model from Unit 6
     - MMS TTS: `facebook/mms-tts-nld` (Dutch, often better quality)
     - Pre-trained: `sanchit-gandhi/speecht5_tts_vox_nl`
3. Update `requirements.txt` if needed
4. Set Space visibility to **Public** (required for assessment)
5. Test by recording/uploading audio and verifying the output is in your target language
6. Submit your Space ID at the [assessment page](https://huggingface.co/spaces/huggingface-course/audio-course-u7-assessment)
7. Check your progress at the [progress tracker](https://huggingface.co/spaces/MariaK/Check-my-progress-Audio-Course)

**Key gotchas**:
- The Space must be **public** or the assessment bot can't reach it
- Free CPU tier is fine -- no GPU needed
- If using MMS TTS, you may need a specific transformers branch in requirements.txt (noted in the notebook)
- Test the Space end-to-end before submitting -- the assessment sends real audio and checks the response

---

## Summary & Timeline

| Order | Exercise | Est. Time | Difficulty | Metric Target |
|-------|----------|-----------|------------|---------------|
| 1st | Unit 4: Music Classifier | 1-2 hrs | Easy | >= 87% accuracy |
| 2nd | Unit 5: ASR | 1-2 hrs | Medium | WER < 0.37 |
| 3rd | Unit 6: TTS | 3-5 hrs | Hard | None (just push) |
| 4th | Unit 7: S2S Demo | 1-2 hrs | Medium | Non-English output |

**Total estimated active time**: ~6-11 hours across multiple Colab sessions

**Dependencies**:
- Exercises 1 & 2 are independent -- could be done in either order
- Exercise 3 (TTS) should be done before Exercise 4 (S2S) since you can reuse the TTS model
- Pick the same target language for Exercises 3 & 4 (Dutch recommended)

## Verification

After completing all exercises, check your progress at:
https://huggingface.co/spaces/MariaK/Check-my-progress-Audio-Course

You need green checkmarks on at least 3/4 exercises for the completion certificate, or 4/4 for honors.
