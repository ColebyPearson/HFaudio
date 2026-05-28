"""
Fixed copy of `huggingface-course/audio-course-u7-assessment`.

Same verification logic as the official Space, but:
  1. Drops the dependency on the now-404 private dataset
     `huggingface-course/audio-course-u7-hands-on` (this was the root
     cause of the official Space's RUNTIME_ERROR — the clone fails at
     startup so the Space never reaches `demo.launch()`).
  2. Tracks already-passed usernames in-process only (in this fork the
     "already passed" check is effectively disabled, which is what we
     want for verifying a single submission anyway).
  3. Pins Python 3.12 + gradio + transformers + numpy so we don't get
     swept along by the python-3.13 / transformers-HEAD breaking changes.

Verification semantics are identical to the official:
  - Submit a repo_id of an S2S Gradio Space.
  - We POST `test_short.wav` to its `/predict` endpoint.
  - Read the returned audio, run `facebook/mms-lid-126` on it.
  - Pass if  top language is not 'eng'  AND  top score >= 0.5.

Author: Coleby Pearson (VoicesColeby) — patch posted in discussion #23.
"""
from __future__ import annotations

import os
from typing import List

import gradio as gr
import soundfile as sf
import torch
from gradio_client import Client, handle_file
from transformers import pipeline

HF_TOKEN = os.environ.get("HF_TOKEN")
THRESHOLD = 0.5
PASS_MESSAGE = "Congratulations USER! Your demo passed the assessment!"

# In-process roster of usernames that have passed in this Space session.
# (The original used a private dataset we no longer have access to.)
_PASSED: List[str] = []


# Load the language-ID checkpoint once at startup.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("audio-classification", model="facebook/mms-lid-126", device=DEVICE)


TITLE = "🤗 Audio Transformers Course: Unit 7 Assessment (community fork)"
DESCRIPTION = """
**This is an unofficial community fork** of [`huggingface-course/audio-course-u7-assessment`](https://huggingface.co/spaces/huggingface-course/audio-course-u7-assessment) — the official Space has been in `RUNTIME_ERROR` since at least 2026-05-22 because it clones a now-404 private dataset at startup. This fork runs the same verification logic so you can confirm your S2S model meets the Unit 7 bar (non-English audio output, mms-lid-126 score ≥ 0.5).

**Important caveat:** because this fork does **not** have write access to the official roster, passing here does **not** auto-update the [course progress tracker](https://huggingface.co/spaces/MariaK/Check-my-progress-Audio-Course). It is proof of completion only — for the certificate you'll need to wait for HF to restore the official Space.

Submit the `username/space-name` of your S2S Gradio demo below. The submission flow is otherwise identical to the official assessor.
"""


def verify_demo(repo_id):
    if "/" not in repo_id:
        raise gr.Error(f"Ensure you pass a valid repo id to the assessor, got `{repo_id}`")

    split = repo_id.split("/")
    user_name = split[-2]
    if len(split) > 2:
        repo_id = "/".join(split[-2:])

    if user_name in _PASSED:
        raise gr.Error(
            f"Username {user_name} has already passed this assessor session "
            "(local check only; this fork does not write to the official roster)."
        )

    try:
        client = Client(repo_id, hf_token=HF_TOKEN)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(
            "Error loading Space. Check that your Space has been built and is running, "
            "and that it exposes /predict taking an audio file and returning an audio file. "
            f"Error: {e}"
        )

    try:
        # Modern gradio_client requires file inputs wrapped via handle_file()
        # (older versions accepted a bare path string, which is what the
        # original assessor used and is now rejected by the pydantic-validated
        # gradio data model with "'meta' field must be explicitly provided").
        audio_file = client.predict(handle_file("test_short.wav"), api_name="/predict")
    except Exception as e:  # noqa: BLE001
        raise gr.Error(
            f"Error querying your Space — verify it accepts an audio input and returns an audio output at /predict: {e}"
        )

    audio, sampling_rate = sf.read(audio_file)
    language_prediction = pipe({"array": audio, "sampling_rate": sampling_rate})

    label_outputs = {pred["label"]: pred["score"] for pred in language_prediction}
    top = language_prediction[0]

    if top["score"] < THRESHOLD:
        raise gr.Error(
            f"Model made random predictions — predicted {top['label']} with probability {top['score']:.2f}"
        )
    if top["label"] == "eng":
        raise gr.Error(
            "Model generated English audio — ensure the model is set to generate audio in a non-English language (e.g. Dutch, French, German, Spanish)."
        )

    _PASSED.append(user_name)
    message = PASS_MESSAGE.replace("USER", user_name)
    return message, "test_short.wav", (sampling_rate, audio), label_outputs


demo = gr.Interface(
    fn=verify_demo,
    inputs=gr.Textbox(placeholder="username/speech-to-speech-translation", label="Repo id or URL of your demo"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Source Speech", type="filepath"),
        gr.Audio(label="Generated Speech", type="numpy"),
        gr.Label(label="Language prediction"),
    ],
    title=TITLE,
    description=DESCRIPTION,
    allow_flagging="never",
)
demo.launch()
