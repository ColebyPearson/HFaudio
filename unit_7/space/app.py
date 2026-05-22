import torch
import numpy as np
import gradio as gr
from transformers import pipeline, VitsModel, VitsTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Translation: any language speech -> English text (Whisper) -> Dutch text (opus-mt)
asr_pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
translator = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-en-nl", device=device
)

# TTS: Dutch (MMS) — pretrained, supports Dutch out of the box
tts_model = VitsModel.from_pretrained("facebook/mms-tts-nld")
tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-nld")


def translate(audio):
    outputs = asr_pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    translated = translator(outputs["text"], max_length=512)
    return translated[0]["translation_text"]


def synthesise(text):
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = tts_model(inputs["input_ids"])
    return outputs.waveform[0].numpy()


target_dtype = np.int16
max_range = np.iinfo(target_dtype).max


# NOTE: the Unit 7 assessment Space calls this with api_name="/predict",
# so the function MUST be named `predict` for gradio to expose "/predict".
def predict(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech * max_range).astype(np.int16)
    return 16000, synthesised_speech


title = "Speech-to-Speech Translation (→ Dutch)"
description = """
Cascaded speech-to-speech translation: Whisper translates input speech (any
language) to English text, opus-mt translates it to Dutch, and MMS-TTS
synthesises Dutch speech.
"""

mic_translate = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title=title,
    description=description,
)
file_translate = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title=title,
    description=description,
)

demo = gr.Blocks()
with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

if __name__ == "__main__":
    demo.launch()
