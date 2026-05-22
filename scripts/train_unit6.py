"""Unit 6 - SpeechT5 TTS fine-tuned on VoxPopuli Dutch. Local training run.

Adapted from unit_6/06_text_to_speech.ipynb:
  - loads auto-converted parquet (qmeeus/voxpopuli has no speaker_id column,
    so we cap to a subset instead of speaker-balancing)
  - decodes audio with soundfile/librosa (torchcodec/ffmpeg unavailable here)
  - uses processor(text=, audio_target=) per the official course (produces labels)
  - no metric threshold: train a capped run and push tagged text-to-speech
  - Windows-safe: __main__ guard + freeze_support()
"""
import io
import multiprocessing
import torch
import numpy as np
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)

TARGET_SR = 16000
SUBSET = 1200          # raw rows to consider (keeps CPU embedding step tractable)
MAX_STEPS = 800        # no metric bar; enough to demonstrate fine-tuning
VOX_GLOB = "hf://datasets/qmeeus/voxpopuli@refs/convert/parquet/nl/train/*.parquet"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

from speechbrain.inference.classifiers import EncoderClassifier
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"}
)

REPLACEMENTS = [("à","a"),("ç","c"),("è","e"),("ë","e"),
                ("í","i"),("ï","i"),("ö","o"),("ü","u")]


def clean_text(t):
    for s, d in REPLACEMENTS:
        t = t.replace(s, d)
    return t


def speaker_embedding(waveform):
    with torch.no_grad():
        emb = speaker_model.encode_batch(torch.tensor(waveform).unsqueeze(0))
        emb = torch.nn.functional.normalize(emb, dim=2).squeeze()
    return emb.numpy()


def prepare(example):
    a = example["audio"]
    arr, in_sr = sf.read(io.BytesIO(a["bytes"]))
    arr = arr.astype("float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if in_sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=in_sr, target_sr=TARGET_SR)
    dur = len(arr) / TARGET_SR
    proc = processor(text=clean_text(example["text"]), audio_target=arr,
                     sampling_rate=TARGET_SR)
    out = {
        "input_ids": proc["input_ids"],
        "labels": proc["labels"][0],
        "speaker_embeddings": speaker_embedding(arr),
        "input_length": dur,
    }
    return out


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    model: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]
        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100.0
        )
        del batch["decoder_attention_mask"]
        rf = self.model.config.reduction_factor
        if rf > 1:
            tl = torch.tensor([len(f["labels"]) for f in features])
            tl = tl.new([l - l % rf for l in tl])
            batch["labels"] = batch["labels"][:, : max(tl)]
        batch["speaker_embeddings"] = torch.tensor(
            np.array(speaker_features), dtype=torch.float32
        )
        return batch


def main():
    print(f"[cfg] speecht5 voxpopuli-nl subset={SUBSET} max_steps={MAX_STEPS}", flush=True)
    ds = load_dataset("parquet", data_files=VOX_GLOB, split="train")
    ds = ds.select(range(min(SUBSET, len(ds))))
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"[data] raw subset {len(ds)} rows; extracting features + speaker embeddings (CPU)...", flush=True)

    ds = ds.map(prepare, remove_columns=ds.column_names, num_proc=1)
    ds = ds.filter(lambda x: x < 10.0, input_columns=["input_length"])
    ds = ds.filter(lambda x: len(x) < 200, input_columns=["input_ids"])
    ds = ds.remove_columns(["input_length"])
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(f"[data] usable: {ds}", flush=True)

    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.config.use_cache = False

    args = Seq2SeqTrainingArguments(
        output_dir="runs/speecht5-finetuned-voxpopuli-nl",
        per_device_train_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=1e-5, warmup_steps=100, max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, fp16=True,
        eval_strategy="steps", per_device_eval_batch_size=2,
        save_steps=500, eval_steps=500, logging_steps=25,
        report_to="none", load_best_model_at_end=True,
        greater_is_better=False, label_names=["labels"],
        push_to_hub=False, dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        args=args, model=model,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=TTSDataCollatorWithPadding(processor=processor, model=model),
        processing_class=processor,
    )
    trainer.train()
    print("[done] training complete; pushing to Hub tagged text-to-speech", flush=True)
    trainer.push_to_hub(
        dataset_tags="qmeeus/voxpopuli", dataset="VoxPopuli",
        dataset_args="config: nl, split: train",
        finetuned_from="microsoft/speecht5_tts", tasks="text-to-speech",
    )
    print("[push] done", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
