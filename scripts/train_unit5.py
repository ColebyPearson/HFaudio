"""Unit 5 - Whisper ASR on MINDS-14 en-US. Local training run.

Adapted from unit_5/05_speech_recognition.ipynb:
  - loads the auto-converted parquet branch (datasets 4.x dropped script support)
  - decodes audio with soundfile/librosa (torchcodec/ffmpeg unavailable here)
  - first 450 train / rest eval; num_proc=1 (per exercise rules)
  - train -> verify normalized WER < 0.37 -> push
  - Windows-safe: __main__ guard + freeze_support()
"""
import io
import multiprocessing
import torch
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate

MODEL_ID = "openai/whisper-tiny"
TARGET_SR = 16000

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="english", task="transcribe")
wer_metric = evaluate.load("wer")


def _decode(a):
    arr, in_sr = sf.read(io.BytesIO(a["bytes"]))
    arr = arr.astype("float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if in_sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=in_sr, target_sr=TARGET_SR)
    return arr


def prepare_dataset(batch):
    arr = _decode(batch["audio"])
    batch["input_features"] = feature_extractor(arr, sampling_rate=TARGET_SR).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_ortho = wer_metric.compute(predictions=pred_str, references=label_str)
    pred_norm = [tokenizer._normalize(t) for t in pred_str]
    label_norm = [tokenizer._normalize(t) for t in label_str]
    keep = [i for i in range(len(label_norm)) if len(label_norm[i]) > 0]
    wer = wer_metric.compute(
        predictions=[pred_norm[i] for i in keep],
        references=[label_norm[i] for i in keep],
    )
    return {"wer": wer, "wer_ortho": wer_ortho}


def main():
    print("[cfg] whisper-tiny minds14 en-US", flush=True)
    minds = load_dataset(
        "parquet",
        data_files="hf://datasets/PolyAI/minds14@refs/convert/parquet/en-US/train/*.parquet",
        split="train",
    )
    minds = minds.cast_column("audio", Audio(decode=False))
    train = minds.select(range(450))
    ev = minds.select(range(450, len(minds)))
    print(f"[data] train={len(train)} eval={len(ev)}", flush=True)

    train = train.map(prepare_dataset, remove_columns=train.column_names, num_proc=1)
    ev = ev.map(prepare_dataset, remove_columns=ev.column_names, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.use_cache = False
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    args = Seq2SeqTrainingArguments(
        output_dir="runs/whisper-tiny-minds14-en-us",
        per_device_train_batch_size=16, gradient_accumulation_steps=1,
        learning_rate=1e-5, lr_scheduler_type="constant_with_warmup",
        warmup_steps=50, max_steps=500,
        gradient_checkpointing=False, fp16=True, fp16_full_eval=False,
        eval_strategy="steps", per_device_eval_batch_size=8,
        predict_with_generate=True, generation_max_length=225,
        save_steps=100, eval_steps=100, logging_steps=25,
        report_to="none", load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        push_to_hub=False, dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        args=args, model=model, train_dataset=train, eval_dataset=ev,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics, processing_class=processor,
    )
    trainer.train()
    res = trainer.evaluate()
    wer = res["eval_wer"]
    print(f"\n[RESULT] wer={wer:.4f} wer_ortho={res['eval_wer_ortho']:.4f} "
          f"target<0.37 {'PASS' if wer < 0.37 else 'FAIL'}", flush=True)

    if wer < 0.37:
        print("[push] pushing to Hub...", flush=True)
        trainer.push_to_hub(
            dataset_tags="PolyAI/minds14",
            finetuned_from="openai/whisper-tiny",
            tasks="automatic-speech-recognition",
        )
        print("[push] done", flush=True)
    else:
        print("[push] skipped (above WER threshold)", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
