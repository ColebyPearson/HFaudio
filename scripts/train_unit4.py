"""Unit 4 - Music Genre Classifier (GTZAN). Local training run.

Adapted from unit_4/04_music_genre_classifier.ipynb:
  - no notebook_login (already authenticated via huggingface-cli)
  - loads the auto-converted parquet branch (datasets 4.x dropped script support)
  - decodes audio with soundfile/librosa (torchcodec/ffmpeg unavailable here)
  - train -> verify >=0.87 -> push (instead of push during training)
  - Windows-safe: all work under __main__ guard + freeze_support()
"""
import io
import sys
import multiprocessing
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "ntu-spml/distilhubert"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
model_name = MODEL_ID.split("/")[-1]

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID, do_normalize=True, return_attention_mask=True
)
SR = feature_extractor.sampling_rate
MAX_DURATION = 30.0


def _decode(a):
    arr, in_sr = sf.read(io.BytesIO(a["bytes"]))
    arr = arr.astype("float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if in_sr != SR:
        arr = librosa.resample(arr, orig_sr=in_sr, target_sr=SR)
    return arr


def preprocess(examples):
    arrays = [_decode(a) for a in examples["audio"]]
    return feature_extractor(
        arrays, sampling_rate=SR,
        max_length=int(SR * MAX_DURATION), truncation=True,
        return_attention_mask=True,
    )


def main():
    print(f"[cfg] model={MODEL_ID} epochs={EPOCHS} lr={LR} sr={SR}", flush=True)

    gtzan = load_dataset("marsyas/gtzan", revision="refs/convert/parquet")
    before = gtzan["train"].num_rows
    gtzan["train"] = gtzan["train"].filter(
        lambda f: "jazz.00054" not in f, input_columns=["file"]
    )
    print(f"[data] filtered {before} -> {gtzan['train'].num_rows} rows", flush=True)

    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
    labels = gtzan["train"].features["genre"].names
    label2id = {l: str(i) for i, l in enumerate(labels)}
    id2label = {str(i): l for i, l in enumerate(labels)}

    gtzan = gtzan.cast_column("audio", Audio(decode=False))
    encoded = gtzan.map(
        preprocess, remove_columns=["audio", "file"],
        batched=True, batch_size=100, num_proc=1,
        load_from_cache_file=False,
    ).rename_column("genre", "label")
    print(f"[data] encoded: {encoded}", flush=True)

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), label2id=label2id, id2label=id2label,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return accuracy.compute(predictions=preds, references=p.label_ids)

    args = TrainingArguments(
        output_dir=f"runs/{model_name}-finetuned-gtzan",
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=LR, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, num_train_epochs=EPOCHS,
        warmup_ratio=0.1, logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        fp16=True, push_to_hub=False, report_to="none",
        save_total_limit=2, dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=encoded["train"], eval_dataset=encoded["test"],
        processing_class=feature_extractor, compute_metrics=compute_metrics,
    )
    trainer.train()
    res = trainer.evaluate()
    acc = res["eval_accuracy"]
    print(f"\n[RESULT] eval_accuracy={acc:.4f} target=0.87 "
          f"{'PASS' if acc >= 0.87 else 'FAIL'}", flush=True)

    if acc >= 0.87:
        print("[push] pushing to Hub...", flush=True)
        trainer.push_to_hub(
            dataset_tags="marsyas/gtzan", dataset="GTZAN",
            model_name=f"{model_name}-finetuned-gtzan",
            finetuned_from=MODEL_ID, tasks="audio-classification",
        )
        print("[push] done", flush=True)
    else:
        print("[push] skipped (below threshold)", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
