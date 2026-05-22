# TODO

## ⏳ Submit Unit 7 to the assessment (for honors / 4-of-4)

Everything is built and verified — this is the only remaining step, and it's
**blocked on HuggingFace's side**, not yours.

- [ ] **Submit the Space for assessment** once the assessor is back up.
  1. Open the [Unit 7 assessment Space](https://huggingface.co/spaces/huggingface-course/audio-course-u7-assessment).
  2. Enter your Space ID: `VoicesColeby/speech-to-speech-translation`
  3. It should return **"Congratulations ... passed the assessment!"**
- [ ] **Confirm the green tick** on the [progress tracker](https://huggingface.co/spaces/MariaK/Check-my-progress-Audio-Course).

### Why it's blocked (2026-05-22)
Both the assessment Space and the progress tracker are in `RUNTIME_ERROR` on
HF's side — the assessor can't load `facebook/mms-lid-126`
(`No space left on device`). Retry the link periodically; it should work after
HF reboots/fixes that Space.

- Tracking issue (opened from your account): https://huggingface.co/spaces/huggingface-course/audio-course-u7-assessment/discussions/23

### Proof it will pass (already verified locally)
Ran the assessor's exact logic — its own `test_short.wav` → your Space's
`/predict` → `facebook/mms-lid-126` classified the output as `nld` (Dutch),
score **1.0** (passes the non-English ≥0.5 check).

---

## ✅ Done
- Unit 4 — `VoicesColeby/distilhubert-finetuned-gtzan` (0.89 acc)
- Unit 5 — `VoicesColeby/whisper-tiny-minds14-en-us` (WER 0.321)
- Unit 6 — `VoicesColeby/speecht5-finetuned-voxpopuli-nl` (tagged `text-to-speech`)
- Unit 7 — `spaces/VoicesColeby/speech-to-speech-translation` (public, verified) — *pending submission above*
- PR: https://github.com/ColebyPearson/HFaudio/pull/1

You already meet the **completion certificate (3/4)** via Units 4–6. The step
above is only needed for **honors (4/4)**.
