---
title: Audio Course U7 Assessment (Community Fork)
emoji: 🎧
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.12"
pinned: false
license: apache-2.0
duplicated_from: huggingface-course/audio-course-u7-assessment
tags:
- audio-course
- speech-to-speech
- assessment-fix
---

# Audio Course U7 Assessment — Community Fork

Unofficial fork of the official Unit 7 assessment Space.

**Why this exists**: the official Space at
[`huggingface-course/audio-course-u7-assessment`](https://huggingface.co/spaces/huggingface-course/audio-course-u7-assessment)
has been stuck in `RUNTIME_ERROR` since 2026-05-22. The root cause is at
the top of the original `app.py`:

```python
usernames_repo = Repository(
    local_dir="usernames",
    clone_from="https://huggingface.co/datasets/huggingface-course/audio-course-u7-hands-on",
    use_auth_token=HF_TOKEN,
)
usernames_repo.git_pull()
```

That dataset now returns **404** (deleted or perms-revoked), so the
Space never reaches `demo.launch()`.

## What this fork changes

1. **Drops the private-dataset clone.** Passed-usernames are tracked
   in-memory for this Space session only (no auto-update of the
   official progress tracker — see caveat below).
2. **Pins versions** to escape the Python-3.13 / `transformers`-HEAD
   ABI churn that's also affecting other Gradio Spaces this week:
   - `python_version: "3.12"`
   - `sdk_version: 5.49.1` (gradio)
   - `transformers>=4.45,<4.60`
   - `numpy<2`
3. **Keeps the assessment logic byte-identical** otherwise — same
   `/predict` call, same `facebook/mms-lid-126` LID check, same
   `score ≥ 0.5` + `label != "eng"` pass criteria.

## Caveat

Because this fork does **not** have write access to the official roster
dataset, passing this assessor does **not** auto-update
[`MariaK/Check-my-progress-Audio-Course`](https://huggingface.co/spaces/MariaK/Check-my-progress-Audio-Course).

This is **proof-of-completion only** — useful for:

- Self-verification that your S2S Space meets the bar.
- Attaching a screenshot to the existing discussion thread
  (`huggingface-course/audio-course-u7-assessment/discussions/23`)
  so HF can credit the certificate manually once the official Space is
  fixed.

## Usage

Enter your S2S Space id (e.g. `VoicesColeby/speech-to-speech-translation`)
in the textbox. The Space will:

1. Connect to your Space via `gradio_client.Client(repo_id)`.
2. POST `test_short.wav` to your `/predict` endpoint.
3. Read the returned audio.
4. Run `facebook/mms-lid-126` on it.
5. Pass if top language `!= "eng"` and top score `>= 0.5`.
