# ASR Demo for Project Aria

A Streamlit-based live Automatic Speech Recognition (ASR) demo for Project Aria audio streaming.  
The project supports multiple ASR backends, real-time transcription display, live event logging, and automatic evaluation with quality and latency metrics.

## Features

- Live audio streaming from **Project Aria**
- Streamlit UI for controlling the demo
- Support for multiple ASR models:
  - Zipformer kroko
  - Vosk
  - Whisper
  - Wav2Vec2
  - Qwen3
  - Whisper + VAD
  - Qwen3 + VAD
  - SimulStreaming
- Real-time transcript visualization
- Ground-truth comparison with word-level highlighting
- Session logging and persistence
- Automatic metric computation:
  - WER
  - CER
  - SER
  - PER (optional)
  - BERTScore Precision / Recall / F1 (optional)
- Latency tracking:
  - first output latency
  - first partial/final latency
  - RTF
  - mean inter-output delay

---

## Project Structure

```text
asr_demo/
├── app.py                      # Streamlit frontend
├── streamlit_backend.py        # Backend process for live streaming and decoding
├── live_output.log             # Live backend log
├── live_backend.pid            # Backend PID file
├── latest_session.json         # Latest finished session payload
├── logo.png                    # Optional UI logo
├── sessions/
│   ├── hypotheses_sessions.csv # Stored hypotheses per session
│   └── metrics.csv             # Computed evaluation metrics
├── core/
│   ├── streaming_controller.py
│   ├── audio_preprocessing.py
│   ├── base_observer.py
│   └── types.py
├── metrics/
│   └── metrics.py
└── models/
    ├── zipformer_model.py
    ├── vosk_model.py
    ├── whisper_model.py
    ├── wav2vec2_model.py
    ├── qwen3_model.py
    ├── whispervad_model.py
    ├── qwen3vad_model.py
    ├── simulstreaming_model.py
    └── ...
```

> Adjust file names if your frontend file is not `app.py`.

---

## How It Works

### Frontend (`app.py`)
The Streamlit app provides:

- model selection
- interface selection (`usb` / `wifi`)
- gain and denoise configuration
- test metadata input (`test_id`, `person_id`, `gender`)
- start/stop control for the backend
- metrics computation after a session ends

It also:

- reads the live log file
- reconstructs live partial/final transcription
- compares hypothesis vs. ground truth
- renders the latest metrics from `sessions/metrics.csv`

### Backend (`streamlit_backend.py`)
The backend:

- reads runtime configuration from environment variables
- starts the Project Aria streaming controller
- loads the selected ASR model
- logs partial and final hypotheses
- tracks latency and output timing
- writes the final session payload to `latest_session.json`
- stores session hypotheses to CSV

### Metrics (`metrics/metrics.py`)
The metrics module provides:

- text normalization
- WER / CER / SER computation
- PER using `phonemizer` (optional)
- BERTScore using `bert-score` (optional)
- latency and inter-output statistics
- CSV export for later analysis

---

## Requirements

### System Requirements

This project is designed for Linux and depends on:

- **Python 3.10+**
- **Project Aria SDK / projectaria_tools**
- access to audio streaming from Aria device
- ASR model files stored locally under paths such as:
  - `/home/iat/projectaria_client_sdk_samples/ASR_models/...`

Some optional features also require:

- `espeak-ng` for PER via `phonemizer`
- PyTorch-compatible environment for transformer-based models and BERTScore

---

## Installation

### 1. Clone your repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install system dependency for phonemizer (optional, for PER)

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng
```

### 5. Make sure your local ASR models exist

From the code, the project expects model folders such as:

- `ASR_models/Zipformer/de/kroko_64l`
- `ASR_models/vosk-model-small-de-0.15`
- `ASR_models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`
- `ASR_models/sherpa-onnx-whisper-small`
- `ASR_models/SimulStreaming`

Update the hardcoded paths in `app.py` and `streamlit_backend.py` if your setup differs.

---

## Running the App

Start the Streamlit frontend:

```bash
streamlit run app.py
```

Then in the UI:

1. choose a model
2. choose interface (`usb` or `wifi`)
3. set gain / denoise
4. enter optional metadata
5. click **Start**
6. speak / stream audio
7. click **Stop**
8. click **Compute Metrics**

---

## Output Files

### `live_output.log`
Contains runtime events such as:

- `[INFO]`
- `[PARTIAL]`
- `[FINAL]`
- `[LIVE_HYP]`
- `[ERROR]`

### `latest_session.json`
Stores the most recent finished session with:

- session metadata
- hypothesis text
- ground truth text
- latency metrics
- output timing metrics

### `sessions/hypotheses_sessions.csv`
Stores one row per finished session.

### `sessions/metrics.csv`
Stores computed evaluation metrics and latency values.

---

## Supported Models in Current UI

The frontend currently exposes these model keys:

- `zipformer`
- `vosk`
- `whisper`
- `wav2vec2`
- `qwen3`
- `whispervad`
- `simulstreaming`
- `qwen3vad`

The backend also contains placeholders for additional models such as `nemo` and `parakeet`, but these are not currently listed in the Streamlit selectbox.

---

## Notes About Optional Dependencies

### PER
PER is only available if `phonemizer` is installed successfully.

If not installed, PER is stored as empty / `None`.

### BERTScore
BERTScore is only available if `bert-score` is installed successfully.

If not installed, BERTScore metrics are stored as empty / `None`.

---

## Known Configuration Notes

- `PROJECT_ROOT` is hardcoded to:

```python
/home/iat/projectaria_client_sdk_samples
```

If you move the project, update:

- `PROJECT_ROOT` in `app.py`
- `PROJECT_ROOT` in `streamlit_backend.py`

- The frontend expects:
  - `streamlit_backend.py`
  - `metrics.metrics`
  - `logo.png` (optional)

- The backend depends on your internal modules under `core/`, `models/`, and `metrics/`.

---

## Example Workflow

```bash
streamlit run app.py
```

Inside the app:

- select `qwen3vad`
- choose `usb`
- set gain to `8.0`
- enable or disable denoise
- click **Start**
- stop after speaking
- click **Compute Metrics**

You will then see:

- live transcription
- ground truth comparison
- WER / CER / PER / BERTScore
- latency metrics like RTF and first output delay

---

## Troubleshooting

### Backend does not start
Check that:

- `streamlit_backend.py` exists
- the selected model files exist
- all required imports under `core/`, `models/`, and `metrics/` are available

### No live transcription appears
Check:

- Aria stream is actually running
- the correct interface is selected
- `live_output.log` is being written
- model path is valid

### PER is empty
Install:

```bash
pip install phonemizer
sudo apt-get install -y espeak-ng
```

### BERTScore is empty
Install:

```bash
pip install bert-score
```

### `projectaria_tools` or `aria.sdk` import fails
Make sure the Project Aria SDK and Python bindings are installed correctly in the current environment.

---

## Future Improvements

Possible next steps for the project:

- remove hardcoded paths via `.env` or config file
- add batch evaluation dashboards
- add plots for model comparison
- add export of latency graphs
- support more streaming-capable German ASR models
- add automatic session browser in Streamlit

---

## License

Add your preferred license here, for example MIT.
