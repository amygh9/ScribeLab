"""
This is the Frontend of the demonstrator.

It provides a user interface for interacting with the ASR system,
allowing users to start/stop sessions, view transcriptions, and
monitor performance metrics.

"""

import os
import sys
import csv
import json
import time
import signal
import subprocess
import html
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Paths used by the frontend and backend
PROJECT_ROOT = Path("/home/iat/projectaria_client_sdk_samples")
ASR_DEMO_DIR = PROJECT_ROOT / "asr_demo"
BACKEND_PATH = ASR_DEMO_DIR / "streamlit_backend.py"
LOG_PATH = ASR_DEMO_DIR / "live_output.log"
PID_PATH = ASR_DEMO_DIR / "live_backend.pid"
LATEST_SESSION_PATH = ASR_DEMO_DIR / "latest_session.json"
SESSIONS_DIR = ASR_DEMO_DIR / "sessions"
METRICS_CSV_PATH = SESSIONS_DIR / "metrics.csv"
LOGO_PATH = ASR_DEMO_DIR / "images" / "logo.png"

from metrics.metrics import (
    MetricsStore,
    LatencyResult,
    OutputLatencyResult,
)
#Ground truth text
GROUND_TRUTH_TEXT = """Die Sterilisation ist ein validiertes Verfahren zur Sicherstellung der Sterilität.
Dabei werden Mikroorganismen in einem kontrollierten Prozess inaktiviert.

Vor dem Sterilisationszyklus erfolgt die Reinigung mit einem geeigneten Reinigungsmittel.
Die Reinigung kann manuell oder in einem Reinigungs Desinfektionsgerät durchgeführt werden.
Anschließend erfolgt die chemische Desinfektion.

Die mikrobielle Kontamination wird als Bioburden beschrieben.
Bei der Belastungsprüfung werden koloniebildende Einheiten analysiert.

Der Sterilisationsprozess läuft in einer Kammer mit definierter Referenztemperatur.
Die Haltezeit, die Aufbaudauer und die Abbaudauer bestimmen die Zykluszeit.
Parameter wie Luftfeuchte werden während des Prozesses überwacht.

Bei der Verwendung von Filtern wird eine Filterintegritätsprüfung durchgeführt.

In strahlenbasierten Verfahren wird die Dosimetrie eingesetzt.
Dabei wird die absorbierte Strahlendosis überwacht, um die Sterilitätssicherheit zu gewährleisten.
Dies erfolgt mit Dosimeter und Dosimetriesystem.

Ein aseptischer Arbeitsbereich dient der Vermeidung extrinsischer und intrinsischer Kontamination.

"""
# page title 
st.set_page_config(page_title="ASR Demo", layout="wide")
# Apply custom dark theme styling 
st.markdown(
    """
    <style>
    .stApp { background: #000; color: white; }
    html, body, [class*="css"] { color: white; }

    .block-container {
        padding-top: 1rem;
    }

    header[data-testid="stHeader"] {
        background: #000000 !important;
    }

    div[data-testid="stToolbar"] {
        background: #000000 !important;
    }

    .stAppHeader {
        background: #000000 !important;
    }

    section[data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #222;
    }
    /* Sidebar title: "Controls" */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #cfcfcf !important;
    }

    

    /* Lighter gray field names in sidebar */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #cfcfcf !important;
        font-weight: 600 !important;
    }
    /* Sidebar expander arrow / chevron */
    section[data-testid="stSidebar"] svg {
        color: #cfcfcf !important;
        fill: #cfcfcf !important;
    }

    .panel {
        background: #111111;
        border: 1px solid #252525;
        border-radius: 18px;
        padding: 1rem;
    }

    .panel-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: #ffffff;
    }

    .transcript-box, .gt-box, .event-box {
        background: #0b0b0b;
        border: 1px solid #2b2b2b;
        border-radius: 14px;
        padding: 1rem;
        white-space: pre-wrap;
        color: #f5f5f5;
        line-height: 1.7;
        word-wrap: break-word;
    }

    .transcript-box { min-height: 240px; }
    .gt-box { min-height: 240px; }

    .event-box {
        min-height: 260px;
        font-family: monospace;
        font-size: 0.9rem;
        overflow-y: auto;
    }

    .status-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.86rem;
        margin-right: 0.5rem;
        margin-bottom: 0.35rem;
        border: 1px solid #333;
    }

    .status-running {
        background: rgba(0, 180, 90, 0.16);
        color: #88f0b2;
        border-color: rgba(0, 180, 90, 0.35);
    }

    .status-stopped {
        background: rgba(255, 110, 110, 0.12);
        color: #ffb0b0;
        border-color: rgba(255, 110, 110, 0.30);
    }

    .info-chip {
        background: #131313;
        color: #d7d7d7;
    }

    .metric-card {
        background: #111111;
        border: 1px solid #252525;
        border-radius: 16px;
        padding: 0.9rem;
        min-height: 100px;
    }

    .metric-label {
        color: #bbbbbb;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: white;
        font-weight: 800;
        font-size: 1.5rem;
    }

    .metric-help {
        color: #888888;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }

    .stButton button {
        border-radius: 12px !important;
        border: 1px solid #333 !important;
        background: #171717 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stButton button:hover {
        border-color: #555 !important;
        background: #202020 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #121212 !important;
        color: #ffffff !important;
        border-color: #333 !important;
    }

    .stTextInput input {
        background-color: #121212 !important;
        color: #ffffff !important;
    }

    .word-correct {
        color: #f5f5f5;
    }

    .word-wrong {
        color: #ff4d4d;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Initialize session state variables on first startup

def init_state():
    defaults = {
        "model_name": "zipformer",
        "interface": "usb",
        "gain": 8.0,
        "denoise": False,
        "test_id": "",
        "person_id": "",
        "gender": "not_specified",
        "stable_text": "",
        "last_partial": "",
        "live_hyp": "",
        "all_lines": [],
        "last_session_data": None,
        "metrics_store": MetricsStore(out_dir=str(SESSIONS_DIR), per_language="de"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Format metric values for display

def fmt_metric(x: Optional[float], digits: int = 4) -> str:
    if x is None or x == "":
        return "n/a"
    return f"{float(x):.{digits}f}"


def fmt_seconds(x: Optional[float], digits: int = 3) -> str:
    if x is None or x == "":
        return "n/a"
    return f"{float(x):.{digits}f} s"


def render_metric_card(title: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_pid():
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text().strip())
    except Exception:
        return None


def write_pid(pid: int):
    PID_PATH.write_text(str(pid), encoding="utf-8")


def clear_pid():
    if PID_PATH.exists():
        PID_PATH.unlink()


def is_running() -> bool:
    pid = read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        clear_pid()
        return False

# Start the ASR backend process with the selected settings
def start_backend():
    if is_running():
        return

    ASR_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    if LATEST_SESSION_PATH.exists():
        LATEST_SESSION_PATH.unlink()

    env = os.environ.copy()
    env["ASR_MODEL"] = st.session_state.model_name
    env["ASR_INTERFACE"] = st.session_state.interface
    env["ASR_GAIN"] = str(st.session_state.gain)
    env["ASR_DENOISE"] = "true" if st.session_state.denoise else "false"
    env["ASR_TEST_ID"] = st.session_state.test_id
    env["ASR_PERSON_ID"] = st.session_state.person_id
    env["ASR_GENDER"] = st.session_state.gender

    proc = subprocess.Popen(
        [sys.executable, str(BACKEND_PATH)],
        cwd=str(ASR_DEMO_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    write_pid(proc.pid)

    st.session_state.stable_text = ""
    st.session_state.last_partial = ""
    st.session_state.live_hyp = ""
    st.session_state.all_lines = []
    st.session_state.last_session_data = None

# Stop the backend and load the latest session results
def stop_backend():
    pid = read_pid()
    if pid is None:
        return

    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        pass

    time.sleep(0.6)
    clear_pid()
    process_log()
    load_latest_session()


def read_log_lines():
    if not LOG_PATH.exists():
        return []
    try:
        return LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def process_log():
    lines = read_log_lines()
    st.session_state.all_lines = lines[-200:]

    stable_text = ""
    last_partial = ""
    live_hyp = ""

    for line in lines:
        if line.startswith("[FINAL] "):
            txt = line[len("[FINAL] "):].strip()
            if txt:
                stable_text = (stable_text + " " + txt).strip() if stable_text else txt
            last_partial = ""

        elif line.startswith("[PARTIAL] "):
            last_partial = line[len("[PARTIAL] "):].strip()

        elif line.startswith("[LIVE_HYP] "):
            live_hyp = line[len("[LIVE_HYP] "):].strip()

    st.session_state.stable_text = stable_text
    st.session_state.last_partial = last_partial
    st.session_state.live_hyp = live_hyp


def load_latest_session():
    if not LATEST_SESSION_PATH.exists():
        return
    try:
        with open(LATEST_SESSION_PATH, "r", encoding="utf-8") as f:
            st.session_state.last_session_data = json.load(f)
    except Exception:
        st.session_state.last_session_data = None

# Compute evaluation metrics after a recording session
def compute_metrics():
    if is_running():
        st.warning("Stop the stream first.")
        return

    data = st.session_state.last_session_data
    if not data:
        st.warning("No finished backend session found.")
        return

    hyp_text = (data.get("hyp_text") or "").strip()
    if not hyp_text:
        st.warning("Hypothesis is empty.")
        return

    latency = LatencyResult(**(data.get("latency") or {}))
    output_latency = OutputLatencyResult(**(data.get("output_latency") or {}))

    st.session_state.metrics_store.append(
        session_id=data["session_id"],
        model=data["model"],
        test_id=data["test_id"],
        gt_text=data["ground_truth_text"],
        hyp_text=hyp_text,
        latency=latency,
        output_latency=output_latency,
        meta=data.get("meta") or {},
    )

    st.rerun()


def load_last_metrics_row_for_session(session_id: str):
    if not METRICS_CSV_PATH.exists():
        return None

    last_match = None
    try:
        with open(METRICS_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("session_id") == session_id:
                    last_match = row
    except Exception:
        return None

    return last_match


def tokenize_words(text: str):
    return text.split()


def render_word_diff_html(reference_text: str, candidate_text: str) -> str:
    ref_words = tokenize_words(reference_text)
    cand_words = tokenize_words(candidate_text)

    matcher = SequenceMatcher(None, ref_words, cand_words)
    output_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for word in cand_words[j1:j2]:
                output_parts.append(f'<span class="word-correct">{html.escape(word)}</span>')
        elif tag in {"replace", "insert"}:
            for word in cand_words[j1:j2]:
                output_parts.append(f'<span class="word-wrong">{html.escape(word)}</span>')

    return " ".join(output_parts)


def render_ground_truth_diff_html(reference_text: str, candidate_text: str) -> str:
    ref_words = tokenize_words(reference_text)
    cand_words = tokenize_words(candidate_text)

    matcher = SequenceMatcher(None, ref_words, cand_words)
    output_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for word in ref_words[i1:i2]:
                output_parts.append(f'<span class="word-correct">{html.escape(word)}</span>')
        elif tag in {"replace", "delete"}:
            for word in ref_words[i1:i2]:
                output_parts.append(f'<span class="word-wrong">{html.escape(word)}</span>')

    return " ".join(output_parts)


SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
init_state()

st_autorefresh(interval=500, key="live_refresh")
process_log()
load_latest_session()

# Sidebar controls for choosing a model and recording settings
st.sidebar.title("Controls")
model_options = ["zipformer", "vosk", "whispervad", "simulstreaming", "qwen3vad"]

st.sidebar.selectbox(
    "Model",
    model_options,
    key="model_name",
)

st.sidebar.selectbox(
    "Interface",
    ["usb", "wifi"],
    key="interface",
)

st.sidebar.slider(
    "Gain",
    1.0,
    12.0,
    8.0,
    0.5,
    key="gain",
)

st.sidebar.checkbox(
    "Denoise",
    key="denoise",
)

st.sidebar.text_input(
    "Test ID",
    key="test_id",
    placeholder="e.g. test_014",
)

st.sidebar.text_input(
    "Person ID",
    key="person_id",
    placeholder="e.g. person_007",
)

st.sidebar.selectbox(
    "Gender",
    ["not_specified", "female", "male", "diverse"],
    key="gender",
)

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Start", use_container_width=True):
        start_backend()
with c2:
    if st.button("Stop", use_container_width=True):
        stop_backend()

if st.sidebar.button("Compute Metrics", use_container_width=True):
    compute_metrics()

running = is_running()
status_class = "status-running" if running else "status-stopped"
status_text = "RUNNING" if running else "STOPPED"

header_left, header_right = st.columns([1, 5])

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=620)

with header_right:
    st.title("")

st.markdown(
    f"""
    <span class="status-chip {status_class}">{status_text}</span>
    <span class="status-chip info-chip">model: {st.session_state.model_name}</span>
    <span class="status-chip info-chip">denoise: {st.session_state.denoise}</span>
    <span class="status-chip info-chip">interface: {st.session_state.interface}</span>
    <span class="status-chip info-chip">test_id: {st.session_state.test_id}</span>
    <span class="status-chip info-chip">person_id: {st.session_state.person_id}</span>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)
# Display live transcription
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Live Transcription</div>', unsafe_allow_html=True)

    if st.session_state.model_name == "simulstreaming" and st.session_state.live_hyp:
        live_text = st.session_state.live_hyp
    else:
        live_text = st.session_state.stable_text
        if st.session_state.last_partial:
            live_text = (live_text + " " + st.session_state.last_partial).strip()

    if live_text:
        live_text_html = render_word_diff_html(GROUND_TRUTH_TEXT, live_text)
    else:
        live_text_html = "No transcription yet."

    st.markdown(
        f'<div class="transcript-box">{live_text_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Live Event Log", expanded=False):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    log_text = "\n".join(st.session_state.all_lines[-40:]) if st.session_state.all_lines else "No output yet."
    st.markdown(
        f'<div class="event-box">{html.escape(log_text)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
# Display reference text for comparison    
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Ground Truth</div>', unsafe_allow_html=True)

    if live_text:
        gt_html = render_ground_truth_diff_html(GROUND_TRUTH_TEXT, live_text)
    else:
        gt_html = html.escape(GROUND_TRUTH_TEXT)

    st.markdown(f'<div class="gt-box">{gt_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# Show recognition quality metrics
st.markdown("### Metrics")
data = st.session_state.last_session_data
metrics_row = None
if data:
    metrics_row = load_last_metrics_row_for_session(data["session_id"])

if not metrics_row:
    st.info("Stop the stream, then click Compute Metrics.")
else:
    row1 = st.columns(4)
    with row1[0]:
        render_metric_card("WER", fmt_metric(metrics_row.get("wer")), "lower is better")
    with row1[1]:
        render_metric_card("CER", fmt_metric(metrics_row.get("cer")), "lower is better")
    with row1[2]:
        render_metric_card("PER", fmt_metric(metrics_row.get("per")), "lower is better")
    with row1[3]:
        render_metric_card("BERTScore-F1", fmt_metric(metrics_row.get("bertscore_f1")), "higher is better")

    row2m = st.columns(3)
    with row2m[0]:
        render_metric_card("SER", fmt_metric(metrics_row.get("ser")), "lower is better")
    with row2m[1]:
        render_metric_card("BERTScore-P", fmt_metric(metrics_row.get("bertscore_precision")), "higher is better")
    with row2m[2]:
        render_metric_card("BERTScore-R", fmt_metric(metrics_row.get("bertscore_recall")), "higher is better")
# Show timing and latency metrics
st.markdown("### Important Latency Metrics")
if data and data.get("latency") and data.get("output_latency"):
    lat = data["latency"]
    olat = data["output_latency"]

    row2 = st.columns(4)
    with row2[0]:
        render_metric_card("RTF", fmt_metric(lat.get("rtf")), "compute time / audio duration")
    with row2[1]:
        render_metric_card("First Output", fmt_seconds(olat.get("start_to_first_output_s")), "stream start → first visible output")
    with row2[2]:
        render_metric_card("Mean Inter-Output", fmt_seconds(olat.get("mean_inter_output_s")), "average delay between outputs")
    with row2[3]:
        render_metric_card("Num Outputs", str(olat.get("num_outputs", "n/a")), "partial + final outputs")
else:
    st.info("Latency values appear after a finished session.")