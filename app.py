import os
import sys
import csv
import json
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = Path("/home/iat/projectaria_client_sdk_samples")
ASR_DEMO_DIR = PROJECT_ROOT / "asr_demo"
BACKEND_PATH = ASR_DEMO_DIR / "streamlit_backend.py"
LOG_PATH = ASR_DEMO_DIR / "live_output.log"
PID_PATH = ASR_DEMO_DIR / "live_backend.pid"
LATEST_SESSION_PATH = ASR_DEMO_DIR / "latest_session.json"
SESSIONS_DIR = ASR_DEMO_DIR / "sessions"
METRICS_CSV_PATH = SESSIONS_DIR / "metrics.csv"
LOGO_PATH = ASR_DEMO_DIR / "logo.png"

from metrics.metrics import (
    MetricsStore,
    LatencyResult,
    OutputLatencyResult,
)

GROUND_TRUTH_TEXT = """Sterilisation ist ein validiertes Verfahren zur Inaktivierung lebensfähiger Mikroorganismen.
Der Zustand der Sterilität beschreibt das Freisein von vermehrungsfähigen Keimen.
Da absolute Keimfreiheit praktisch nicht nachweisbar ist, wird das Sterilitätssicherheitsniveau
als Wahrscheinlichkeitsmaß verwendet.
Vor der Sterilisation erfolgt die Reinigung zur Entfernung sichtbarer Anschmutzungen sowie die
Desinfektion zur Reduktion der mikrobiellen Belastung.
Die vorhandene Ausgangsbelastung wird als Bioburden bezeichnet.
Die Wirksamkeit eines Sterilisationsverfahrens hängt von definierten Prozessparametern wie
Temperatur, Druck, Einwirkzeit, Luftfeuchte und Konzentration des sterilisierenden Agens ab.
Die Validierung umfasst Installationsqualifizierung, Funktionsqualifizierung und
Leistungsqualifizierung.
Die Verifizierung bestätigt die Einhaltung festgelegter Anforderungen.
Ein Sterilbarrieresystem verhindert das Eindringen von Mikroorganismen und ermöglicht eine
aseptische Bereitstellung.
Die parametrische Freigabe erfolgt auf Basis dokumentierter Prozessdaten ohne zusätzliche
Endproduktprüfung.
Die Aufbereitung von wiederverwendbaren Medizinprodukten beinhaltet Vorbereitung, Reinigung,
Desinfektion, Prüfung auf Sauberkeit und Funktionalität, Verpackung, Sterilisation und
sachgerechte Lagerung.
"""

st.set_page_config(page_title="ScribeLab ASR Dashboard", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #030303 0%, #090909 100%);
        color: white;
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"] {
        color: white;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
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

    /* Sidebar text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Branding */
    .brand-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.08em;
        line-height: 1.1;
        margin-top: 0.3rem;
    }

    .brand-subtitle {
        color: #9aa0a6;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        font-family: 'Inter', sans-serif;
    }

    /* Panels */
    .panel {
        background: #111111;
        border: 1px solid rgba(127, 255, 212, 0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 0 18px rgba(255, 255, 255, 0.02);
    }

    .panel-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #ffffff;
        letter-spacing: 0.04em;
    }

    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.04em;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* Content boxes */
    .transcript-box, .gt-box, .event-box {
        background: #0b0b0b;
        border: 1px solid #2b2b2b;
        border-radius: 14px;
        padding: 1rem;
        white-space: pre-wrap;
        line-height: 1.55;
        font-family: 'Inter', sans-serif;
    }

    .transcript-box {
        min-height: 320px;
        color: #f5f5f5;
        font-size: 1rem;
    }

    .gt-box {
        min-height: 320px;
        color: #c8c8c8;
        font-size: 0.95rem;
    }

    .event-box {
        min-height: 300px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.45;
        color: #d8d8d8;
        overflow-y: auto;
    }

    /* Status chips */
    .status-chip {
        display: inline-block;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.35rem;
        border: 1px solid #333;
        letter-spacing: 0.03em;
    }

    .status-running {
        background: rgba(0, 255, 170, 0.10);
        color: #7fffd4;
        border-color: rgba(0, 255, 170, 0.22);
    }

    .status-stopped {
        background: rgba(255, 90, 90, 0.10);
        color: #ff9d9d;
        border-color: rgba(255, 90, 90, 0.22);
    }

    .info-chip {
        background: #101214;
        color: #c7c9cc;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(180deg, #111111 0%, #0b0b0b 100%);
        border: 1px solid #252525;
        border-radius: 16px;
        padding: 0.9rem;
        min-height: 108px;
        box-shadow: 0 0 18px rgba(255, 255, 255, 0.03);
    }

    .metric-label {
        color: #bbbbbb;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
        font-family: 'Inter', sans-serif;
    }

    .metric-value {
        color: white;
        font-weight: 800;
        font-size: 1.6rem;
        line-height: 1.1;
        font-family: 'Inter', sans-serif;
    }

    .metric-help {
        color: #888888;
        font-size: 0.78rem;
        margin-top: 0.25rem;
        font-family: 'Inter', sans-serif;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
    }

    div[data-baseweb="select"] * {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    div[role="listbox"] {
        background: #121212 !important;
        border: 1px solid #333 !important;
    }

    div[role="option"] {
        background: #121212 !important;
        color: #ffffff !important;
    }

    div[role="option"]:hover {
        background: #1c1c1c !important;
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background-color: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stSlider * {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stCheckbox * {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 12px !important;
        border: 1px solid #333 !important;
        background: #171717 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton button:hover {
        border-color: #555 !important;
        background: #202020 !important;
    }

    /* SVG icons including select arrows */
    svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #111111 !important;
        border: 1px solid #252525 !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] details {
        background: #111111 !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] summary {
        background: #111111 !important;
        color: #ffffff !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpanderContent"] {
        background: #111111 !important;
        color: #ffffff !important;
    }

    details {
        background: #111111 !important;
        border: 1px solid #252525 !important;
        border-radius: 14px !important;
    }

    summary {
        background: #111111 !important;
        color: #ffffff !important;
        border-radius: 14px !important;
    }

    .live-dot {
        color: #7fffd4;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "model_name": "zipformer",
        "interface": "usb",
        "gain": 8.0,
        "denoise": False,
        "test_id": "test_001",
        "person_id": "person_001",
        "stable_text": "",
        "last_partial": "",
        "all_lines": [],
        "last_session_data": None,
        "metrics_store": MetricsStore(out_dir=str(SESSIONS_DIR), per_language="de"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    st.session_state.all_lines = []
    st.session_state.last_session_data = None


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

    for line in lines:
        if line.startswith("[FINAL] "):
            txt = line[len("[FINAL] "):].strip()
            if txt:
                stable_text = (stable_text + " " + txt).strip() if stable_text else txt
            last_partial = ""
        elif line.startswith("[PARTIAL] "):
            last_partial = line[len("[PARTIAL] "):].strip()

    st.session_state.stable_text = stable_text
    st.session_state.last_partial = last_partial


def load_latest_session():
    if not LATEST_SESSION_PATH.exists():
        return
    try:
        with open(LATEST_SESSION_PATH, "r", encoding="utf-8") as f:
            st.session_state.last_session_data = json.load(f)
    except Exception:
        st.session_state.last_session_data = None


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


SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
init_state()

st_autorefresh(interval=500, key="live_refresh")
process_log()
load_latest_session()

st.sidebar.title("Controls")
st.session_state.model_name = st.sidebar.selectbox(
    "Model",
    ["zipformer", "vosk", "whisper", "wav2vec2"],
    index=["zipformer", "vosk", "whisper", "wav2vec2"].index(st.session_state.model_name),
)
st.session_state.interface = st.sidebar.selectbox(
    "Interface",
    ["usb", "wifi"],
    index=0 if st.session_state.interface == "usb" else 1,
)
st.session_state.gain = st.sidebar.slider("Gain", 1.0, 12.0, float(st.session_state.gain), 0.5)
st.session_state.denoise = st.sidebar.checkbox("Denoise", value=st.session_state.denoise)
st.session_state.test_id = st.sidebar.text_input("Test ID", value=st.session_state.test_id)
st.session_state.person_id = st.sidebar.text_input("Person ID", value=st.session_state.person_id)

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

header_spacer_left, header_left, header_right, header_spacer_right = st.columns([0.25, 1.1, 2.6, 0.25])

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=230)

with header_right:
    st.markdown(
        """
        <div style="padding-top: 1.0rem; text-align: center;">
            <div class="brand-title">Live ASR and evaluation dashboard</div>
            <div class="brand-subtitle"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='height:1px; background: linear-gradient(90deg, transparent, #2a2a2a, transparent); margin: 0.8rem 0 1.2rem 0;'></div>",
    unsafe_allow_html=True,
)

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

st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)

left, right = st.columns([1.2, 0.8])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Live Transcription <span class="live-dot">● live</span></div>',
        unsafe_allow_html=True,
    )

    live_text = st.session_state.stable_text
    if st.session_state.last_partial:
        live_text = (live_text + " " + st.session_state.last_partial).strip()

    st.markdown(
        f'<div class="transcript-box">{live_text if live_text else "No transcription yet."}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    log_text = "\n".join(st.session_state.all_lines[-40:]) if st.session_state.all_lines else "No output yet."
    with st.expander("Live Event Log", expanded=True):
        st.markdown(
            f'<div class="event-box">{log_text}</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Ground Truth</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="gt-box">{GROUND_TRUTH_TEXT}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-title">Important Latency Metrics</div>', unsafe_allow_html=True)
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