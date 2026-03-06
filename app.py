"""
Streamlit Web UI for Natural Scene Text Detection & Recognition.

Launch with:
    streamlit run app.py

Provides an interactive interface to:
  - Upload natural scene images
  - Adjust detection and recognition thresholds via sliders
  - View annotated results side-by-side with the original
  - Browse detected text in a structured table
  - Download results as JSON
"""

import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    EAST_CONF_THRESHOLD,
    EAST_INPUT_HEIGHT,
    EAST_INPUT_WIDTH,
    EAST_MODEL_PATH,
    EAST_NMS_THRESHOLD,
    TESSERACT_CONF_THRESHOLD,
)
from src.detector import load_east_model
from src.pipeline import SceneTextPipeline, annotate_image

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Scene Text Detection & Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached model loading — only happens once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EAST text detection model...")
def get_pipeline(east_width, east_height, east_conf, east_nms, tess_conf):
    """
    Create and cache a SceneTextPipeline instance.

    Streamlit caches this across reruns so the EAST model is loaded only once.
    When any parameter changes the pipeline is recreated.
    """
    return SceneTextPipeline(
        east_model_path=EAST_MODEL_PATH,
        east_width=east_width,
        east_height=east_height,
        east_conf=east_conf,
        east_nms=east_nms,
        tess_conf=tess_conf,
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def pil_to_cv2(pil_image):
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    rgb = np.array(pil_image)
    if len(rgb.shape) == 2:
        # Grayscale — convert to BGR
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert an OpenCV BGR numpy array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_temp_image(pil_image, suffix=".jpg"):
    """Save a PIL image to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_image.save(tmp, format="JPEG" if suffix == ".jpg" else "PNG")
    tmp.close()
    return tmp.name


def format_confidence(conf):
    """Format a confidence value as a colored string for display."""
    if conf >= 80:
        return f"High ({conf:.1f}%)"
    elif conf >= 50:
        return f"Medium ({conf:.1f}%)"
    else:
        return f"Low ({conf:.1f}%)"


def render_page_header():
    """Render the hero section with top-level context."""
    st.title("Natural Scene Text Detection & Recognition")
    st.caption(
        "A hybrid EAST + Tesseract OCR pipeline for robust scene text understanding."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Backbone", "EAST (OpenCV dnn)")
    col2.metric("Recognition Engine", "Tesseract OCR")
    col3.metric("Pipeline Latency", "~1s / 1MP image (CPU)")

    st.markdown(
        "Built for research demos, assistive tooling, and navigation use cases where "
        "natural scene images replace clean scanned documents."
    )


def render_project_snapshot():
    """Show high-level goals, scope, and usage tips."""
    st.markdown("### Project Snapshot")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Core Objectives (PRD §3)**")
        st.markdown(
            "- O1: Localize text regions with EAST on CPU hardware.\n"
            "- O2: Recognize cropped regions with Tesseract + preprocessing.\n"
            "- O3: Handle noisy, multi-oriented street or storefront text.\n"
            "- O4: Offer both CLI and Streamlit experiences."
        )
        st.markdown("**In Scope**")
        st.markdown(
            "- Printed English text (JPEG/PNG scenes)\n"
            "- Batch & single-image processing\n"
            "- Evaluation against ICDAR-style ground truth\n"
            "- Modular code for experimentation"
        )

    with col2:
        st.markdown("**Target Users**")
        st.markdown(
            "- Researchers / students benchmarking OCR stacks\n"
            "- Assistive tech builders translating street signs\n"
            "- Navigation or AR apps enriching map metadata\n"
            "- Annotation teams accelerating labelling workflows"
        )
        st.markdown("**Key Assets**")
        st.markdown(
            "- `models/frozen_east_text_detection.pb`\n"
            "- `data/sample_images/` + `data/ground_truth/`\n"
            "- `output/annotated/` + `output/results/`\n"
            "- Documentation: `README.md`, `PRD.md`"
        )


def load_sample_assets():
    """Return a sample prediction JSON and annotated image if available."""
    results_dir = Path("output/results")
    annotated_dir = Path("output/annotated")

    json_path = next(results_dir.glob("*.json"), None) if results_dir.exists() else None
    if json_path is None:
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    image_name = Path(payload.get("image_path", "")).stem
    annotated_path = None
    if annotated_dir.exists():
        annotated_path = next(
            annotated_dir.glob(f"{image_name}*_annotated.*"), None
        )

    return {
        "json_path": json_path,
        "annotated_path": annotated_path,
        "payload": payload,
    }


def render_pipeline_tab():
    """Display an in-depth walkthrough of the workflow."""
    st.subheader("Pipeline Walkthrough")
    st.markdown(
        "The system follows a modular, two-stage design so detection and recognition "
        "can be tuned independently."
    )

    steps = [
        {
            "title": "Step 1: Preprocessing",
            "file": "src/preprocessor.py",
            "details": [
                "Resize inputs to multiples of 32 (default 320x320) while preserving a copy of the original frame.",
                "Apply channel-wise mean subtraction (123.68, 116.78, 103.94).",
                "Expose helpers for rotation-aware cropping and enhancement.",
            ],
        },
        {
            "title": "Step 2: EAST Detection",
            "file": "src/detector.py",
            "details": [
                "Loads the frozen EAST graph once via OpenCV `dnn`.",
                "Decodes geometry + score maps into rotated boxes then applies NMS.",
                "Thresholds controlled by `EAST_CONF_THRESHOLD` and `EAST_NMS_THRESHOLD`.",
            ],
        },
        {
            "title": "Step 3: Crop & Enhance",
            "file": "src/preprocessor.py",
            "details": [
                "Deskew crops by rotating around detected angles.",
                "Adds configurable padding (default 4 px).",
                "Produces grayscale + adaptive thresholded variants for OCR fallback.",
            ],
        },
        {
            "title": "Step 4: Tesseract Recognition",
            "file": "src/recognizer.py",
            "details": [
                "Uses PSM 7 (single line) with English whitelist.",
                "Falls back from enhanced to raw crop when confidence < threshold.",
                "Outputs text, confidence, and crop source tag.",
            ],
        },
        {
            "title": "Step 5: Pipeline Orchestration",
            "file": "src/pipeline.py",
            "details": [
                "Coordinates detection -> preprocessing -> recognition.",
                "Flattens bounding boxes, logs timings, and returns JSON-serializable results.",
                "Provides helpers for directory processing and annotation rendering.",
            ],
        },
    ]

    for step in steps:
        with st.expander(f"{step['title']} - {step['file']}"):
            for bullet in step["details"]:
                st.markdown(f"- {bullet}")

    st.markdown("#### System Architecture (PRD §6)")
    architecture = "\n".join(
        [
            "Input Image",
            "    |",
            "    v",
            "Preprocessor -> EAST Detector -> Crop & Enhance -> "
            "Tesseract OCR -> Output JSON + Annotated Image",
        ]
    )
    st.code(architecture, language="text")

    st.markdown("#### Module Breakdown")
    module_rows = [
        {
            "Module": "CLI Entrypoint",
            "File": "main.py",
            "Purpose": "Batch/CLI execution of the SceneTextPipeline",
        },
        {
            "Module": "Streamlit App",
            "File": "app.py",
            "Purpose": "Interactive upload + workflow showcase",
        },
        {
            "Module": "Evaluator",
            "File": "evaluate.py",
            "Purpose": "Computes precision/recall, word accuracy, CER",
        },
        {
            "Module": "Config",
            "File": "src/config.py",
            "Purpose": "Centralizes all tunable constants",
        },
        {
            "Module": "Tests",
            "File": "tests/",
            "Purpose": "Unit tests for preprocessor/detector/recognizer/pipeline",
        },
    ]
    st.table(module_rows)


def render_data_tab():
    """Surface dataset information, evaluation knobs, and sample outputs."""
    st.subheader("Datasets & Evaluation Data")
    dataset_rows = [
        {
            "Dataset": "Sample Scenes",
            "Location": "data/sample_images/",
            "Purpose": "Manual smoke tests + demo inside this UI",
        },
        {
            "Dataset": "Ground Truth",
            "Location": "data/ground_truth/",
            "Purpose": "JSON annotations for evaluation script",
        },
        {
            "Dataset": "ICDAR 2013/2015",
            "Location": "external",
            "Purpose": "Benchmark datasets referenced in PRD §17",
        },
    ]
    st.table(dataset_rows)

    st.markdown("#### Evaluation Workflow")
    st.markdown(
        "Run `evaluate.py` to compare predictions against `data/ground_truth/`.\n"
        "Key metrics include Precision, Recall, F1 for detection and Word Accuracy + CER for recognition."
    )
    st.code(
        "python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/\n"
        "python evaluate.py --images data/sample_images/ --ground-truth data/ground_truth/ --iou 0.6",
        language="bash",
    )

    st.markdown("#### Sample Prediction Artifact")
    sample = load_sample_assets()
    if sample is None:
        st.info(
            "No prediction artifacts found in `output/results/`. Upload an image via the demo or run `main.py` to generate them."
        )
        return

    payload = sample["payload"]
    detections = payload.get("detections", [])
    recognized = [d for d in detections if d.get("text")]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regions Detected", len(detections))
    col_b.metric("Text Recognized", len(recognized))
    col_c.metric("Processing Time", f"{payload.get('processing_time_ms', 0):.0f} ms")

    annotated_path = sample.get("annotated_path")
    if annotated_path and annotated_path.exists():
        st.image(
            str(annotated_path),
            caption=f"Annotated output: {annotated_path.name}",
            use_container_width=True,
        )

    preview = {
        "image_path": payload.get("image_path"),
        "image_size": payload.get("image_size"),
        "detections_example": detections[:3],
    }
    st.json(preview)


def render_docs_tab():
    """Provide quick references for commands, configuration, and troubleshooting."""
    st.subheader("Docs & Resources")
    st.markdown(
        "- **Product Requirements:** `PRD.md`\n"
        "- **Technical Overview & Usage:** `README.md`\n"
        "- **Setup Script:** `setup.sh` (installs deps + downloads EAST model)"
    )

    st.markdown("#### CLI Usage")
    st.code(
        "# Single image\npython main.py --image path/to/image.jpg --visualize\n\n"
        "# Batch directory\npython main.py --dir data/sample_images/ --save-json",
        language="bash",
    )

    st.markdown("#### Configuration Tips (`src/config.py`)")
    st.markdown(
        "- `EAST_INPUT_WIDTH/HEIGHT`: multiples of 32; increase for smaller text.\n"
        "- `EAST_CONF_THRESHOLD`: raise to reduce false positives.\n"
        "- `TESSERACT_CONF_THRESHOLD`: lower to inspect low-confidence words.\n"
        "- `CROP_PADDING`: increase if characters are clipped."
    )

    st.markdown("#### Testing & Quality Gates")
    st.code("python -m pytest tests/ -v", language="bash")
    st.markdown(
        "Unit tests cover preprocessing, detection decoding, OCR integration, and the full pipeline."
    )

    st.markdown("#### Troubleshooting")
    st.markdown(
        "- Missing EAST model -> run `./setup.sh` or download into `models/`.\n"
        "- Missing Tesseract -> install via system package manager (`tesseract-ocr`).\n"
        "- Non-multiple input sizes -> adjust Streamlit sliders or CLI args."
    )

    st.markdown("#### References")
    st.markdown(
        "- EAST: Zhou et al., 2017 (`frozen_east_text_detection.pb`).\n"
        "- Tesseract OCR 4.x.\n"
        "- ICDAR 2013 & 2015 scene text benchmarks."
    )

# ---------------------------------------------------------------------------
# Sidebar — parameters and controls
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the sidebar with parameter controls and return their values."""
    st.sidebar.title("Settings")

    st.sidebar.markdown("### EAST Detector")

    east_width = st.sidebar.select_slider(
        "Input Width",
        options=[160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768],
        value=EAST_INPUT_WIDTH,
        help="EAST model input width (must be a multiple of 32). "
        "Higher values detect smaller text but are slower.",
    )

    east_height = st.sidebar.select_slider(
        "Input Height",
        options=[160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768],
        value=EAST_INPUT_HEIGHT,
        help="EAST model input height (must be a multiple of 32).",
    )

    east_conf = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=EAST_CONF_THRESHOLD,
        step=0.05,
        help="Minimum confidence score for EAST to consider a region as text. "
        "Lower values find more text but may include false positives.",
    )

    east_nms = st.sidebar.slider(
        "NMS Threshold",
        min_value=0.1,
        max_value=1.0,
        value=EAST_NMS_THRESHOLD,
        step=0.05,
        help="Non-Maximum Suppression threshold. Higher values allow more "
        "overlapping boxes to survive.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tesseract OCR")

    tess_conf = st.sidebar.slider(
        "Min Word Confidence",
        min_value=0,
        max_value=100,
        value=int(TESSERACT_CONF_THRESHOLD),
        step=5,
        help="Minimum word-level confidence (0–100) from Tesseract. "
        "Words below this threshold are discarded.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Scene Text Detection & Recognition**\n\n"
        "A hybrid OCR pipeline combining the EAST deep learning "
        "text detector with Tesseract OCR for accurate text "
        "extraction from natural scene images.\n\n"
        "[Project PRD](PRD.md)"
    )

    return {
        "east_width": east_width,
        "east_height": east_height,
        "east_conf": east_conf,
        "east_nms": east_nms,
        "tess_conf": float(tess_conf),
    }


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
def render_upload_section():
    """Render the image upload section and return the uploaded file."""
    st.subheader("Run the Pipeline on Your Image")
    st.markdown(
        "Upload a natural scene image (street sign, shop board, billboard, etc.) "
        "and the system will detect and recognize text in it."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Supported formats: JPEG, PNG, BMP, TIFF, WebP",
    )

    return uploaded_file


def render_results(result, original_pil, annotated_pil):
    """Render detection results: images side by side + text table."""

    # ---- Images side by side ----
    st.markdown("---")
    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(original_pil, use_container_width=True)
    with col2:
        st.markdown("**Annotated Image**")
        st.image(annotated_pil, use_container_width=True)

    # ---- Metrics row ----
    st.markdown("---")
    detections = result.get("detections", [])
    text_detections = [d for d in detections if d.get("text")]
    total_regions = len(detections)
    recognized = len(text_detections)
    elapsed = result.get("processing_time_ms", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regions Detected", total_regions)
    m2.metric("Text Recognized", recognized)
    m3.metric(
        "Avg OCR Confidence",
        (
            f"{sum(d['confidence'] for d in text_detections) / len(text_detections):.1f}%"
            if text_detections
            else "N/A"
        ),
    )
    m4.metric("Processing Time", f"{elapsed:.0f} ms")

    # ---- Detection table ----
    if text_detections:
        st.markdown("---")
        st.subheader("Detected Text")

        table_data = []
        for det in text_detections:
            table_data.append(
                {
                    "ID": det["id"],
                    "Text": det["text"],
                    "OCR Confidence": format_confidence(det["confidence"]),
                    "Detection Confidence": format_confidence(
                        det.get("detection_confidence", 0)
                    ),
                    "Source": det.get("source", "unknown"),
                }
            )

        st.table(table_data)
    else:
        st.warning(
            "No text was recognized in this image. Try:\n"
            "- Lowering the detection confidence threshold\n"
            "- Increasing the input width/height for smaller text\n"
            "- Lowering the minimum word confidence"
        )

    # ---- All detections (including empty) in expander ----
    if detections:
        with st.expander(f"All {total_regions} detected regions (including empty)"):
            for det in detections:
                text = det.get("text", "")
                conf = det.get("confidence", 0)
                det_conf = det.get("detection_confidence", 0)
                source = det.get("source", "unknown")
                status = f'"{text}"' if text else "(no text recognized)"

                st.markdown(
                    f"**Region {det['id']}:** {status} - "
                    f"OCR: {conf:.1f}%, Detection: {det_conf:.1f}%, Source: {source}"
                )

    # ---- Download button ----
    st.markdown("---")
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])

    with col_dl1:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Results",
            data=json_str,
            file_name="scene_text_results.json",
            mime="application/json",
        )

    with col_dl2:
        # Convert annotated image to bytes for download
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Annotated Image",
            data=buf.getvalue(),
            file_name="annotated_result.jpg",
            mime="image/jpeg",
        )


def render_no_model_error():
    """Show an error message when the EAST model file is missing."""
    st.error(
        f"**EAST model not found** at `{EAST_MODEL_PATH}`.\n\n"
        "Please run the setup script to download the model:\n"
        "```\n"
        "chmod +x setup.sh && ./setup.sh\n"
        "```\n\n"
        "Or download it manually from:\n"
        "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb\n\n"
        f"and place it at `{EAST_MODEL_PATH}`."
    )


# ---------------------------------------------------------------------------
# Main app flow
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit application."""

    model_available = os.path.isfile(EAST_MODEL_PATH)

    # Render sidebar and get parameters (even if the model is missing so users can explore settings)
    params = render_sidebar()

    # Hero + context sections
    render_page_header()
    render_project_snapshot()

    demo_tab, pipeline_tab, data_tab, docs_tab = st.tabs(
        ["Interactive Demo", "Pipeline", "Data & Evaluation", "Docs"]
    )

    with pipeline_tab:
        render_pipeline_tab()

    with data_tab:
        render_data_tab()

    with docs_tab:
        render_docs_tab()

    with demo_tab:
        if not model_available:
            render_upload_section()
            render_no_model_error()
            return

        uploaded_file = render_upload_section()

        if uploaded_file is None:
            st.markdown("---")
            st.markdown(
                """
                ### How it works

                1. **Upload** a natural scene image containing text (signs, boards, etc.)
                2. **EAST Detector** localizes text regions using a deep learning model
                3. **Preprocessing** enhances each detected region (grayscale, threshold, denoise)
                4. **Tesseract OCR** recognizes characters in each preprocessed region
                5. **Results** are displayed with annotated bounding boxes and recognized text

                ### Tips for best results

                - Image size: Larger images with clearly visible text work best.
                - Small text: Increase the input width/height in the sidebar (e.g., 640x640).
                - False positives: Increase the detection confidence threshold.
                - Missed text: Decrease the detection confidence threshold.
                - Poor recognition: Decrease the minimum word confidence.
                """
            )

            with st.expander("Sample JSON output format"):
                sample = {
                    "image_path": "/path/to/image.jpg",
                    "image_size": [480, 640],
                    "detections": [
                        {
                            "id": 1,
                            "bbox": [102, 55, 310, 55, 310, 98, 102, 98],
                            "text": "MAIN STREET",
                            "confidence": 87.5,
                            "detection_confidence": 92.3,
                            "source": "enhanced",
                        }
                    ],
                    "total_detections": 1,
                    "processing_time_ms": 312.4,
                }
                st.json(sample)
            return

        # ---- Process the uploaded image ----
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to open the uploaded image: {e}")
            return

        with st.spinner("Detecting and recognizing text..."):
            temp_path = save_temp_image(pil_image, suffix=".jpg")

            try:
                pipeline = get_pipeline(
                    east_width=params["east_width"],
                    east_height=params["east_height"],
                    east_conf=params["east_conf"],
                    east_nms=params["east_nms"],
                    tess_conf=params["tess_conf"],
                )

                result = pipeline.process_image(temp_path)
                result["image_path"] = uploaded_file.name

                cv2_image = pil_to_cv2(pil_image)
                annotated_cv2 = annotate_image(cv2_image, result.get("detections", []))
                annotated_pil = cv2_to_pil(annotated_cv2)

                render_results(result, pil_image, annotated_pil)

            except EnvironmentError as e:
                st.error(
                    f"**Environment error:** {e}\n\n"
                    "Make sure Tesseract OCR is installed:\n"
                    "```\n"
                    "sudo apt install tesseract-ocr tesseract-ocr-eng\n"
                    "```"
                )
            except Exception as e:
                st.error(f"**Processing failed:** {e}")
                logger.exception("Error processing image: %s", e)
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
