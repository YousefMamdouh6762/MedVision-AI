
from pathlib import Path
import io
import base64
import hashlib
import html

import streamlit as st
import torch
from PIL import Image, ImageFile
from torchvision import models, transforms
from ultralytics import YOLO


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

REGION_MODEL_PATH = (
    ROOT / "models" / "bone_region_resnet18.pth"
)

FRACTURE_MODELS = {
    "YOLO11n V3": ROOT / "models" / "fracture_yolo_v3.pt",
    "YOLO11s V4": ROOT / "models" / "fracture_yolo_v4.pt"
}

st.set_page_config(
    page_title="MedVision-AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. ASSETS
# ==========================================

def find_asset(keywords):
    if not ASSETS.exists():
        return None

    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    files = [
        path for path in ASSETS.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    for keyword in keywords:
        for path in files:
            if keyword in path.stem.lower():
                return path

    return None


def image_data_uri(path):
    if path is None:
        return None

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }

    mime = mime_types.get(
        path.suffix.lower(), "image/png"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


logo_uri = image_data_uri(
    find_asset(["logo", "icon", "brand"])
)

banner_uri = image_data_uri(
    find_asset(["banner", "hero", "chest", "header"])
)


# ==========================================
# 3. CSS DESIGN
# ==========================================

banner_css = ""

if banner_uri:
    banner_css = f"""
    .mv-hero {{
        background-image:
            linear-gradient(
                90deg,
                rgba(5,24,44,.98) 0%,
                rgba(5,24,44,.92) 45%,
                rgba(5,24,44,.10) 100%
            ),
            url('{banner_uri}');
        background-size: cover;
        background-position: center right;
    }}
    """

st.markdown(
    """
    <style>
    .stApp {
        background: #061629;
        color: #eef6ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #082542, #06192e
        );
        border-right: 1px solid #20517b;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        color: #f3f8ff !important;
    }

    /* Brand */

    .mv-brand {
        text-align: center;
        padding: 15px 0 20px;
    }

    .mv-brand-logo {
        width: 88px;
        height: 88px;
        object-fit: contain;
        display: block;
        margin: 0 auto 13px;
    }

    .mv-brand-name {
        color: white;
        font-size: 1.75rem;
        font-weight: 850;
        white-space: nowrap;
        letter-spacing: -0.7px;
    }

    .mv-brand-name span {
        color: #259cff;
    }

    .mv-brand-subtitle {
        color: #a5c6e7;
        font-size: .84rem;
        margin-top: 10px;
    }

    /* Sidebar navigation buttons */

    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        color: #dceeff;
        border: 1px solid transparent;
        border-radius: 12px;
        text-align: left;
        justify-content: flex-start;
        padding: 12px 15px;
        font-weight: 650;
        min-height: 46px;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #123a60;
        color: white;
        border-color: #205d90;
    }

    [data-testid="stSidebar"]
    .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg, #0879db, #0755a7
        );
        border: 1px solid #218ef1;
        color: white;
    }

    /* Hero */

    .mv-hero {
        min-height: 230px;
        background-color: #09233f;
        border: 1px solid #18527f;
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 22px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }

    .mv-kicker {
        color: #39a8ff;
        font-weight: 800;
        font-size: .77rem;
        letter-spacing: 3px;
        margin-bottom: 13px;
    }

    .mv-hero-title {
        color: white;
        font-size: clamp(2rem, 3.5vw, 3.3rem);
        font-weight: 850;
        line-height: 1.12;
    }

    .mv-hero-title span {
        color: #209bff;
    }

    .mv-hero-subtitle {
        color: #bdd8f2;
        font-size: 1.25rem;
        margin-top: 9px;
    }

    .mv-hero-description {
        color: #a6c2dc;
        font-size: .92rem;
        margin-top: 15px;
        max-width: 620px;
    }

    .mv-accent {
        background: #129bff;
        width: 68px;
        height: 5px;
        border-radius: 10px;
        margin-top: 14px;
    }

    /* Sections */

    .mv-section-title {
        color: #f2f7ff;
        font-size: 1.25rem;
        font-weight: 750;
        margin: 18px 0 12px;
    }

    /* Cards */

    .mv-card {
        background: linear-gradient(
            135deg, #102e4d, #0a2139
        );
        border: 1px solid #20527b;
        border-radius: 15px;
        padding: 17px;
        margin-bottom: 12px;
    }

    .mv-card-label {
        color: #9cbfdf;
        font-size: .82rem;
        margin-bottom: 7px;
    }

    .mv-card-value {
        color: white;
        font-size: 1.55rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .mv-card-note {
        color: #87b4db;
        font-size: .79rem;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }

    /* Scores */

    .mv-score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        margin-bottom: 6px;
        gap: 12px;
    }

    .mv-score-name {
        color: #d9ebfc;
        font-size: .9rem;
        font-weight: 600;
    }

    .mv-score-value {
        color: #8fd0ff;
        font-size: .87rem;
        font-weight: 700;
    }

    .mv-score-track {
        width: 100%;
        height: 9px;
        background: #1a3856;
        border-radius: 20px;
        overflow: hidden;
    }

    .mv-score-fill {
        height: 100%;
        background: linear-gradient(
            90deg, #0875db, #35b4ff
        );
        border-radius: 20px;
    }

    /* Images */

    [data-testid="stImage"] img {
        border-radius: 13px;
        border: 1px solid #20517a;
    }

    /* Upload */

    [data-testid="stFileUploader"] {
        background: #0b2a47;
        border: 1px solid #205a88;
        border-radius: 15px;
        padding: 12px;
    }

    /* Main action button */

    .stMain .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg, #108ff6, #0866d8
        );
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 750;
        min-height: 3.3rem;
    }

    /* Downloads */

    [data-testid="stDownloadButton"] button {
        background: #0e3557;
        color: #d9eeff;
        border: 1px solid #2477b5;
        border-radius: 12px;
        font-weight: 650;
        min-height: 3rem;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #145080;
        color: white;
        border-color: #32a1ff;
    }

    /* Disclaimer */

    .mv-disclaimer {
        background: #0a304b;
        border: 1px solid #1376ac;
        border-radius: 12px;
        padding: 15px;
        color: #b9d8ed;
        font-size: .86rem;
        line-height: 1.6;
        margin-top: 14px;
    }

    .mv-footer {
        color: #7896b2;
        text-align: center;
        padding: 20px 0;
        font-size: .82rem;
    }

    hr {
        border-color: #214565 !important;
    }
    </style>
    """
    + f"<style>{banner_css}</style>",
    unsafe_allow_html=True
)


# ==========================================
# 4. SESSION STATE
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "YOLO11s V4"

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "analyzed_hash" not in st.session_state:
    st.session_state.analyzed_hash = None


# ==========================================
# 5. AI MODELS
# ==========================================

@st.cache_resource
def load_region_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        REGION_MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    labels = checkpoint["labels"]

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(labels)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, labels, device


@st.cache_resource
def load_fracture_model(model_path):

    model = YOLO(str(model_path))

    if len(model.names) != 1:
        raise ValueError(
            f"Unexpected model classes: {model.names}"
        )

    if str(model.names[0]).lower() != "fracture":
        raise ValueError(
            f"Not a fracture model: {model.names}"
        )

    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 6. DETECTION HELPERS
# ==========================================

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (
        max(0, box_a[2] - box_a[0])
        * max(0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0, box_b[2] - box_b[0])
        * max(0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def find_overlapping_pairs(detections, threshold=0.3):

    pairs = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):

            overlap = calculate_iou(
                detections[i]["box"],
                detections[j]["box"]
            )

            if overlap >= threshold:
                pairs.append((i + 1, j + 1, overlap))

    return pairs


def analyze_image(image, model_name, threshold):

    region_model, labels, device = load_region_model()

    model_path = FRACTURE_MODELS[model_name]

    fracture_model = load_fracture_model(
        str(model_path)
    )

    tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = region_model(tensor)
        probabilities = torch.sigmoid(output)[0]

    scores = sorted(
        zip(labels, probabilities.cpu().tolist()),
        key=lambda item: item[1],
        reverse=True
    )

    results = fracture_model.predict(
        source=image,
        conf=threshold,
        imgsz=800 if model_name == "YOLO11s V4" else 640,
        device=0 if device.type == "cuda" else "cpu",
        verbose=False
    )

    result = results[0]

    annotated_bgr = result.plot(
        line_width=2,
        labels=False,
        conf=False
    )

    annotated_rgb = annotated_bgr[:, :, ::-1].copy()

    detections = []

    for box in result.boxes:

        detections.append({
            "score": float(box.conf[0].item()),
            "box": [
                round(float(value), 1)
                for value in box.xyxy[0].tolist()
            ]
        })

    overlaps = find_overlapping_pairs(detections)

    return scores, detections, annotated_rgb, overlaps


# ==========================================
# 7. DOWNLOAD HELPERS
# ==========================================

def create_detection_png(image_array):

    buffer = io.BytesIO()

    Image.fromarray(image_array).save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def create_analysis_report(analysis):

    scores = analysis["scores"]
    detections = analysis["detections"]

    best_region, best_score = scores[0]

    lines = [
        "MEDVISION-AI | ANALYSIS REPORT",
        "=" * 45,
        "",
        f"Fracture model: {analysis['model']}",
        f"Most likely bone region: {best_region.title()}",
        f"Region model score: {best_score * 100:.2f}%",
        f"Potential detection boxes: {len(detections)}",
        f"Detection threshold: {analysis['threshold']:.2f}",
        "",
        "BONE REGION SCORES",
        "-" * 45
    ]

    for label, score in scores:
        lines.append(
            f"{label.title()}: {score * 100:.2f}%"
        )

    lines.extend([
        "",
        "DETECTION DETAILS",
        "-" * 45
    ])

    if detections:
        for index, detection in enumerate(
            detections, start=1
        ):
            lines.extend([
                f"Detection {index}",
                f"Score: {detection['score'] * 100:.2f}%",
                f"Box: {detection['box']}",
                ""
            ])
    else:
        lines.append("No detections at this threshold.")

    if analysis["overlaps"]:
        lines.extend([
            "",
            "OVERLAPPING BOXES",
            "-" * 45
        ])

        for first, second, overlap in analysis["overlaps"]:
            lines.append(
                f"Boxes {first} and {second}: IoU {overlap:.2f}"
            )

    lines.extend([
        "",
        "RESEARCH DISCLAIMER",
        "-" * 45,
        "A detection does not confirm a fracture.",
        "No detection does not rule out a fracture.",
        "Model scores are not calibrated probabilities.",
        "Not for clinical diagnosis or treatment decisions."
    ])

    return "\n".join(lines).encode("utf-8")


# ==========================================
# 8. REUSABLE UI COMPONENTS
# ==========================================

def section_title(title):

    st.markdown(
        f'<div class="mv-section-title">'
        f'{html.escape(title)}</div>',
        unsafe_allow_html=True
    )


def result_card(label, value, note):

    st.markdown(
        f"""
        <div class="mv-card">
            <div class="mv-card-label">
                {html.escape(str(label))}
            </div>
            <div class="mv-card-value">
                {html.escape(str(value))}
            </div>
            <div class="mv-card-note">
                {html.escape(str(note))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_scores(scores):

    for label, score in scores:

        percent = min(
            max(float(score) * 100, 0),
            100
        )

        st.markdown(
            f"""
            <div class="mv-score-row">
                <span class="mv-score-name">
                    {html.escape(str(label).title())}
                </span>
                <span class="mv-score-value">
                    {percent:.2f}%
                </span>
            </div>
            <div class="mv-score-track">
                <div
                    class="mv-score-fill"
                    style="width:{percent:.2f}%;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 9. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    logo_html = (
        f'<img class="mv-brand-logo" src="{logo_uri}">'
        if logo_uri else ""
    )

    st.markdown(
        f"""
        <div class="mv-brand">
            {logo_html}
            <div class="mv-brand-name">
                MedVision-<span>AI</span>
            </div>
            <div class="mv-brand-subtitle">
                AI-Assisted X-ray Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    navigation = [
        ("Home", "🏠 Home"),
        ("Analyze X-ray", "🩻 Analyze X-ray"),
        ("Results", "📊 Results"),
        ("About", "ℹ️ About"),
        ("Settings", "⚙️ Settings")
    ]

    for page_name, button_label in navigation:

        is_active = st.session_state.page == page_name

        if st.button(
            button_label,
            key=f"nav_{page_name}",
            type="primary" if is_active else "secondary",
            width="stretch"
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()

    st.caption(
        f"Research Prototype | "
        f"{st.session_state.selected_model} + ResNet18"
    )

    st.warning(
        "Not for clinical diagnosis or treatment decisions."
    )


# ==========================================
# 10. HERO
# ==========================================

st.markdown(
    """
    <div class="mv-hero">
        <div class="mv-kicker">
            BONE FRACTURE ANALYSIS
        </div>
        <div class="mv-hero-title">
            MedVision-<span>AI</span>
        </div>
        <div class="mv-hero-subtitle">
            AI-Powered X-ray Analysis
        </div>
        <div class="mv-accent"></div>
        <div class="mv-hero-description">
            Explore potential fracture regions and
            bone-region model predictions using
            your trained AI models.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 11. RESULTS DISPLAY
# ==========================================

def render_results(analysis):

    scores = analysis["scores"]
    detections = analysis["detections"]

    best_region, best_score = scores[0]

    st.caption(
        f"Fracture detection model used: "
        f"{analysis['model']}"
    )

    section_title("X-ray Analysis")

    original_col, detection_col, overview_col = st.columns(
        [1, 1, 0.85],
        gap="medium"
    )

    with original_col:

        st.markdown("**Original X-ray**")

        st.image(
            analysis["original"],
            width="stretch"
        )

    with detection_col:

        st.markdown("**AI Detection Result**")

        st.image(
            analysis["annotated"],
            width="stretch"
        )

    with overview_col:

        st.markdown("**Analysis Overview**")

        result_card(
            "Most likely bone region",
            best_region.title(),
            f"Region model score: {best_score * 100:.2f}%"
        )

        result_card(
            "Potential detection boxes",
            len(detections),
            "Overlapping boxes may refer to the same region"
        )

        result_card(
            "Detection threshold",
            f"{analysis['threshold']:.2f}",
            "Experimental setting"
        )

    if analysis["overlaps"]:

        pairs = ", ".join(
            f"{first} & {second} (IoU {overlap:.2f})"
            for first, second, overlap
            in analysis["overlaps"]
        )

        st.info(
            "Some detection boxes overlap: "
            + pairs
            + ". Overlapping boxes do not necessarily "
            "represent separate fractures."
        )

    detail_col1, detail_col2 = st.columns(
        2,
        gap="large"
    )

    with detail_col1:

        section_title("Bone Region Scores")
        render_scores(scores)

    with detail_col2:

        section_title("Fracture Detection Details")

        if detections:

            for index, detection in enumerate(
                detections, start=1
            ):

                result_card(
                    f"Potential region {index}",
                    f"Model score: "
                    f"{detection['score'] * 100:.2f}%",
                    f"Bounding box: {detection['box']}"
                )

        else:

            st.info(
                "No potential fracture regions detected "
                "at the selected threshold."
            )

    section_title("Download Results")

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(
            "⬇ Download Detection Image (PNG)",
            data=create_detection_png(
                analysis["annotated"]
            ),
            file_name="medvision_detection.png",
            mime="image/png",
            width="stretch"
        )

    with download_col2:

        st.download_button(
            "⬇ Download Analysis Report (TXT)",
            data=create_analysis_report(analysis),
            file_name="medvision_analysis_report.txt",
            mime="text/plain",
            width="stretch"
        )

    st.markdown(
        """
        <div class="mv-disclaimer">
            <b>Research interpretation:</b>
            A detection box does not confirm a fracture.
            No detection does not rule one out.
            Scores are model outputs, not calibrated
            medical probabilities.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 12. PAGE ROUTING
# ==========================================

page = st.session_state.page


if page in ["Home", "Analyze X-ray"]:

    analysis_type = st.radio(
        "Select Analysis Type",
        ["Fracture Analysis", "Chest Disease Analysis"],
        horizontal=True,
        key="analysis_type"
    )

    section_title("Upload & Analyze")

    upload_col, threshold_col = st.columns(
        [2, 1],
        gap="large"
    )

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload an X-ray image",
            type=["jpg", "jpeg", "png"]
        )

    with threshold_col:

        threshold = st.slider(
            "Detection score threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05
        )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        if st.session_state.analyzed_hash != file_hash:
            st.session_state.analysis = None

        try:

            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            if st.button(
                "Analyze X-ray",
                type="primary",
                width="stretch"
            ):

                with st.spinner("Analyzing X-ray..."):

                    (
                        scores,
                        detections,
                        annotated,
                        overlaps
                    ) = analyze_image(
                        image,
                        st.session_state.selected_model,
                        threshold
                    )

                    st.session_state.analysis = {
                        "scores": scores,
                        "detections": detections,
                        "annotated": annotated,
                        "original": image.copy(),
                        "threshold": threshold,
                        "model": st.session_state.selected_model,
                        "overlaps": overlaps
                    }

                    st.session_state.analyzed_hash = file_hash

                st.success("Analysis completed!")

        except Exception as error:

            st.error(f"Analysis failed: {error}")

    if st.session_state.analysis is not None:

        render_results(
            st.session_state.analysis
        )

    elif uploaded_file is None:

        st.info(
            "Upload an X-ray image to start the analysis."
        )


elif page == "Results":

    st.header("Analysis Results")

    if st.session_state.analysis is None:

        st.info(
            "No analysis available yet. "
            "Analyze an X-ray image first."
        )

    else:

        render_results(
            st.session_state.analysis
        )


elif page == "About":

    st.header("About MedVision-AI")

    st.write(
        "MedVision-AI is an AI-assisted medical imaging "
        "research platform focused on bone-region "
        "classification and potential fracture localization."
    )

    section_title("AI Architecture")

    result_card(
        "Bone Region Classification",
        "ResNet18",
        "Multi-label region prediction"
    )

    result_card(
        "Fracture Detection",
        "YOLO11n V3 / YOLO11s V4",
        "Experimental bounding-box localization"
    )

    st.info(
        "Chest disease analysis is planned "
        "as a future module."
    )


elif page == "Settings":

    st.header("Settings")

    st.markdown("### Fracture Detection Model")

    st.selectbox(
        "Choose the model used for new analyses",
        options=list(FRACTURE_MODELS.keys()),
        key="selected_model"
    )

    selected_path = FRACTURE_MODELS[
        st.session_state.selected_model
    ]

    result_card(
        "Selected Fracture Model",
        st.session_state.selected_model,
        selected_path.name
    )

    result_card(
        "Bone Region Model",
        "ResNet18",
        REGION_MODEL_PATH.name
    )

    result_card(
        "GPU Availability",
        "Available" if torch.cuda.is_available()
        else "Not Available",
        "PyTorch CUDA status"
    )

    if selected_path.exists():

        st.success("Selected model file is available.")

    else:

        st.error(
            f"Selected model file not found: {selected_path}"
        )

    st.info(
        "Changing the selected model affects the next "
        "analysis. Previous results retain their original "
        "model information."
    )


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="mv-footer">
        MedVision-AI | Graduation Project |
        Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
