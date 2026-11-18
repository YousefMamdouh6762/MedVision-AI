
from pathlib import Path
import io
import html

import streamlit as st
import torch
from PIL import Image, ImageFile
from torchvision import models, transforms
from ultralytics import YOLO


# =====================================
# Configuration
# =====================================

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent
REGION_MODEL_PATH = ROOT / "models" / "bone_region_resnet18.pth"
FRACTURE_MODEL_PATH = ROOT / "models" / "fracture_yolo_v3.pt"

st.set_page_config(
    page_title="MedVision-AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================
# Professional Dashboard Styling
# =====================================

st.markdown("""
<style>
    .stApp {
        background: #071426;
        color: #eaf2ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2039, #071426);
        border-right: 1px solid #19395c;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f0f7ff !important;
    }

    h1 {
        font-size: 2.3rem !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #102a47, #0c1c32);
        border: 1px solid #22517b;
        border-radius: 16px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9c4df;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    [data-testid="stFileUploader"] {
        background: #10243d;
        border: 1px solid #28527c;
        border-radius: 15px;
        padding: 15px;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0877e8, #2196f3);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 3.2rem;
        font-weight: 700;
    }

    .stButton > button[kind="primary"]:hover {
        background: #0065c9;
        color: white;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #23496e;
    }

    .hero {
        background: linear-gradient(110deg, #102d4f, #071a30);
        border: 1px solid #22517b;
        border-radius: 18px;
        padding: 27px 30px;
        margin-bottom: 22px;
    }

    .hero-kicker {
        color: #46b1ff;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero-title {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 10px 0;
    }

    .hero-description {
        color: #b5cbe2;
        font-size: 0.98rem;
    }

    .section-title {
        color: #f2f7ff;
        font-size: 1.2rem;
        font-weight: 750;
        margin: 18px 0 12px;
    }

    .small-card {
        background: #10243d;
        border: 1px solid #23496e;
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }

    .small-card-title {
        color: #8eb9e0;
        font-size: 0.82rem;
    }

    .small-card-value {
        color: white;
        font-size: 1.3rem;
        font-weight: 750;
    }

    [data-testid="stProgress"] > div > div {
        background-color: #198ef1;
    }

    .disclaimer {
        background: #0b2b40;
        border: 1px solid #1777ad;
        border-radius: 12px;
        padding: 14px;
        color: #b9d8ed;
        font-size: 0.88rem;
    }

    hr {
        border-color: #254260 !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# AI Models
# =====================================

@st.cache_resource
def load_models():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        REGION_MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    labels = checkpoint["labels"]

    region_model = models.resnet18(weights=None)
    region_model.fc = torch.nn.Linear(
        region_model.fc.in_features,
        len(labels)
    )

    region_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    region_model = region_model.to(device)
    region_model.eval()

    fracture_model = YOLO(str(FRACTURE_MODEL_PATH))

    return region_model, fracture_model, labels, device


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def analyze_image(image, region_model, fracture_model,
                  labels, device, threshold):

    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = region_model(image_tensor)
        probabilities = torch.sigmoid(outputs)[0]

    scores = sorted(
        zip(labels, probabilities.cpu().tolist()),
        key=lambda item: item[1],
        reverse=True
    )

    results = fracture_model.predict(
        source=image,
        conf=threshold,
        imgsz=640,
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
        coords = [
            round(float(value), 1)
            for value in box.xyxy[0].tolist()
        ]
        detections.append({
            "score": float(box.conf[0].item()),
            "box": coords
        })

    return scores, detections, annotated_rgb


# =====================================
# Session State
# =====================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "analyzed_file" not in st.session_state:
    st.session_state.analyzed_file = None


# =====================================
# Sidebar
# =====================================

with st.sidebar:
    st.title("🩻 MedVision-AI")
    st.caption("AI-Assisted X-ray Analysis")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🩻 Analyze X-ray",
            "📊 Results",
            "ℹ️ About",
            "⚙️ Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Research Prototype • YOLO V3 + ResNet18")
    st.warning(
        "Not for clinical diagnosis or treatment decisions."
    )


# =====================================
# Dashboard Header
# =====================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">BONE FRACTURE ANALYSIS</div>
    <div class="hero-title">AI-Powered X-ray Analysis</div>
    <div class="hero-description">
        Explore potential fracture regions and bone-region
        model predictions using your trained AI models.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================
# Main Analysis Page
# =====================================

if page in ["🏠 Home", "🩻 Analyze X-ray"]:

    st.markdown(
        '<div class="section-title">Upload & Analyze</div>',
        unsafe_allow_html=True
    )

    upload_col, settings_col = st.columns([2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload an X-ray image",
            type=["jpg", "jpeg", "png"]
        )

    with settings_col:
        threshold = st.slider(
            "Detection score threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # Clear old results when a different image is uploaded.
        if st.session_state.analyzed_file != file_bytes:
            st.session_state.analysis = None

        try:
            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            if st.button(
                "🧠 Analyze X-ray",
                type="primary",
                width="stretch"
            ):
                with st.spinner("Running AI analysis..."):
                    region_model, fracture_model, labels, device = (
                        load_models()
                    )

                    scores, detections, annotated = analyze_image(
                        image,
                        region_model,
                        fracture_model,
                        labels,
                        device,
                        threshold
                    )

                    st.session_state.analysis = {
                        "scores": scores,
                        "detections": detections,
                        "annotated": annotated,
                        "original": image.copy(),
                        "threshold": threshold
                    }

                    st.session_state.analyzed_file = file_bytes

                st.success("Analysis completed successfully!")

        except Exception as error:
            st.error(f"Analysis failed: {error}")

    if st.session_state.analysis is not None:

        analysis = st.session_state.analysis
        scores = analysis["scores"]
        detections = analysis["detections"]

        best_region, best_score = scores[0]

        st.markdown(
            '<div class="section-title">Analysis Overview</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Most likely bone region",
                best_region.title()
            )
            st.caption(
                f"Region model score: {best_score * 100:.2f}%"
            )

        with metric2:
            st.metric(
                "Potential fracture detections",
                len(detections)
            )

        with metric3:
            st.metric(
                "Detection threshold",
                f"{analysis['threshold']:.2f}"
            )

        st.markdown(
            '<div class="section-title">X-ray Comparison</div>',
            unsafe_allow_html=True
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:
            st.image(
                analysis["original"],
                caption="Original X-ray",
                width="stretch"
            )

        with image_col2:
            st.image(
                analysis["annotated"],
                caption="AI Detection Result",
                width="stretch"
            )

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown(
                '<div class="section-title">Bone Region Scores</div>',
                unsafe_allow_html=True
            )

            for label, score in scores:
                st.write(
                    f"**{label.title()}** — {score * 100:.2f}%"
                )
                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

        with detail_col2:
            st.markdown(
                '<div class="section-title">Detection Details</div>',
                unsafe_allow_html=True
            )

            if detections:
                for index, detection in enumerate(
                    detections, start=1
                ):
                    score = detection["score"] * 100
                    coords = detection["box"]

                    st.markdown(
                        f"""
                        <div class="small-card">
                            <div class="small-card-title">
                                Potential Region {index}
                            </div>
                            <div class="small-card-value">
                                Score: {score:.2f}%
                            </div>
                            <div class="small-card-title">
                                Box: {html.escape(str(coords))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info(
                    "No potential fracture regions detected "
                    "at this threshold."
                )

        st.markdown(
            """
            <div class="disclaimer">
                <b>Research interpretation:</b>
                A detection box does not confirm a fracture.
                No detection does not rule one out.
                Scores are model outputs, not calibrated
                medical probabilities.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif uploaded_file is None:
        st.info(
            "Upload an X-ray image to start the analysis."
        )


# =====================================
# Results Page
# =====================================

elif page == "📊 Results":

    st.header("Analysis Results")

    if st.session_state.analysis is None:
        st.info(
            "No analysis available yet. "
            "Go to Analyze X-ray and run an analysis."
        )
    else:
        analysis = st.session_state.analysis

        st.image(
            analysis["annotated"],
            caption="Latest AI Detection Result",
            width=650
        )

        st.subheader("Bone Region Scores")

        for label, score in analysis["scores"]:
            st.write(
                f"{label.title()}: {score * 100:.2f}%"
            )

        st.subheader("Potential Fracture Detections")
        st.write(len(analysis["detections"]))


# =====================================
# About Page
# =====================================

elif page == "ℹ️ About":

    st.header("About MedVision-AI")

    st.write(
        "MedVision-AI is an AI-assisted medical imaging "
        "research project designed to explore bone-region "
        "classification and potential fracture localization "
        "in X-ray images."
    )

    st.subheader("AI Architecture")

    st.write("**Bone Region Model:** ResNet18")
    st.write("**Fracture Detection Model:** YOLO11n V3")
    st.write("**Framework:** PyTorch")
    st.write("**Interface:** Streamlit")

    st.info(
        "A chest disease analysis module is planned "
        "for a future development stage."
    )


# =====================================
# Settings Page
# =====================================

elif page == "⚙️ Settings":

    st.header("Settings")

    st.write("Current application configuration")

    st.write("**Fracture Model:** fracture_yolo_v3.pt")
    st.write("**Bone Region Model:** bone_region_resnet18.pth")

    st.write(
        "**GPU available:**",
        "Yes" if torch.cuda.is_available() else "No"
    )

    st.caption(
        "Detection threshold can be adjusted "
        "on the analysis page."
    )


# =====================================
# Footer
# =====================================

st.divider()

st.caption(
    "MedVision-AI | Graduation Project | Research Prototype"
)
