
from pathlib import Path
from io import BytesIO
import hashlib
import html

import numpy as np
import streamlit as st
import torch
import timm

from PIL import Image, ImageFile
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# MEDVISION-AI | CONFIGURATION
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent

FRACTURE_MODEL_PATH = ROOT / "models" / "fracture_yolo_v4.pt"
REGION_MODEL_PATH = ROOT / "models" / "bone_region_resnet18.pth"

CHEST_MODEL_PATH = (
    ROOT / "models" / "chest_v4_full"
    / "chest_densenet121_v4_best.pth"
)

# Fixed experimental setting, pending validation optimization.
FRACTURE_THRESHOLD = 0.25

CHEST_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

DISEASE_INFO = {
    "Atelectasis": "Part of the lung may not be fully expanded.",
    "Cardiomegaly": "The heart may appear enlarged.",
    "Consolidation": "An area of the lung may contain fluid or inflammation.",
    "Edema": "There may be excess fluid in the lungs.",
    "Effusion": "Fluid may be present around the lungs.",
    "Emphysema": "The lung air spaces may show chronic changes.",
    "Fibrosis": "There may be scarring in the lung tissue.",
    "Hernia": "An anatomical structure may be displaced.",
    "Infiltration": "There may be an abnormal area in the lung.",
    "Mass": "A larger abnormal opacity may be present.",
    "Nodule": "A small rounded opacity may be present.",
    "Pleural_Thickening": "The lining around the lungs may be thickened.",
    "Pneumonia": "There may be changes associated with lung infection.",
    "Pneumothorax": "Air may be present around a lung.",
}


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="MedVision-AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: #061629;
    color: #eef6ff;
}

[data-testid="stSidebar"] {
    background: #08213a;
    border-right: 1px solid #1b456b;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.hero {
    background: linear-gradient(110deg, #0b3156, #071b30);
    border: 1px solid #1a5889;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 44px;
    font-weight: 850;
    color: white;
}

.hero-title span {
    color: #299eff;
}

.hero-text {
    color: #b8d6ef;
    font-size: 17px;
    margin-top: 8px;
}

.result-box {
    background: #102e4d;
    border: 1px solid #24608c;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}

.result-title {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
}

.result-description {
    font-size: 15px;
    color: #bdd5e9;
    margin-top: 10px;
}

.small-label {
    color: #88b9df;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI HELPERS
# ============================================================

def result_box(title, description="", label="AI FINDING"):
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))
    safe_label = html.escape(str(label))

    card_html = (
        '<div class="result-box">'
        f'<div class="small-label">{safe_label}</div>'
        f'<div class="result-title">{safe_title}</div>'
        f'<div class="result-description">{safe_description}</div>'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def show_hero():
    hero_html = (
        '<div class="hero">'
        '<div class="hero-title">MedVision-<span>AI</span></div>'
        '<div class="hero-text">'
        'Upload an X-ray and view simple AI-assisted findings.'
        '</div>'
        '</div>'
    )

    st.markdown(hero_html, unsafe_allow_html=True)


def get_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    return image, image_hash


# ============================================================
# SESSION STATE
# ============================================================

if "fracture_result" not in st.session_state:
    st.session_state.fracture_result = None

if "chest_result" not in st.session_state:
    st.session_state.chest_result = None

if "fracture_hash" not in st.session_state:
    st.session_state.fracture_hash = None

if "chest_hash" not in st.session_state:
    st.session_state.chest_hash = None

if "page" not in st.session_state:
    st.session_state.page = "Analyze X-ray"


# ============================================================
# FRACTURE MODEL
# ============================================================

@st.cache_resource
def load_fracture_model():
    if not FRACTURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing fracture model: {FRACTURE_MODEL_PATH}"
        )

    model = YOLO(str(FRACTURE_MODEL_PATH))

    if (
        len(model.names) != 1
        or str(model.names[0]).lower() != "fracture"
    ):
        raise ValueError("Unexpected fracture model classes.")

    return model


# ============================================================
# BONE REGION MODEL
# ============================================================

@st.cache_resource
def load_region_model():
    if not REGION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing bone model: {REGION_MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        REGION_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    labels = checkpoint["labels"]

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(labels),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, labels, device


# ============================================================
# CHEST MODEL
# ============================================================

@st.cache_resource
def load_chest_model():
    if not CHEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing chest model: {CHEST_MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use only your own trusted checkpoint.
    checkpoint = torch.load(
        CHEST_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if list(checkpoint["classes"]) != CHEST_CLASSES:
        raise ValueError("Chest class order mismatch.")

    if int(checkpoint["image_size"]) != 320:
        raise ValueError("Chest image size mismatch.")

    thresholds = np.asarray(
        checkpoint["metrics"]["thresholds"],
        dtype=np.float32,
    )

    if (
        thresholds.shape != (14,)
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds < 0)
        or np.any(thresholds > 1)
    ):
        raise ValueError("Invalid chest thresholds.")

    model = timm.create_model(
        "densenet121",
        pretrained=False,
        num_classes=14,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, device, thresholds


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

region_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

chest_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# FRACTURE ANALYSIS
# ============================================================

def analyze_fracture(image):
    fracture_model = load_fracture_model()
    region_model, labels, device = load_region_model()

    tensor = region_transform(
        image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        region_logits = region_model(tensor)
        region_scores = torch.sigmoid(region_logits)[0]

    best_index = int(
        torch.argmax(region_scores).item()
    )

    bone_name = str(labels[best_index])

    results = fracture_model.predict(
        source=image,
        conf=FRACTURE_THRESHOLD,
        imgsz=800,
        device=0 if device.type == "cuda" else "cpu",
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        detections.append({
            "score": float(box.conf[0].item()),
            "box": [
                round(float(value), 1)
                for value in box.xyxy[0].tolist()
            ],
        })

    annotated_bgr = result.plot(
        line_width=3,
        labels=False,
        conf=False,
    )

    annotated = annotated_bgr[:, :, ::-1].copy()

    return {
        "original": image.copy(),
        "annotated": annotated,
        "bone_name": bone_name,
        "detections": detections,
        "fracture_detected": len(detections) > 0,
    }


# ============================================================
# CHEST ANALYSIS
# ============================================================

def analyze_chest(image):
    model, device, thresholds = load_chest_model()

    tensor = chest_transform(
        image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

        probabilities = (
            torch.sigmoid(logits)
            .float()
            .cpu()
            .numpy()[0]
        )

    detected_diseases = []

    for index, name in enumerate(CHEST_CLASSES):
        score = float(probabilities[index])
        threshold = float(thresholds[index])

        if score >= threshold:
            detected_diseases.append({
                "name": name,
                "score": score,
                "description": DISEASE_INFO[name],
            })

    detected_diseases.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "original": image.copy(),
        "diseases": detected_diseases,
    }


# ============================================================
# FRACTURE RESULTS
# ============================================================

def show_fracture_results(result):
    st.divider()
    st.header("Fracture Analysis Result")

    image_col, result_col = st.columns(
        [1.4, 1],
        gap="large",
    )

    with image_col:
        st.subheader("X-ray Image")
        st.image(
            result["annotated"],
            width="stretch",
        )

    with result_col:
        st.subheader("AI Findings")

        bone_name = (
            result["bone_name"]
            .replace("_", " ")
            .title()
        )

        if result["fracture_detected"]:
            result_box(
                "Possible Fracture Detected",
                "The AI highlighted one or more areas "
                "that may contain a fracture.",
                label="FRACTURE FINDING",
            )
        else:
            result_box(
                "No Fracture Region Flagged",
                "The AI did not flag a fracture at "
                "its current detection threshold. "
                "A fracture may still be present.",
                label="FRACTURE FINDING",
            )

        result_box(
            bone_name,
            "Most likely bone region predicted by AI.",
            label="PREDICTED BONE REGION",
        )

    buffer = BytesIO()

    Image.fromarray(
        result["annotated"]
    ).save(
        buffer,
        format="PNG",
    )

    st.download_button(
        "Download Result Image",
        data=buffer.getvalue(),
        file_name="fracture_result.png",
        mime="image/png",
        width="stretch",
    )

    st.caption(
        "The predicted bone region is an image-level "
        "estimate and may not identify the exact bone "
        "within each detection box."
    )

    st.warning(
        "Research prototype only. Results are not "
        "a medical diagnosis. No detection does not "
        "rule out a fracture."
    )


# ============================================================
# CHEST RESULTS
# ============================================================

def show_chest_results(result):
    st.divider()
    st.header("Chest Analysis Result")

    image_col, result_col = st.columns(
        [1.2, 1],
        gap="large",
    )

    with image_col:
        st.subheader("Chest X-ray")
        st.image(
            result["original"],
            width="stretch",
        )

    with result_col:
        st.subheader("AI Findings")

        diseases = result["diseases"]

        if diseases:
            for disease in diseases:
                disease_name = (
                    disease["name"].replace("_", " ")
                )

                result_box(
                    disease_name,
                    disease["description"],
                    label="POSSIBLE CHEST FINDING",
                )
        else:
            result_box(
                "No Findings Flagged",
                "No condition exceeded its validation "
                "threshold. This does not mean the "
                "X-ray is normal.",
                label="CHEST ANALYSIS",
            )

    st.warning(
        "Research prototype only. AI findings are "
        "not confirmed diagnoses. Conditions cannot "
        "be ruled out using this result."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("MedVision-AI")
    st.caption("AI-Assisted X-ray Analysis")
    st.divider()

    if st.button("🏠 Home", width="stretch"):
        st.session_state.page = "Home"
        st.rerun()

    if st.button("🩻 Analyze X-ray", width="stretch"):
        st.session_state.page = "Analyze X-ray"
        st.rerun()

    if st.button("📊 Results", width="stretch"):
        st.session_state.page = "Results"
        st.rerun()

    if st.button("ℹ️ About", width="stretch"):
        st.session_state.page = "About"
        st.rerun()

    st.divider()
    st.caption("MedVision-AI Graduation Project")


# ============================================================
# MAIN APPLICATION
# ============================================================

page = st.session_state.page

if page in ["Home", "Analyze X-ray"]:
    show_hero()

    analysis_type = st.radio(
        "Choose Analysis Type",
        ["Fracture Analysis", "Chest Disease Analysis"],
        horizontal=True,
    )

    if analysis_type == "Fracture Analysis":
        st.subheader("Fracture Analysis")

        uploaded = st.file_uploader(
            "Upload X-ray Image",
            type=["jpg", "jpeg", "png"],
            key="fracture_upload",
        )

        if uploaded is not None:
            try:
                image, image_hash = get_image(uploaded)

                if st.session_state.fracture_hash != image_hash:
                    st.session_state.fracture_result = None

                st.image(
                    image,
                    caption="Uploaded X-ray",
                    width=450,
                )

                if st.button(
                    "Analyze X-ray",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner("Analyzing X-ray..."):
                        result = analyze_fracture(image)

                        st.session_state.fracture_result = result
                        st.session_state.fracture_hash = image_hash

                    st.success("Analysis completed.")

            except Exception as error:
                st.error(f"Analysis error: {error}")

        if st.session_state.fracture_result is not None:
            show_fracture_results(
                st.session_state.fracture_result
            )

    else:
        st.subheader("Chest Disease Analysis")

        uploaded = st.file_uploader(
            "Upload Chest X-ray",
            type=["jpg", "jpeg", "png"],
            key="chest_upload",
        )

        if uploaded is not None:
            try:
                image, image_hash = get_image(uploaded)

                if st.session_state.chest_hash != image_hash:
                    st.session_state.chest_result = None

                st.image(
                    image,
                    caption="Uploaded Chest X-ray",
                    width=450,
                )

                if st.button(
                    "Analyze Chest X-ray",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner("Analyzing chest X-ray..."):
                        result = analyze_chest(image)

                        st.session_state.chest_result = result
                        st.session_state.chest_hash = image_hash

                    st.success("Analysis completed.")

            except Exception as error:
                st.error(f"Analysis error: {error}")

        if st.session_state.chest_result is not None:
            show_chest_results(
                st.session_state.chest_result
            )


# ============================================================
# RESULTS PAGE
# ============================================================

elif page == "Results":
    st.header("Previous Analysis Results")

    fracture_tab, chest_tab = st.tabs([
        "Fracture",
        "Chest",
    ])

    with fracture_tab:
        if st.session_state.fracture_result is not None:
            show_fracture_results(
                st.session_state.fracture_result
            )
        else:
            st.info("No fracture analysis available.")

    with chest_tab:
        if st.session_state.chest_result is not None:
            show_chest_results(
                st.session_state.chest_result
            )
        else:
            st.info("No chest analysis available.")


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":
    st.header("About MedVision-AI")

    st.write(
        "MedVision-AI is a graduation research project "
        "that uses artificial intelligence to analyze "
        "X-ray images."
    )

    st.subheader("Fracture Analysis")
    st.write(
        "YOLO11s V4 highlights potential fracture "
        "regions. ResNet18 predicts the bone region."
    )

    st.subheader("Chest Disease Analysis")
    st.write(
        "DenseNet121 V4 analyzes chest X-rays for "
        "14 possible findings and displays only "
        "those above their validation thresholds."
    )

    st.warning(
        "The application is for research and "
        "educational purposes, not clinical use."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "MedVision-AI | Graduation Project | Research Prototype"
)
