from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Weld Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "weld_yolo26s_v1.pt"


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .status-good {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #2e7d32;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .status-bad {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #c62828;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .status-none {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #777;
        font-size: 20px;
        text-align: center;
    }

    .info-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #555;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Inspection Control")

st.sidebar.markdown("### AI Model")

st.sidebar.success("YOLO26s - Active")

st.sidebar.write("Model Version:")
st.sidebar.code("weld_yolo26s_v1.pt")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Detectable Classes")

st.sidebar.write("""
- Bad Welding
- Crack
- Excess Reinforcement
- Good Welding
- Porosity
- Spatters
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-Based Visual Weld Quality Inspection Prototype"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">AI-Based Weld Quality Inspection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Intelligent Computer Vision System for Automated Weld Defect Detection'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("Weld Inspection")

uploaded_file = st.file_uploader(
    "Upload a weld image for AI inspection",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    with st.spinner("AI is analysing the weld..."):

        results = model.predict(
            source=image_array,
            conf=confidence_threshold,
            verbose=False
        )

    result = results[0]

    # Annotated YOLO image
    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # ORIGINAL VS RESULT
    # ========================================================

    st.subheader("Visual Inspection")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Weld")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### AI Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )


    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            confidence = float(box.conf[0])

            detections.append(
                {
                    "Defect / Classification": class_name,
                    "Confidence (%)": round(
                        confidence * 100,
                        2
                    )
                }
            )


    # ========================================================
    # METRICS
    # ========================================================

    st.divider()

    st.subheader("Inspection Summary")

    if detections:

        df = pd.DataFrame(detections)

        defect_detections = [
            d for d in detections
            if d["Defect / Classification"] != "Good Welding"
        ]

        highest_confidence = max(
            d["Confidence (%)"]
            for d in detections
        )

        unique_classes = len(
            set(
                d["Defect / Classification"]
                for d in detections
            )
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Total Detections",
            len(detections)
        )

        metric2.metric(
            "Abnormalities",
            len(defect_detections)
        )

        metric3.metric(
            "Highest Confidence",
            f"{highest_confidence:.1f}%"
        )

        metric4.metric(
            "Detected Classes",
            unique_classes
        )


        # ====================================================
        # RESULT TABLE
        # ====================================================

        st.markdown("### Detection Details")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # FINAL STATUS
        # ====================================================

        st.markdown("### Final Inspection Status")

        if len(defect_detections) > 0:

            detected_defects = list(
                set(
                    d["Defect / Classification"]
                    for d in defect_detections
                )
            )

            st.markdown(
                """
                <div class="status-bad">
                ⚠ WELD ABNORMALITY DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.write(
                "**Detected abnormalities:**",
                ", ".join(detected_defects)
            )

        else:

            st.markdown(
                """
                <div class="status-good">
                ✓ GOOD WELDING DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # INSPECTION INFORMATION
        # ====================================================

        st.markdown("### Inspection Information")

        info1, info2, info3 = st.columns(3)

        with info1:
            st.write("**Image:**")
            st.write(uploaded_file.name)

        with info2:
            st.write("**Model:**")
            st.write("YOLO26s")

        with info3:
            st.write("**Inspection Time:**")
            st.write(
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
            )


    else:

        st.markdown(
            """
            <div class="status-none">
            No weld classification detected above the selected confidence threshold.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DEFAULT SCREEN
# ============================================================

else:

    st.info(
        "Upload a weld image above to begin AI inspection."
    )

    st.markdown("### System Workflow")

    st.write(
        """
        **1. Upload Weld Image**

        **2. AI Image Analysis**

        **3. Weld Defect Detection**

        **4. Defect Classification**

        **5. Confidence Analysis**

        **6. Inspection Result**
        """
    )