import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np
import cv2
import torch.nn as nn
from grad_cam.grad_cam import GradCAM

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI DOC – Pneumonia Detection",
    layout="wide"
)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("<h1 style='color:#F5C518;'>AI DOC</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #F5C518;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='color: white;'>
        This tool uses a Deep Learning model to detect **Pneumonia** in chest X-ray images.<br><br>
        <b>Model:</b> DenseNet121 (Transfer Learning)<br>
        <b>Explanation:</b> Grad-CAM Visualizer
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<hr style='border: 1px solid #F5C518;'>", unsafe_allow_html=True)
    st.caption("© 2025 Medical AI Lab")

# -------------------- Model Load --------------------
@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

# -------------------- Transforms --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- Header --------------------
st.markdown(
    "<h1 style='color:#F5C518; font-size: 36px;'>Pneumonia Detection from Chest X-rays</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 1px solid #F5C518;'>", unsafe_allow_html=True)

# -------------------- Upload --------------------
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    with st.spinner("Analyzing the image..."):
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        class_names = ['Normal', 'Pneumonia']
        diagnosis = class_names[pred_class]

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#1c1c1c; padding: 24px; border-radius: 10px;">
                <h3 style="color:#F5C518;">Diagnosis Result</h3>
                <p style="color:white; font-size:18px;">Prediction: <strong>{diagnosis}</strong></p>
                <p style="color:white; font-size:18px;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if diagnosis == "Pneumonia":
            grad_cam = GradCAM(model, target_layer_name="features.denseblock4")
            cam = grad_cam.generate(input_tensor, class_idx=pred_class)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img_resized = np.array(img.resize((224, 224)))
            superimposed_img = heatmap * 0.4 + img_resized * 0.6
            superimposed_img = superimposed_img / superimposed_img.max() * 255
            superimposed_img = superimposed_img.astype(np.uint8)

            st.markdown("<h4 style='color:#F5C518;'>Grad-CAM Heatmap Overlay</h4>", unsafe_allow_html=True)
            st.image(superimposed_img, use_container_width=True)
            grad_cam.remove_hooks()
