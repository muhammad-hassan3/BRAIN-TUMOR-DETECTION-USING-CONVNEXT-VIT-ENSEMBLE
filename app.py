import streamlit as st
import sys
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 1. This MUST be the first command. It initializes the UI.
st.set_page_config(page_title="Brain Tumor AI", layout="wide")


st.title("🧠 Brain Tumor Diagnostic System")
st.info("System is starting... Please wait.")

# 2. Define the loading function (Imports happen HERE, not at the top)
@st.cache_resource
def load_model_safely():
    status_text = st.empty()
    status_text.text("Importing PyTorch libraries... (This may take a moment)")
    
    try:
        import torch
        import torch.nn as nn
        import timm
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        status_text.text("Libraries loaded. Initializing Model architecture...")
        
        # Define the class inside to ensure it's available
        class HybridTumorModel(nn.Module):
            def __init__(self, num_classes=4):
                super(HybridTumorModel, self).__init__()
                self.convnext = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
                self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)

            def forward(self, x):
                out1 = self.convnext(x)
                out2 = self.vit(x)
                return (out1 + out2) / 2
        
        status_text.text("Loading trained weights from file...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridTumorModel(num_classes=4)
        
        # Load the weights
        model_path = 'tumor_classifier_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        status_text.empty() # Clear the status message
        return model, device, transforms, Image, GradCAMPlusPlus, ClassifierOutputTarget, show_cam_on_image, np

    except Exception as e:
        st.error(f"❌ Critical Error loading model: {e}")
        st.stop()
        return None

# 3. Actually load the model now
model_data = load_model_safely()

# If loading failed, stop here
if model_data is None:
    st.stop()

# Unpack the loaded libraries
model, device, transforms, Image, GradCAMPlusPlus, ClassifierOutputTarget, show_cam_on_image, np = model_data

st.success("System Ready! Upload an MRI scan.")

# --- 4. Main App Interface ---
uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display Original
    image = Image.open(uploaded_file).convert('RGB')
    col1.image(image, caption="Uploaded Scan", width="stretch")
    
    if st.button("Analyze Tumor"):
        with st.spinner("Processing..."):
            # Prepare Image
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = data_transforms(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                pred_class = class_names[predicted.item()]
            
            # Grad-CAM
            target_layers = [model.convnext.stages[-1].blocks[-1]]
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted.item())])
            grayscale_cam = grayscale_cam[0, :]
            
            # Visualize
            rgb_img = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Show Results
            col2.image(visualization, caption=f"AI Focus Area ({pred_class})", use_container_width=True)
            st.markdown(f"### Prediction: **{pred_class.upper()}**")
            st.progress(float(confidence.item()))
            st.caption(f"Confidence: {confidence.item()*100:.2f}%")