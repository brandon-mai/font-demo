import streamlit as st
from streamlit_cropperjs import st_cropperjs
import torch
import numpy as np
from PIL import Image
import io
from predict_font import predict_font
from font_list import FONT_LIST, MYFONT_RENDER

from nets.alexnet import AlexNetClassifier
from nets.henet import HENet
from nets.fontclassifier import FontClassifier
from nets.scae import SCAE


st.set_page_config(
   page_title="Font Recognition",
   page_icon=":sparkles:",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Initialize session state for models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_classes = len(FONT_LIST)
    
    # Load AlexNet model
    try:
        alexnet = AlexNetClassifier(num_classes=num_classes)
        alexnet.load_state_dict(torch.load('checkpoints/alexnet20.pt', weights_only=True, map_location=device))
        models['AlexNet'] = alexnet
        # st.success("✅ AlexNet model loaded successfully")
    except Exception as e:
        pass
        # st.error(f"❌ Failed to load AlexNet: {e}")
    
    # Load HENet model
    try:
        henet = HENet(num_classes=num_classes)
        henet.load_state_dict(torch.load('checkpoints/henet_final_model_epoch_5.pt', weights_only=True, map_location=device))
        models['HENet'] = henet
        # st.success("✅ HENet model loaded successfully")
    except Exception as e:
        pass
        # st.error(f"❌ Failed to load HENet: {e}")
    
    # Load FontClassifier model
    try:
        pretrained_scae = SCAE()
        pretrained_scae.load_state_dict(torch.load('checkpoints/scae_2nd_model_epoch_3.pt', weights_only=True, map_location=device))
        fontclassifier = FontClassifier(pretrained_scae, num_classes=num_classes)
        fontclassifier.load_state_dict(torch.load('checkpoints/best_font_model.pt', weights_only=True, map_location=device))
        models['FontClassifier'] = fontclassifier
        # st.success("✅ FontClassifier model loaded successfully")
    except Exception as e:
        pass
        # st.error(f"❌ Failed to load FontClassifier: {e}")
    
    return models

# Load models
models = load_models()

with st.sidebar:
    option = st.selectbox(
        "I want to...",
        ("Take a picture", "Upload a picture"),
        index=0,
    )

    if option == "Take a picture":
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
    else:
        picture = st.file_uploader("Upload a picture")

    if picture:
        pic = picture.read()
        cropped_pic = st_cropperjs(pic=pic, btn_text="✂️ Crop Image", key="cropper")
        if cropped_pic:
            st.session_state['cropped_image'] = cropped_pic

st.title("Font Recognition")

# Model selection
if models:
    selected_model = st.selectbox(
        "🎯 Select AI Model:",
        list(models.keys()),
        index=1,
        help="Choose which trained model to use for prediction"
    )
    
    # # Model info
    # model_info = {
    #     'AlexNet': 'Basic CNN based on AlexNet architecture, trained for 20 epochs.',
    #     'HENet': 'Hide and Enhance Network.',
    #     'FontClassifier': 'Flagship model, CNN with frozen encoder from SCAE.'
    # }
    
    # if selected_model in model_info:
    #     st.info(f"ℹ️ {model_info[selected_model]}")
else:
    st.error("❌ No models available. Please check model files.")
    selected_model = None

input_col, pred_col = st.columns([1, 2], gap="medium")

with input_col:
    st.subheader("Input Image")

    if 'cropped_image' in st.session_state and st.session_state['cropped_image']:
        cropped_image = st.session_state['cropped_image']
    else:
        cropped_image = None
        st.warning("Please take or upload a picture to crop.")

    if cropped_image:
        # Display the cropped image
        st.image(cropped_image, caption="Cropped Image", use_container_width=True)
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(cropped_image))
        image = image.convert("RGB")

with pred_col:
    st.subheader("Prediction")
    
    if cropped_image and models and selected_model:
        if st.button("🔍 Find Similar Fonts", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing font..."):
                try:
                    # Convert PIL Image to numpy array for predict_font function
                    img_array = np.array(image)
                    
                    # Get prediction using the selected model
                    model = models[selected_model]
                    prediction_probs, predicted_class = predict_font(model, img_array)
                    
                    # Create list of (font_name, probability) tuples
                    font_predictions = []
                    for i, prob in enumerate(prediction_probs):
                        font_name = FONT_LIST[i]
                        font_id = MYFONT_RENDER.get(font_name, None)
                        font_predictions.append((font_name, font_id, float(prob)))
                    
                    # Sort by probability (descending)
                    font_predictions.sort(key=lambda x: x[-1], reverse=True)

                    preview_text = "Bogos binted"
                    
                    # Display top 5 predictions
                    for i, (font_name, font_id, prob) in enumerate(font_predictions[:10]):
                        with st.container():
                            st.markdown(f"**{i+1}. {font_name}** - {prob}")

                            if font_id:
                                st.image(
                                    f"https://render.myfonts.net/fonts/font_rend.php?id={font_id}&rt={'%20'.join(preview_text.split())}",
                                    use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)  # This will show the full traceback for debugging
    
    elif not cropped_image:
        st.info("👆 Please crop an image first to enable font prediction")
        st.markdown("""
        **Steps to get started:**
        1. Take a picture or upload an image
        2. Crop the text area you want to analyze  
        3. Select your preferred AI model
        4. Click 'Find Similar Fonts'
        """)
    
    elif not models:
        st.error("❌ No models loaded. Please check your model files in the 'checkpoints' folder.")
    
    elif not selected_model:
        st.warning("⚠️ Please select a model to proceed with prediction.")