import streamlit as st
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google import genai
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="PricePal E-Waste Advisor", layout="centered", page_icon="♻️")
st.title("Smart E-Waste Classifier and Advisor")
st.markdown("Upload a photo of your electronic waste to get instant classification and recycling advice.")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # 1. Load the CNN Brain
    model_path = r'C:\Users\Raj\Desktop\Shell project\E-Waste Classification\ewaste_classifier_model.h5'
    model = load_model(model_path)
    
    # 2. Setup Gemini Client
    # Using your active key ending in ...DrdQ
    api_key = "AIzaSyDnt9E1Op0HtxZAfs5DpGa56SlVSmXDrdQ" 
    client = genai.Client(api_key=api_key)
    
    return model, client

model, client = load_assets()

class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# --- UPDATED ADVICE FUNCTION WITH RETRY LOGIC ---
def get_llm_advice(category, retries=3, delay=5):
    """Fetches advice with automatic retry logic if the free tier limit is hit."""
    prompt = (
        f"I have detected a piece of e-waste: {category}. "
        "Provide a concise 3-step guide for safe disposal in Ahmedabad, India "
        "and name one toxic material found inside this specific device."
    )
    
    for i in range(retries):
        try:
            # UPDATED: Using 'gemini-2.5-flash' as 1.5 and 2.0 are retired
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e): # Rate limit error
                st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 # Wait longer for the next retry
            else:
                return f"An error occurred: {e}"
    return "Error: Could not fetch advice. Please wait a minute and try again."

# --- UI: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Run CNN Prediction
    with st.spinner('Analyzing with CNN...'):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        preds = model.predict(img_array)
        label = class_names[np.argmax(preds[0])]
        conf = np.max(preds[0]) * 100
        
    st.success(f"Detected: **{label}** ({conf:.2f}% Confidence)")
    
    # Run LLM Advice
    st.divider()
    st.subheader(f"💡 Recycling Advice for {label}")
    
    with st.spinner('Generating expert advice...'):
        advice = get_llm_advice(label)

        st.info(advice)

