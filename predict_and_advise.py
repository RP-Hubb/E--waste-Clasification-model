import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google import genai 

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyDnt9E1Op0HtxZAfs5DpGa56SlVSmXDrdQ"
client = genai.Client(api_key=API_KEY)

# Use your absolute path
MODEL_PATH = r'C:/Users/Arnav Sharma/OneDrive/Arnav_Backup/shell_project/ewaste_classifier_model.h5'
model = load_model(MODEL_PATH)

class_names = [
    'Battery', 'Keyboard', 'Microwave', 'Mobile', 
    'Mouse', 'PCB', 'Player', 'Printer', 
    'Television', 'Washing Machine'
]

# --- 2. CNN PREDICTION LOGIC ---
def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    score_index = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    return class_names[score_index], confidence

# --- 3. LLM ADVICE LOGIC WITH RETRY ---
def get_llm_advice(category, retries=3, delay=5):
    """Fetches advice with automatic retry logic and updated model names."""
    prompt = (
        f"I have detected a piece of e-waste categorized as a '{category}'. "
        "Provide a concise 3-step guide on how to safely dispose of it in India "
        "and mention one toxic material found inside this specific device."
    )
    
    for i in range(retries):
        try:
            # UPDATED: Using 'gemini-2.5-flash' as 1.5 is now retired
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            return response.text
        except Exception as e:
            # Check for the 429 Rate Limit error
            if "429" in str(e):
                print(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {i+1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Wait longer each time
            else:
                # If it's still a 404, we catch it here to give a clear message
                if "404" in str(e):
                    return "Error 404: The selected model is unavailable. Ensure you are using 'gemini-2.5-flash'."
                raise e
    return "Error: Could not fetch advice after multiple retries. Please try again later."

# --- 4. EXECUTION ---
test_image_path = r'C:\Shell_Project\keyboard.jpg' 

try:
    detected_item, confidence = predict_waste(test_image_path)
    print(f"\n[CNN RESULT]")
    print(f"Detected: {detected_item} ({confidence:.2f}% confidence)")

    print(f"\n[GEMINI ADVICE FOR {detected_item.upper()}]")
    advice = get_llm_advice(detected_item)
    print(advice)

except Exception as e:
    print(f"An unexpected error occurred: {e}")