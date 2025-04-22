import io
import requests
import base64
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from llama_index.llms.ollama import Ollama

# Set up DeepSeek-R1
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=360.0)

# UI
st.title("🧴 DermAI – Skin Health Assistant")
st.markdown("Describe your skin problem and upload an image.")

# Conversation context
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("🗣 Describe your symptoms:")
uploaded_image = st.file_uploader("📷 Upload a skin image:", type=["jpg", "jpeg", "png"])

# Image preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# Image analysis placeholder
def analyze_image(img):
    # Convert image to bytes
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # API request to Hugging Face
    api_url = "https://api-inference.huggingface.co/models/sagarvk24/skin-disorders-classifier"
    headers = {
        "Authorization": "Bearer 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",  # Replace with your token
        "Content-Type": "image/jpeg"
    }

    response = requests.post(api_url, headers=headers, data=img_bytes)

    # Handle the response
    if response.status_code == 200:
        prediction = response.json()
        label = prediction[0]['label']
        score = round(prediction[0]['score'] * 100, 2)
        return f"Prediction: **{label}** ({score}%)", label
    else:
        return f"❌ Error: {response.status_code} - {response.text}", "Unknown"


def get_ai_response(user_input, prediction_label):
    prompt = f"""
You are a friendly skin health assistant. A user described their skin symptoms as: "{user_input}" and uploaded an image.

The image was analyzed and predicted as: "{prediction_label}".

1. Suggest whether a visit to a dermatologist is necessary based on the prediction.
2. Give simple at-home remedies or skincare tips (if safe to do so).
3. Keep the tone helpful and clear. Avoid medical jargon.
"""
    response = llm.complete(prompt)
    return response.text.strip()

#main logic
if st.button("Analyze"):
    prediction = "No image provided"
    pred_label = "Unknown"

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image).convert("RGB")
        prediction, pred_label = analyze_image(image)
        st.markdown("### 🧠 AI Diagnosis Result:")
        st.write(prediction)

    # Use fallback input for LLM
    text_for_llm = user_input if user_input.strip() else "No text symptoms were provided. Only an image was uploaded."

    response = get_ai_response(text_for_llm, pred_label)
    st.markdown("### 🤖 DermAI Chatbot:")
    st.write(response)
