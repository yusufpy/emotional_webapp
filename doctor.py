import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('eion_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Define a function to predict the emotion
def predict_emotion(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    label_to_text = {0: "Surprise", 1: "Fear", 2: "Angry", 3: "Neutral", 4: "Sad", 5: "Disgust", 6: "Happy"}
    predicted_label = np.argmax(prediction)
    ##
    predicted_label = int(str(predicted_label)[0])
    if predicted_label>6:
        predicted_label=3
    ##
    predicted_emotion = label_to_text[predicted_label]
    return predicted_emotion

# Streamlit app
def app():
    st.title("Emotion Detection App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")
        predicted_emotion = predict_emotion(image)
        st.write(f"Predicted emotion: {predicted_emotion}")

if __name__ == "__main__":
    app()