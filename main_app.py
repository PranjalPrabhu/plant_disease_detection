import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from keras.models import load_model

model = load_model('C:\\archive\\Streamlit app\\model.h5')

CLASS_NAMES = ['Tomato___Bacterial_spot', 'Potato___Early_blight', 'Corn_(maize)___Common_rust_']

st.title("Plant Leaf Disease Detection")
st.markdown("Upload an image of plant leaf")

plant_image = st.file_uploader("Choose an image...", type='jpg')
submit = st.button("Predict Disease")

if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image.shape = (1, 256, 256, 3)

        y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(y_pred)]

        # Split result safely
        result_parts = result.split('___')
        if len(result_parts) == 2:
            plant, disease = result_parts
            st.title(f"This is a {plant} leaf with {disease}")
        else:
            st.title(f"Disease detected: {result}")
