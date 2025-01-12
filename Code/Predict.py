import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('Best_Model.h5')

label = ["E-Waste","Oraganic","Recycleable",]

# Preprocess the image
IMG_SIZE = (128, 128)  # Must match the size used during training
def preprocess_image(image_path):
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Rescale to 0-1 range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

#Streamlit app UI

st.title("Waste Classification App")
st.write("Upload An Image For Classifiacation ")
Upload_File = st.file_uploader("Choose an Image", type = ['jpg','png','jpeg'])

if Upload_File is not None:
    image = load_img(Upload_File)
    st.image(image,"Uploaded Image",use_column_width=True)
    st.write("Classifying...")

    processed_img = preprocess_image(Upload_File)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction[0])


    st.write(f"Predicted Class: {label[predicted_class]}")