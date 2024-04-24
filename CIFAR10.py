import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model(r'D:\New folder (2)\cifar10_model.h5') #D:\New folder (2)\cifar10_model.h5

# Create a dictionary of class labels
class_labels = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class, class_labels[predicted_class]

def main():
    st.title('CIFAR-10 Image Classification')
    st.write('Upload an image to predict its class.')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Make a prediction
        predicted_class_index, predicted_class = predict(uploaded_file)
        
        # Show the prediction
        st.write(f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
    main()
