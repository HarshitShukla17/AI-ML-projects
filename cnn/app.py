import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("keras_cifar10_model.h5")

# Define class labels (adjust based on your dataset)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit App
st.title("Image Classification with CNN")
st.write("Upload an image to classify using the trained CNN model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Processing the image...")
    image = image.resize((32, 32))  # Resize to match the model's input shape
    image_array = img_to_array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions
    
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest score
    confidence = np.max(predictions[0])  # Get the highest score as confidence

    # Display results
    
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")

# # Display first 25 images from the training set
# st.write("Displaying the first 25 images from the training set:")

# # Assuming you have `train_images` and `train_labels` loaded in your code
# # For demonstration, let's create some sample images and labels
# # Replace this with your actual train data
# train_images = np.random.rand(25, 32, 32, 3)  # 25 random images of size 32x32x3
# train_labels = np.random.randint(0, 10, size=(25, 1))  # 25 random labels (0 to 9)

# # Plot the first 25 images with class labels
# fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# for i in range(25):
#     ax = axes[i // 5, i % 5]
#     ax.imshow(train_images[i])  # Display image
#     ax.set_xticks([])  # No ticks
#     ax.set_yticks([])  # No ticks
#     ax.set_title(class_names[train_labels[i][0]])

# # Display the plot in Streamlit
# st.pyplot(fig)
