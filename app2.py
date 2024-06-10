import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tf_explain.core.grad_cam import GradCAM
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import cv2
import matplotlib.pyplot as plt
import shap


# Load Mobilenet model
@st.cache(allow_output_mutation=True)
def load_custom_model():
    model = tf.keras.models.load_model("/content/drive/MyDrive/bestmodel.h5")
    return model

# Load VGG16 model
@st.cache(allow_output_mutation=True)
def load_custom_model1():
    model = tf.keras.models.load_model("/content/drive/MyDrive/bestmodel1.h5")
    return model

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to model input size
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply model-specific preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

#VGG16
# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap1(img, img_array, model1, class_index, layer_name):
    predictions = model1.predict(img_array)
    predicted_class = np.argmax(predictions)

    pred = model1.predict(img_array)
    prediction_result = "Not Infected" if pred > 0.5 else "Infected"
    explainer = GradCAM()
    grid = explainer.explain((img_array, None), model1, class_index=predicted_class, layer_name=layer_name)  # Adjust layer_name if needed
    return cv2.resize(grid, (img.width, img.height)), prediction_result

# Function to explain and predict image using Lime
def explain_and_predict_image1(img, model, class_names, top_labels=1):
    explainer = lime_image.LimeImageExplainer()

    # Resize the image to match the expected input shape of the model
    img_resized = tf.image.resize(img, (224, 224))

    i = img_resized / 255
    input_arr = np.array([i])

    # Define the segmentation function
    segmentation_fn = SegmentationAlgorithm('slic', n_segments=150, compactness=1, sigma=1)

    # Create the Lime explanation
    explanation = explainer.explain_instance(input_arr[0], model.predict, top_labels=top_labels,
                                             hide_color=0, num_samples=1000, segmentation_fn=segmentation_fn)

    # Get the image and mask for the top predicted label
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    
    # Predict the image
    pred = model.predict(input_arr)
    prediction_result = "Not Infected" if pred > 0.5 else "Infected"
    
    # Calculate fidelity
    fidelity = explanation.score

    # Calculate coverage
    coverage = len(explanation.intercept) / len(input_arr)

    # Calculate interpretability score based on visualization quality (Example: Mean number of features used in explanations)
    interpretability_score = np.mean([len(exp) for exp in explanation.local_exp.values()])

    return temp, mask, prediction_result, fidelity, coverage, interpretability_score

# Function to explain and predict image with SHAP
def explain_and_predict_image_shap1(img_array, model):
    explainer = shap.GradientExplainer(model, np.random.rand(100, 224, 224, 3))  # Background data
    shap_values, indexes = explainer.shap_values(img_array, ranked_outputs=1)
    pred = model.predict(img_array)
    prediction_result = "Not Infected" if pred > 0.5 else "Infected"
    return shap_values, indexes, prediction_result

#Mobilenet
# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(img, img_array, model, class_index, layer_name):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    pred = model.predict(img_array)
    prediction_result = "Not Infected" if pred > 0.5  else "Infected"
    explainer = GradCAM()
    grid = explainer.explain((img_array, None), model, class_index=predicted_class, layer_name=layer_name)  # Adjust layer_name if needed
    return cv2.resize(grid, (img.width, img.height)), prediction_result

# Function to explain and predict image using Lime
def explain_and_predict_image(img, model, class_names, top_labels=1):
    explainer = lime_image.LimeImageExplainer()

    # Resize the image to match the expected input shape of the model
    img_resized = tf.image.resize(img, (224, 224))

    i = img_resized / 255
    input_arr = np.array([i])

    # Define the segmentation function
    segmentation_fn = SegmentationAlgorithm('slic', n_segments=150, compactness=1, sigma=1)

    # Create the Lime explanation
    explanation = explainer.explain_instance(input_arr[0], model.predict, top_labels=top_labels,
                                             hide_color=0, num_samples=1000, segmentation_fn=segmentation_fn)

    # Get the image and mask for the top predicted label
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    
    # Predict the image
    pred = model.predict(input_arr)
    prediction_result = "Not Infected" if pred > 0.5  else "Infected"
    
    # Calculate fidelity
    fidelity = explanation.score

    # Calculate coverage
    coverage = len(explanation.intercept) / len(input_arr)

    # Calculate interpretability score based on visualization quality (Example: Mean number of features used in explanations)
    interpretability_score = np.mean([len(exp) for exp in explanation.local_exp.values()])

    return temp, mask, prediction_result, fidelity, coverage, interpretability_score

# Function to explain and predict image with SHAP
def explain_and_predict_image_shap(img_array, model):
    explainer = shap.GradientExplainer(model, np.random.rand(100, 224, 224, 3))  # Background data
    shap_values, indexes = explainer.shap_values(img_array, ranked_outputs=1)
    pred = model.predict(img_array)
    prediction_result = "Not Infected" if pred > 0.5  else "Infected"
    return shap_values, indexes, prediction_result

# home function
def app():
    st.markdown('<h1 style="color: #8793E1;">PCOS Ultrasound Image Classification</h1>', unsafe_allow_html=True)
    #st.title("PCOS Ultrasound Image Classification")
    # Sidebar for explanation method selection
    st.sidebar.title("Select Model")
    selected_model = st.sidebar.selectbox("", ["Mobilenet", "VGG16"])

    # Sidebar for explanation method selection
    st.sidebar.title("Explanation Method")
    explanation_method = st.sidebar.selectbox("", ["Grad-CAM", "LIME", "SHAP"])

    # Upload image
    uploaded_file = st.file_uploader("Upload PCOS ultrasound image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Load and preprocess image
        img, img_array = load_and_preprocess_image(uploaded_file)
        
        # Load Mobilenet model
        model = load_custom_model()

        # Load VGG16 model
        model1=load_custom_model1()

        # Switch case based on selected explanation method
        if explanation_method == "Grad-CAM" and selected_model == "Mobilenet":
            # Execute Grad-CAM explanation
            heatmap, prediction_result = generate_gradcam_heatmap(img, img_array, model, 1, 'conv_pw_13_relu')  # Adjust class_index and layer_name if needed
            st.image(heatmap, caption="Grad-CAM Heatmap using Mobilenet", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")

        elif explanation_method == "SHAP" and selected_model == "Mobilenet":
            # Execute Lime explanation
            shap_values, indexes ,prediction_result= explain_and_predict_image_shap(img_array, model)

            # Plot the SHAP values
            fig, ax = plt.subplots()
            shap.image_plot(shap_values, -img_array, show=False)
            plt.savefig("shap_plot.png")

            # Display the SHAP plot
            st.image("shap_plot.png", caption="SHAP Values using Mobilenet", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")

        elif explanation_method == "LIME" and selected_model == "Mobilenet":
            # Execute Lime explanation
            explanation_img, mask, prediction_result, fidelity, coverage, interpretability = explain_and_predict_image(img, model, class_names=None)
            st.image(explanation_img, caption="LIME Explanation using Mobilenet", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")
            st.write(f"Fidelity: {fidelity}")
            st.write(f"Instance Coverage: {coverage}")
            st.write(f"Interpretability: {interpretability}")

        # Switch case based on selected explanation method
        if explanation_method == "Grad-CAM" and selected_model == "VGG16":
            # Execute Grad-CAM explanation
            heatmap, prediction_result = generate_gradcam_heatmap1(img, img_array, model1, 1, 'block5_pool')  # Adjust class_index and layer_name if needed
            st.image(heatmap, caption="Grad-CAM Heatmap using VGG16", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")

        elif explanation_method == "SHAP" and selected_model == "VGG16":
            # Execute Lime explanation
            shap_values, indexes ,prediction_result= explain_and_predict_image_shap1(img_array, model1)

            # Plot the SHAP values
            fig, ax = plt.subplots()
            shap.image_plot(shap_values, -img_array, show=False)
            plt.savefig("shap_plot1.png")

            # Display the SHAP plot
            st.image("shap_plot1.png", caption="SHAP Values using VGG16", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")

        elif explanation_method == "LIME" and selected_model == "VGG16":
            # Execute Lime explanation
            explanation_img, mask, prediction_result, fidelity, coverage, interpretability = explain_and_predict_image1(img, model1, class_names=None)
            st.image(explanation_img, caption="LIME Explanation using VGG16", use_column_width=True)
            st.write(f"Prediction: {prediction_result}")
            st.write(f"Fidelity: {fidelity}")
            st.write(f"Instance Coverage: {coverage}")
            st.write(f"Interpretability: {interpretability}")
if __name__ == "__main__":
    app()