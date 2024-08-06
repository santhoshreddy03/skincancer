import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import plotly.graph_objects as go


st.set_page_config(
    page_title="Advanced Skin Cancer Detection App",
    page_icon="🩺",
    layout="wide",
)

# Load your trained model
@st.cache_resource
def load_classifier_model():
    return load_model('skin_cancer_classifier.h5')

model = load_classifier_model()

# Define the function to predict skin cancer
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]


def create_gauge_chart(prediction):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'cyan'},
                {'range': [50, 100], 'color': 'royalblue'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    fig.update_layout(height=300)
    return fig

# Main function for the Streamlit app
def main():
    st.title("Advanced Skin Cancer Detection App")
    st.markdown("## Developed by Padala Santosh Reddy")
    st.markdown("For inquiries, please contact: [padalasantosh.reddy2021@vitstudent.ac.in](mailto:padalasantosh.reddy2021@vitstudent.ac.in)")
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: None;
            padding: 10px 24px;
            text-align: center;
            text-decoration: None;
            display: inline-block;
            font-size: 16px;
            border-radius: 12px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .prediction-box h2 {
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application utilizes a state-of-the-art Convolutional Neural Network (CNN) to classify dermatological images as potentially cancerous or non-cancerous. 
        Upload a high-quality image of the skin area in question, and our model will provide a prediction based on its analysis.
        """)
        
        st.header("Disclaimer")
        st.markdown("""
        This application is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified healthcare provider for any medical concerns or questions you may have regarding your health or a medical condition.
        """)

        st.header("How to Use")
        st.markdown("""
        1. Upload a clear, well-lit image of the skin area.
        2. Wait for the model to process and analyze the image.
        3. Review the prediction and confidence score.
        4. Remember to consult with a dermatologist for a proper diagnosis.
        """)

    st.write("---")

    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        st.write("### Upload an image of the skin area:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    with col2:
        st.write("### Example Image 1:")
        st.image("ex1.png", use_column_width=True)

    with col3:
        st.write("### Example Image 2:")
        st.image("ex2.jpg", use_column_width=True)

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        with st.spinner("Analyzing image..."):
            # Save the uploaded file to disk
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Make prediction
            prediction = predict_image(model, "temp.jpg")
        
        # Display prediction
        st.write("## Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_gauge_chart(prediction)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if prediction > 0.5:
                st.markdown(
                    """
                    <div class='prediction-box' style='background-color: #FFCCCB;'>
                        <h2>⚠️ Potential Skin Cancer Detected</h2>
                        <p>The image shows characteristics that may be associated with skin cancer. 
                        It is strongly recommended to consult a dermatologist for a professional evaluation.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class='prediction-box' style='background-color: #90EE90;'>
                        <h2>✅ No Immediate Concern Detected</h2>
                        <p>The image does not show immediate signs of skin cancer. 
                        However, regular skin check-ups are always recommended for maintaining skin health.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.write("### Important Note:")
        st.info("This prediction is based on machine learning algorithms and should not be considered as a definitive medical diagnosis. Always consult with a qualified healthcare professional for proper medical advice and treatment.")

        # Additional Information
        st.write("## Additional Information")
        st.write("Learn more about skin cancer and its prevention:")
        st.markdown("""
        - [Skin Cancer Foundation](https://www.skincancer.org/)
        - [American Academy of Dermatology](https://www.aad.org/public/diseases/skin-cancer)
        - [World Health Organization - Skin Cancer](https://www.who.int/news-room/fact-sheets/detail/skin-cancer)
        """)

if __name__ == '__main__':
    main()
