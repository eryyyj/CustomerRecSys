import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize models
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

def generate_recommendation(customer_count):
    prompt = f"""
    I have a retail store with an average of {customer_count} customers present at any given time.
    Based on this customer traffic, provide 3-5 actionable recommendations to improve my business.
    Focus on: staffing optimization, store layout, promotions, and customer experience.
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a retail business consultant providing data-driven recommendations.",
        },
        {"role": "user", "content": prompt},
    ]
    
    generator = load_llm()
    response = generator(
        messages,
        max_new_tokens=350,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    return response[0]['generated_text'][-1]['content']

# Streamlit UI
st.title("🛒 Store Customer Analytics")
st.subheader("Detect customers and get business recommendations")

uploaded_file = st.file_uploader(
    "Upload store image or video", 
    type=["jpg", "jpeg", "png", "mp4", "mov"],
    help="Supported formats: JPG, PNG, MP4"
)

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        model = load_yolo()
        results = model.predict(image, classes=[0], conf=0.5)  # Class 0 = person
        
        # Process results
        customer_count = len(results[0].boxes)
        annotated_image = results[0].plot()[:, :, ::-1]  # Convert to RGB
        
        col1, col2 = st.columns(2)
        col1.image(annotated_image, caption=f"Detected Customers: {customer_count}", use_column_width=True)
        col2.metric("Total Customers Detected", customer_count)
        
        # Generate recommendations
        with st.spinner("Generating business recommendations..."):
            try:
                recommendations = generate_recommendation(customer_count)
                st.subheader("Business Improvement Recommendations")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

    elif uploaded_file.type.startswith('video'):
        # Process video
        st.info("Video processing may take several minutes depending on length")
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        
        # Save video to temp file
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)
        
        model = load_yolo()
        cap = cv2.VideoCapture("temp_video.mp4")
        
        customer_counts = []
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Process video frames
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame, classes=[0], conf=0.5, verbose=False)
            customer_count = len(results[0].boxes)
            customer_counts.append(customer_count)
            
            # Display processed frame
            annotated_frame = results[0].plot()[:, :, ::-1]
            frame_placeholder.image(annotated_frame, caption="Live Processing")
            
            # Update progress
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        frame_placeholder.empty()
        progress_bar.empty()
        
        # Calculate statistics
        avg_customers = int(np.mean(customer_counts))
        max_customers = max(customer_counts)
        
        st.subheader("Video Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Average Customers", avg_customers)
        col2.metric("Peak Customers", max_customers)
        
        st.line_chart(customer_counts)
        
        # Generate recommendations
        with st.spinner("Generating business recommendations..."):
            try:
                recommendations = generate_recommendation(avg_customers)
                st.subheader("Business Improvement Recommendations")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

else:
    st.info("Please upload an image or video file to get started")
    st.image("https://images.unsplash.com/photo-1563014959-7aaa83350992?auto=format&fit=crop&w=1200&h=600", 
             caption="Retail Store Analytics Example")