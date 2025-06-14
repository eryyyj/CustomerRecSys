import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import tempfile

# Set your Hugging Face API token
HF_API_TOKEN = st.secrets["secrets"]["HF_API_TOKEN"].strip()
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# Try to import OpenCV with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.warning(f"OpenCV import warning: {e}")
    CV2_AVAILABLE = False
except OSError as e:
    st.warning(f"OpenCV library missing: {e}")
    CV2_AVAILABLE = False

# Initialize models
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        token=HF_API_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_API_TOKEN
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
    As a retail business consultant, provide 3-5 actionable recommendations to improve a store 
    that typically has {customer_count} customers present at any given time. Focus on:
    - Staffing optimization
    - Store layout improvements
    - Promotional strategies
    - Customer experience enhancements
    - Inventory management
    - Queue management
    Format your response with clear headings for each recommendation category.
    """
    
    generator = load_llm()
    response = generator(
        prompt,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    return response[0]['generated_text'].split(prompt)[-1].strip()

# Streamlit UI
st.set_page_config(layout="wide", page_title="Retail Customer Analytics", page_icon="🛒")
st.title("🛒 Retail Customer Analytics")
st.subheader("Detect customers and get AI-powered business recommendations")

with st.expander("How to use this tool"):
    st.markdown("""
    1. **Upload** an image or video from your store
    2. Our AI will **detect and count** customers
    3. Get **actionable recommendations** to improve your business
    - Supported formats: JPG, PNG
    - For videos: Requires OpenCV with GUI libraries
    """)
    
    if not CV2_AVAILABLE:
        st.warning("""
        **Video processing disabled**: OpenCV dependencies missing. To enable video processing:
        - Install missing libraries: `apt-get install libgl1-mesa-glx`
        - Or use headless environment: `pip install opencv-python-headless`
        """)

uploaded_file = st.file_uploader(
    "Upload store image or video", 
    type=["jpg", "jpeg", "png"] + (["mp4", "mov"] if CV2_AVAILABLE else []),
    help="Supported formats: JPG, PNG" + (" + MP4, MOV" if CV2_AVAILABLE else "")
)

def process_image(image):
    """Process an image and return customer count and annotated image"""
    model = load_yolo()
    results = model.predict(image, classes=[0], conf=0.5)  # Class 0 = person
    customer_count = len(results[0].boxes)
    annotated_image = results[0].plot()[:, :, ::-1]  # Convert to RGB
    return customer_count, annotated_image

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Process image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        
        with st.spinner("Detecting customers..."):
            try:
                customer_count, annotated_image = process_image(image)
                col2.image(annotated_image, caption=f"Detected Customers: {customer_count}", use_column_width=True)
                st.metric("Total Customers Detected", customer_count)
                
                # Generate recommendations
                with st.spinner("Generating business recommendations..."):
                    try:
                        recommendations = generate_recommendation(customer_count)
                        st.subheader("📈 Business Improvement Recommendations")
                        st.markdown(f"```\n{recommendations}\n```")
                        
                        # Download button for results
                        result_text = f"Customer Count: {customer_count}\n\nRecommendations:\n{recommendations}"
                        st.download_button(
                            label="Download Results",
                            data=result_text,
                            file_name="business_recommendations.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.info("This might be due to high demand on the AI model. Please try again later.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif uploaded_file.type.startswith('video') and CV2_AVAILABLE:
        # Process video
        st.info("Video processing may take several minutes. For faster results, use clips under 1 minute.")
        
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name
        
        st.video(video_path)
        
        model = load_yolo()
        cap = cv2.VideoCapture(video_path)
        
        customer_counts = []
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process video frames
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_rate = max(1, total_frames // 100)  # Process max 100 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Skip frames based on sampling rate
            if frame_count % sampling_rate != 0:
                continue
                
            # Process frame
            results = model.predict(frame, classes=[0], conf=0.5, verbose=False)
            customer_count = len(results[0].boxes)
            customer_counts.append(customer_count)
            
            # Display processed frame
            annotated_frame = results[0].plot()[:, :, ::-1]
            frame_placeholder.image(annotated_frame, caption=f"Frame {frame_count}/{total_frames}")
            
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames | Current customers: {customer_count}")
        
        cap.release()
        os.unlink(video_path)  # Delete temp file
        frame_placeholder.empty()
        progress_bar.empty()
        status_text.empty()
        
        # Calculate statistics
        if customer_counts:
            avg_customers = int(np.mean(customer_counts))
            max_customers = max(customer_counts)
            
            st.subheader("📊 Video Analysis Results")
            col1, col2 = st.columns(2)
            col1.metric("Average Customers", avg_customers)
            col2.metric("Peak Customers", max_customers)
            
            st.line_chart(customer_counts, use_container_width=True)
            
            # Generate recommendations
            with st.spinner("Generating business recommendations..."):
                try:
                    recommendations = generate_recommendation(avg_customers)
                    st.subheader("📈 Business Improvement Recommendations")
                    st.markdown(f"```\n{recommendations}\n```")
                    
                    # Download button for results
                    result_text = f"Average Customers: {avg_customers}\nPeak Customers: {max_customers}\n\nRecommendations:\n{recommendations}"
                    st.download_button(
                        label="Download Results",
                        data=result_text,
                        file_name="video_analysis_recommendations.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.info("This might be due to high demand on the AI model. Please try again later.")
        else:
            st.warning("No customers detected in the video")
            
    elif uploaded_file.type.startswith('video') and not CV2_AVAILABLE:
        st.error("Video processing unavailable. OpenCV dependencies missing.")
        st.info("To enable video processing:")
        st.code("""
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install libgl1-mesa-glx

# Then reinstall OpenCV:
pip install opencv-python-headless --force-reinstall
""")

else:
    st.info("Please upload an image file to get started")
    st.image("https://images.unsplash.com/photo-1563014959-7aaa83350992?auto=format&fit=crop&w=1200&h=600", 
             caption="Retail Store Analytics Example", use_column_width=True)

# Add footer
st.markdown("---")
st.caption("AI Retail Advisor • Powered by YOLOv8 and Zephyr-7B-Beta")
