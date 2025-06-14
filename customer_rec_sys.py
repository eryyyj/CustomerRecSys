# IMPORT ALL STANDARD LIBRARIES FIRST
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY'] = '0'  # Fix for PyTorch 2.6 security change
import time
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

# IMPORT STREAMLIT AFTER STANDARD LIBRARIES
import streamlit as st

# SET PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Customer Analytics Inspector",
    page_icon="👥",
    layout="wide"
)

# CONFIGURATION
HF_API_TOKEN = st.secrets["secrets"]["HF_API_TOKEN"].strip()
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# SAFE MODEL LOADING FUNCTION
@st.cache_resource
def load_yolo_model():
    try:
        # Load pre-trained YOLOv8 model with safety workaround
        model = YOLO('yolov8n.pt',task='detect')  # Official COCO model
        
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading YOLO model: {str(e)}")
        try:
            # Fallback to direct loading
            from ultralytics.yolo.engine.model import YOLO as SafeYOLO
            return SafeYOLO('yolov8n.pt')
        except:
            st.error("Failed to load YOLO model with fallback method")
        return None

# LOAD MODEL AT STARTUP
detection_model = load_yolo_model()

# OBJECT DETECTION FUNCTION
def detect_and_count(image_np):
    """
    Detect and count persons in an image
    Returns: (person_count: int, annotated_image: np.array)
    """
    if detection_model is None:
        return 0, image_np
    
    try:
        # Run inference with safety parameters
        results = detection_model.predict(
            image_np, 
            conf=0.5, 
            classes=[0],  # Class 0 = person
            imgsz=640,   # Standard input size
            device='cpu'  # Force CPU if GPU issues occur
        )
        
        # Initialize count
        person_count = 0
        annotated_frame = image_np.copy()
        
        # Process results
        for result in results:
            # Draw bounding boxes and labels
            annotated_frame = result.plot()
            
            # Count persons
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 0:  # Person class
                    person_count += 1
        
        return person_count, annotated_frame
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return 0, image_np

# GENERATE BUSINESS RECOMMENDATIONS (SAME AS BEFORE)
def generate_business_recommendations(person_count, is_video=False, avg_count=0):
    """
    Generate business recommendations based on customer count
    """
    if person_count == 0 and avg_count == 0:
        return "No customers detected. Consider promotional activities to attract more visitors."
    
    # Create prompt based on detection type
    if is_video:
        prompt = f"""
        As a retail business consultant, analyze store traffic with an average of {avg_count:.1f} customers. 
        Provide 5 actionable recommendations to:
        1. Optimize staffing levels
        2. Improve customer experience
        3. Increase conversion rates
        4. Enhance store layout
        5. Boost sales opportunities
        
        Focus on practical, cost-effective solutions suitable for a retail environment.
        """
    else:
        prompt = f"""
        As a retail business consultant, analyze a store snapshot showing {person_count} customers. 
        Provide 5 specific recommendations to:
        1. Improve customer engagement
        2. Optimize product placement
        3. Enhance staff-customer interactions
        4. Increase sales conversion
        5. Manage crowd flow
        
        Offer practical, immediate actions the store manager can implement.
        """
    
    if not HF_API_TOKEN:
        return "⚠️ Error: Hugging Face API token not configured."
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text[:200]}..."
            
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
        
        return f"⚠️ Unexpected response format: {str(result)[:300]}"
    
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# APP TITLE AND DESCRIPTION
st.title("👥 Customer Analytics Inspector")
st.subheader("Retail Customer Detection & Business Optimization")
st.markdown("""
    *Detect and count customers in your store, then get AI-powered recommendations to improve your business*
""")

# FILE UPLOAD SECTION
with st.expander("📤 Upload Media", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload store image or video",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: Images (JPG, PNG), Videos (MP4)"
    )

# PROCESSING SECTION
if uploaded_file:
    # IMAGE PROCESSING
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Store Image", use_column_width=True)
        
        if st.button("🔍 Analyze Customer Traffic", type="primary"):
            if detection_model is None:
                st.error("Detection model not loaded. Unable to process.")
                st.stop()
                
            with st.spinner("Detecting customers..."):
                # Convert to OpenCV format
                img_np = np.array(image)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Run detection
                person_count, annotated_img = detect_and_count(img_np)
                
                # Convert back to RGB for display
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.subheader("🔍 Detection Results")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(annotated_img_rgb, caption="Customer Detection", use_column_width=True)
                with col2:
                    st.metric("Customers Detected", person_count, 
                              help="Number of people detected in the image")
                    
                    st.info(f"**Business Insight**:")
                    if person_count == 0:
                        st.warning("No customers detected - consider promotional activities")
                    elif person_count < 3:
                        st.success("Low traffic - opportunity for personalized service")
                    elif person_count < 10:
                        st.success("Moderate traffic - focus on conversion optimization")
                    else:
                        st.warning("High traffic - ensure staff availability and checkout efficiency")
                
                # Generate recommendations
                st.divider()
                st.subheader("📈 Business Optimization Recommendations")
                
                with st.spinner("Generating AI-powered recommendations..."):
                    recommendations = generate_business_recommendations(person_count)
                    st.markdown(recommendations)
    
    # VIDEO PROCESSING
    elif uploaded_file.type.startswith("video"):
        if detection_model is None:
            st.error("Detection model not loaded. Unable to process video.")
            st.stop()
            
        st.info("Video processing started. This may take 1-2 minutes...")
        
        # Save video to temp file
        temp_video = f"temp_{uploaded_file.name}"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analysis
        total_frames = 0
        processed_frames = 0
        total_customers = 0
        max_customers = 0
        min_customers = 1000
        customer_counts = []
        
        # Create placeholders for UI
        status_text = st.empty()
        progress_bar = st.progress(0)
        video_placeholder = st.empty()
        
        # Process video
        cap = cv2.VideoCapture(temp_video)
        frame_skip = 10  # Process every 10th frame to reduce computation
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Process only every nth frame
            if total_frames % frame_skip != 0:
                continue
                
            processed_frames += 1
            
            # Update progress
            progress = min(int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_video_frames * 100)), 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {total_frames}/{total_video_frames}...")
            
            # Detect and count persons
            person_count, annotated_frame = detect_and_count(frame)
            total_customers += person_count
            customer_counts.append(person_count)
            
            # Track min/max
            if person_count > max_customers:
                max_customers = person_count
            if person_count < min_customers:
                min_customers = person_count
                
            # Display preview every 50 processed frames
            if processed_frames % 50 == 0:
                preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(preview_frame, caption="Processing Preview", width=600)
        
        # Release resources
        cap.release()
        os.remove(temp_video)
        
        # Calculate metrics
        if processed_frames > 0:
            avg_customers = total_customers / processed_frames
            peak_time = (customer_counts.index(max_customers) * frame_skip) / fps
        else:
            avg_customers = 0
            peak_time = 0
        
        # Display results
        st.success("✅ Video analysis complete!")
        st.subheader("📊 Customer Traffic Analysis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Customers", f"{avg_customers:.1f}", 
                   help="Average number of customers per frame")
        col2.metric("Peak Crowd", max_customers, 
                   delta=f"at {peak_time:.1f} seconds",
                   help="Maximum number of customers detected")
        col3.metric("Quiet Period", min_customers, 
                   help="Minimum number of customers detected")
        
        # Show traffic chart
        st.line_chart(customer_counts, use_container_width=True, height=300)
        
        # Generate recommendations
        st.divider()
        st.subheader("📈 Business Optimization Recommendations")
        
        with st.spinner("Generating AI-powered recommendations..."):
            recommendations = generate_business_recommendations(0, True, avg_customers)
            st.markdown(recommendations)

# BUSINESS TIPS SECTION
st.divider()
st.subheader("💡 Retail Optimization Tips")

tips = """
1. **Staff Allocation**: Align staff schedules with peak customer hours
2. **Queue Management**: Implement efficient checkout systems during busy periods
3. **Product Placement**: Position high-margin items in high-traffic areas
4. **Promotional Timing**: Run promotions during low-traffic hours to boost visits
5. **Store Layout**: Optimize aisle design to improve flow and increase browsing
"""
st.markdown(tips)

# SIDEBAR RESOURCES
st.sidebar.title("📚 Retail Analytics Resources")
st.sidebar.markdown("""
- [Customer Behavior Analysis Guide](https://hbr.org/topic/customer-analytics)
- [Retail Staff Optimization](https://www.retailcustomerexperience.com/articles/)
- [Store Layout Best Practices](https://www.nrf.com/resources)
""")

st.sidebar.divider()
st.sidebar.markdown("""
**How to Use:**
1. Upload store image or video
2. Get customer count analytics
3. Receive AI-powered recommendations
4. Implement suggested improvements
""")

# FOOTER
st.divider()
st.caption("""
    *Note: Customer detection accuracy depends on video quality and camera angles. 
    Recommendations are AI-generated and should be validated with business metrics.*
""")
st.caption(f"App version: 1.0 | Last updated: {time.strftime('%Y-%m-%d')}")
