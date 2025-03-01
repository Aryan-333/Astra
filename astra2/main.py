import streamlit as st
import torch
import spacy
import numpy as np
import cv2
import replicate
import requests
from PIL import Image
import io
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import open_clip

# Load models
@st.cache_resource
def load_models():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model and transforms
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai"
    )
    model = model.to(device)
    model.eval()  # Set to evaluation mode for inference
    
    # Load spaCy for text processing
    nlp = spacy.load("en_core_web_sm")
    
    return model, preprocess, nlp, device

# Set page config with more appealing theme
st.set_page_config(
    page_title="Jewelry Image Editor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 10px 15px;
        border-radius: 3px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_edit' not in st.session_state:
    st.session_state.current_edit = None

# Main title with custom styling
st.markdown('<div class="main-header">💎 Jewelry Image Editor</div>', unsafe_allow_html=True)

# Create sidebar for configuration options
with st.sidebar:
    st.markdown('<div class="subheader">Configuration</div>', unsafe_allow_html=True)
    
    with st.expander("Advanced Settings", expanded=False):
        # Add slider for mask threshold adjustment
        mask_threshold = st.slider("Mask Precision", 0.1, 0.9, 0.5, 0.05, 
                                  help="Adjust to control the precision of the mask. Lower values select larger areas.")
        
        # Add option to select different mask generation methods
        mask_method = st.radio(
            "Mask Generation Method",
            ["Improved (Recommended)", "Standard"],
            index=0,
            help="Select the method used to generate the mask"
        )
        
        # Add debug option
        debug_mode = st.checkbox("Debug Mode", value=False, 
                               help="Show additional technical information")
    
    st.markdown('<div class="subheader">Help</div>', unsafe_allow_html=True)
    
    with st.expander("Tips for Better Results", expanded=True):
        st.markdown("""
        - Be specific about which part of the jewelry you want to edit
        - For color changes, specify the exact color (e.g., 'white gold' instead of 'lighter')
        - Adding details like 'high quality' or 'detailed' can improve results
        - Use the mask precision slider if the automatic mask doesn't capture the right area
        """)
    
    with st.expander("Example Prompts", expanded=True):
        st.markdown("""
        - Change the gold band to white gold
        - Make the diamond bigger
        - Change the emerald to ruby
        - Add diamond accents to the band
        - Make the pendant smaller
        """)

# Main content area with tabs
tab1, tab2 = st.tabs(["Edit Image", "Edit History"])

with tab1:
    # Load models
    model, preprocess, nlp, device = load_models()
    
    # Create two columns for upload and instructions
    upload_col, preview_col = st.columns([1, 1])
    
    with upload_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Jewelry Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with preview_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader" style="margin-top:0">Edit Instructions</div>', unsafe_allow_html=True)
        
        prompt = st.text_input("Enter your editing prompt", 
                              placeholder="e.g., 'Change gold to white gold'")
        
        additional_prompt = st.text_input("Additional details (optional)", 
                                         placeholder="e.g., 'high quality, luxury finish'",
                                         help="Add jewelry-specific details to improve the result")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Tip:** Be specific about which part of the jewelry you want to edit and what change you'd like to make.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Function to generate improved mask (unchanged)
    def generate_improved_mask(image, text, model, device, preprocess):
        """
        Generate a more accurate mask using CLIP features and advanced image processing.
        
        Args:
            image (PIL.Image): Input image.
            text (str): Text prompt describing the target area.
            model: CLIP model.
            device: Device to run on (cuda or cpu).
            preprocess: Preprocessing transform for CLIP.
        
        Returns:
            binary_mask (np.ndarray): Binary mask (0 or 255) of shape (height, width).
            heatmap (np.ndarray): Heatmap of similarities for debugging.
        """
        # Convert PIL image to numpy array (RGB)
        np_image = np.array(image)
        
        # Create a copy for visualization and processing
        original_size = (image.width, image.height)
        
        # Preprocess image for CLIP
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize text
        text_tokens = open_clip.tokenize([text]).to(device)
        
        # Get text feature and normalize
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
        # Get image features
        with torch.no_grad():
            # Forward pass through the visual model
            image_features = model.encode_image(image_tensor)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get the raw similarity score
        similarity = (image_features @ text_feature.T).item()
        
        # Create a more detailed similarity map using a sliding window approach
        window_size = 112  # Half of CLIP's default input size
        stride = window_size // 2  # 50% overlap
        
        # Create empty heatmap at original image size
        heatmap = np.zeros(original_size[::-1], dtype=np.float32)
        count_map = np.zeros(original_size[::-1], dtype=np.float32)
        
        # Iterate through the image with sliding windows
        for y in range(0, original_size[1], stride):
            for x in range(0, original_size[0], stride):
                # Check if we're out of bounds
                if y + window_size > original_size[1] or x + window_size > original_size[0]:
                    continue
                
                # Crop the region
                region = image.crop((x, y, x + window_size, y + window_size))
                
                # Preprocess for CLIP
                region_tensor = preprocess(region).unsqueeze(0).to(device)
                
                # Get region features
                with torch.no_grad():
                    region_features = model.encode_image(region_tensor)
                    region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity with text
                region_similarity = (region_features @ text_feature.T).item()
                
                # Add to heatmap
                heatmap[y:y+window_size, x:x+window_size] += region_similarity
                count_map[y:y+window_size, x:x+window_size] += 1
        
        # Average the overlapping areas
        count_map[count_map == 0] = 1  # Avoid division by zero
        heatmap = heatmap / count_map
        
        # Normalize heatmap to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Apply a Gaussian blur to smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Use Otsu's method for adaptive thresholding
        _, binary_mask = cv2.threshold(
            (heatmap * 255).astype(np.uint8), 
            0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and keep only the most significant ones
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Create a new mask with only significant contours
            refined_mask = np.zeros_like(binary_mask)
            
            # Find the total image area
            total_area = original_size[0] * original_size[1]
            
            # Keep contours based on relative size
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > total_area * 0.01 and area < total_area * 0.9:  # Between 1% and 90% of the image
                    cv2.drawContours(refined_mask, [contour], -1, 255, -1)
            
            # If refined mask is not empty, use it
            if np.sum(refined_mask) > 0:
                binary_mask = refined_mask
        
        # Check mask coverage and adjust if necessary
        mask_coverage = np.sum(binary_mask > 0) / (original_size[0] * original_size[1])
        
        # If coverage is too small or too large, try adaptive thresholding
        if mask_coverage < 0.01 or mask_coverage > 0.9:
            print(f"Mask coverage ({mask_coverage:.2%}) outside ideal range, adjusting.")
            
            # Use different percentiles based on if we need more or less coverage
            if mask_coverage < 0.01:
                # Need more coverage - lower threshold
                threshold = np.percentile(heatmap, 70)
            else:
                # Need less coverage - higher threshold
                threshold = np.percentile(heatmap, 90)
            
            binary_mask = (heatmap > threshold).astype(np.uint8) * 255
            
            # Apply morphological operations again
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # If we still have issues, try using GrabCut for further refinement
        if 0.01 < mask_coverage < 0.9:
            # Convert mask to GrabCut format (0: bg, 1: fg, 2: prob bg, 3: prob fg)
            grabcut_mask = np.zeros(np_image.shape[:2], dtype=np.uint8)
            grabcut_mask[binary_mask > 0] = cv2.GC_PR_FGD  # Probably foreground
            grabcut_mask[binary_mask == 0] = cv2.GC_PR_BGD  # Probably background
            
            # Create dummy arrays for GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            try:
                cv2.grabCut(
                    np_image, grabcut_mask, None, bgd_model, fgd_model, 
                    5, cv2.GC_INIT_WITH_MASK
                )
                
                # Extract new mask
                refined_mask = np.where(
                    (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                    255, 0
                ).astype(np.uint8)
                
                # If refined mask is not empty, use it
                if np.sum(refined_mask) > 0:
                    binary_mask = refined_mask
            except cv2.error:
                # GrabCut can sometimes fail, in which case we'll stick with our current mask
                pass
        
        return binary_mask, heatmap

    # Function to parse prompt using spaCy (unchanged)
    def parse_prompt(prompt, nlp):
        doc = nlp(prompt.lower())
        
        # Initialize variables
        action = "change"  # Default action
        target = ""
        new_state = ""
        
        # Simple parsing logic based on keywords
        if "change" in prompt.lower():
            action = "change"
            # Extract target and new state
            parts = prompt.lower().split(" to ")
            if len(parts) > 1:
                target_part = parts[0].replace("change", "").strip()
                if "the" in target_part:
                    target = target_part.split("the")[1].strip()
                else:
                    target = target_part
                new_state = parts[1].strip()
        
        elif "make" in prompt.lower() and "bigger" in prompt.lower():
            action = "make_bigger"
            # Extract target
            parts = prompt.lower().split("make")
            if len(parts) > 1:
                target_part = parts[1].replace("bigger", "").strip()
                if "the" in target_part:
                    target = target_part.split("the")[1].strip()
                else:
                    target = target_part
        
        elif "make" in prompt.lower() and "smaller" in prompt.lower():
            action = "make_smaller"
            # Extract target
            parts = prompt.lower().split("make")
            if len(parts) > 1:
                target_part = parts[1].replace("smaller", "").strip()
                if "the" in target_part:
                    target = target_part.split("the")[1].strip()
                else:
                    target = target_part
        
        elif "add" in prompt.lower():
            action = "add"
            # Extract what to add and where
            parts = prompt.lower().split("add")
            if len(parts) > 1:
                target = parts[1].strip()
        
        # Add fallback case for general edits
        else:
            action = "change"
            # Try to extract keywords
            target = prompt.lower()
            for word in ["modify", "edit", "transform"]:
                if word in target:
                    target = target.replace(word, "").strip()
        
        # Enhanced keyword extraction for jewelry
        jewelry_keywords = ["gold", "silver", "platinum", "diamond", "ruby", "emerald", 
                            "sapphire", "ring", "necklace", "bracelet", "earring", 
                            "pendant", "gem", "stone", "band", "setting"]
        
        # If target is empty or too general, try to extract jewelry-specific terms
        if not target or len(target.split()) > 5:
            for keyword in jewelry_keywords:
                if keyword in prompt.lower():
                    target = keyword
                    break
        
        return {
            "action": action,
            "target": target,
            "new_state": new_state
        }

    # Function to modify mask based on action (unchanged)
    def modify_mask(mask, action):
        if action == "make_bigger":
            # Dilate mask to make target bigger
            kernel = np.ones((7, 7), np.uint8)
            modified_mask = cv2.dilate(mask, kernel, iterations=2)
            return modified_mask
        
        elif action == "make_smaller":
            # Erode mask to make target smaller
            kernel = np.ones((7, 7), np.uint8)
            modified_mask = cv2.erode(mask, kernel, iterations=2)
            return modified_mask
        
        else:
            # For other actions, use the original mask
            return mask

    # Function to perform inpainting using Replicate API (unchanged)
    def inpaint_with_stable_diffusion(image, mask, prompt):
        # Save the images temporarily
        image_path = "temp_image.png"
        mask_path = "temp_mask.png"
        
        image.save(image_path)
        Image.fromarray(mask).save(mask_path)
        
        # Set up Replicate API
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            st.error("Replicate API token not found. Please set the REPLICATE_API_TOKEN environment variable.")
            return None
        
        client = replicate.Client(api_token=api_token)
        
        # Run inpainting model with updated model version
        try:
            output = client.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": open(image_path, "rb"),
                    "mask": open(mask_path, "rb"),
                    "prompt": prompt,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30,
                    "seed": 42  # Fixed seed for reproducibility
                }
            )
        except replicate.exceptions.ReplicateError as e:
            # Fallback to another model version if the first one fails
            st.warning(f"First model attempt failed: {e}. Trying alternative model...")
            try:
                output = client.run(
                    "stability-ai/stable-diffusion-inpainting:0a5df8c5a256f729f2bb0ba3b2f0f3ac2c0e13aab9e9d3c0b8432d6f8a53e0b9",
                    input={
                        "image": open(image_path, "rb"),
                        "mask": open(mask_path, "rb"),
                        "prompt": prompt,
                        "num_outputs": 1,
                        "guidance_scale": 7.5,
                        "num_inference_steps": 30,
                        "seed": 42  # Fixed seed for reproducibility
                    }
                )
            except replicate.exceptions.ReplicateError as e2:
                st.error(f"Alternative model also failed: {e2}")
                # Clean up temporary files
                os.remove(image_path)
                os.remove(mask_path)
                return None
        
        # Clean up temporary files
        os.remove(image_path)
        os.remove(mask_path)
        
        # Download the result
        if output and len(output) > 0:
            response = requests.get(output[0])
            edited_image = Image.open(io.BytesIO(response.content))
            return edited_image
        
        return None
    
    # Process button with loading animation
    if uploaded_file is not None and prompt:
        process_col1, process_col2 = st.columns([3, 1])
        with process_col1:
            edit_button = st.button("✨ Generate Edited Image", use_container_width=True)
        with process_col2:
            cancel_button = st.button("Cancel", use_container_width=True)
    
    # Process the image when button is clicked
    if uploaded_file is not None and prompt and edit_button:
        # Create a progress bar and status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Parse prompt
        progress_bar.progress(10)
        status_text.text("Analyzing your request...")
        parsed_prompt = parse_prompt(prompt, nlp)
        
        # Show parsed information in an expandable section
        with st.expander("Request Analysis", expanded=debug_mode):
            st.json(parsed_prompt)
        
        # Step 2: Generate mask
        progress_bar.progress(30)
        status_text.text("Identifying jewelry elements...")
        
        target_text = parsed_prompt["target"]
        if not target_text:
            st.error("Could not identify what to edit in your prompt. Please be more specific.")
            progress_bar.empty()
            status_text.empty()
        else:
            # Use the selected mask generation method
            if mask_method == "Improved (Recommended)":
                mask, heatmap = generate_improved_mask(image, target_text, model, device, preprocess)
            else:
                # Use standard method (same function for now)
                mask, heatmap = generate_improved_mask(image, target_text, model, device, preprocess)
            
            # Step 3: Show the mask and prepare for editing
            progress_bar.progress(50)
            status_text.text("Preparing to edit selected area...")
            
            # Show the mask in a more visually appealing way
            mask_col1, mask_col2 = st.columns(2)
            with mask_col1:
                st.markdown('<div class="subheader">Selected Area</div>', unsafe_allow_html=True)
                st.image(mask, caption=f"Area to edit: '{target_text}'", use_container_width=True)
            with mask_col2:
                st.markdown('<div class="subheader">Detection Heatmap</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(heatmap, cmap='viridis')
                ax.set_title(f"Detection confidence for '{target_text}'")
                ax.axis('off')
                st.pyplot(fig)
            
            # Step 4: Modify mask based on action
            progress_bar.progress(70)
            status_text.text("Preparing edit instructions...")
            
            modified_mask = modify_mask(mask, parsed_prompt["action"])
            
            # Create inpainting prompt
            if parsed_prompt["action"] == "change":
                if parsed_prompt["new_state"]:
                    inpaint_prompt = f"{parsed_prompt['new_state']} jewelry, high quality, detailed"
                else:
                    inpaint_prompt = f"modified {target_text} jewelry, high quality, detailed"
            elif parsed_prompt["action"] == "make_bigger":
                inpaint_prompt = f"larger {target_text} in jewelry, high quality, detailed"
            elif parsed_prompt["action"] == "make_smaller":
                inpaint_prompt = f"smaller {target_text} in jewelry, high quality, detailed"
            elif parsed_prompt["action"] == "add":
                inpaint_prompt = f"{target_text} in jewelry, high quality, detailed"
            
            # Add additional details if provided
            if additional_prompt:
                inpaint_prompt += f", {additional_prompt}"
            
            # Show the full prompt in debug mode
            if debug_mode:
                st.write(f"Using inpainting prompt: '{inpaint_prompt}'")
            
            # Step 5: Perform inpainting
            progress_bar.progress(80)
            status_text.text("Applying edits (this may take a moment)...")
            
            edited_image = inpaint_with_stable_diffusion(image, modified_mask, inpaint_prompt)
            
            # Final step: Show results
            progress_bar.progress(100)
            
            if edited_image is not None:
                status_text.text("Edit complete! ✅")
                
                # Display before/after comparison
                st.markdown('<div class="subheader">Results</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before**")
                    st.image(image, use_container_width=True)
                with col2:
                    st.markdown("**After**")
                    st.image(edited_image, use_container_width=True)
                
                # Add download button for result
                buf = io.BytesIO()
                edited_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Edited Image",
                    data=byte_im,
                    file_name="edited_jewelry.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Add to history
                st.session_state.history.append({
                    "original": image,
                    "edited": edited_image,
                    "prompt": prompt,
                    "mask": mask,
                    "heatmap": heatmap,
                })
                
                # Reset progress indicators
                progress_bar.empty()
                status_text.empty()
            else:
                status_text.error("Failed to generate edited image. Please try again with a different prompt.")
                progress_bar.empty()

# History tab
with tab2:
    if not st.session_state.history:
        st.info("No edits have been made yet. Edit an image to see your history here.")
    else:
        st.markdown('<div class="subheader">Your Editing History</div>', unsafe_allow_html=True)
        
        # Create a more visually appealing history section
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h4>Edit {len(st.session_state.history) - i}: {item['prompt']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    st.image(item["original"], caption="Original", use_container_width=True)
                with cols[1]:
                    st.image(item["mask"], caption="Selected Area", use_container_width=True)
                with cols[2]:
                    st.image(item["edited"], caption="Result", use_container_width=True)
                
                # Add buttons for each history item
                button_cols = st.columns([1, 1, 1])
                with button_cols[1]:
                    if st.button(f"Download Result #{len(st.session_state.history) - i}", key=f"dl_{i}"):
                        buf = io.BytesIO()
                        item["edited"].save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label=f"Download Edit #{len(st.session_state.history) - i}",
                            data=byte_im,
                            file_name=f"jewelry_edit_{len(st.session_state.history) - i}.png",
                            mime="image/png"
                        )
        
        # Add button to clear history
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
            st.experimental_rerun()

# Footer with credits
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #6B7280;">
    <p>Powered by CLIP and Stable Diffusion. Built with Streamlit.</p>
</div>
""", unsafe_allow_html=True)

# Debug Section
if st.checkbox("Show debug information"):
    if 'mask' in locals() and 'heatmap' in locals():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(image); ax1.set_title("Original Image"); ax1.axis("off")
        ax2.imshow(heatmap, cmap='jet'); ax2.set_title(f"Heatmap for '{target_text}'"); ax2.axis("off")
        ax3.imshow(mask, cmap='gray'); ax3.set_title("Binary Mask"); ax3.axis("off")
        st.pyplot(fig)
        st.write(f"Mask coverage: {np.sum(mask > 0) / (image.width * image.height):.2%}")
        st.write(f"Max heatmap value: {np.max(heatmap):.4f}")
        st.write(f"Min heatmap value: {np.min(heatmap):.4f}")