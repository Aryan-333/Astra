import numpy as np
import cv2
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

def visualize_clip_gradcam(image_path, text, save_path=None):
    """
    Visualize CLIP-GradCAM to see where the model focuses when given text and image
    
    Args:
        image_path: Path to the image
        text: Text prompt to analyze
        save_path: Optional path to save the visualization
    """
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Process text
    text_tokens = clip.tokenize([text]).to(device)
    
    # Get image features with gradient tracking
    image_tensor.requires_grad = True
    image_features = model.encode_image(image_tensor)
    
    # Get text features
    text_features = model.encode_text(text_tokens)
    
    # Compute similarity
    similarity = (image_features @ text_features.T)[0, 0]
    
    # Compute gradients
    model.zero_grad()
    similarity.backward()
    
    # Get gradients from the last convolutional layer
    gradients = model.visual.transformer.resblocks[-1].attn.out_proj.weight.grad
    activations = model.visual.transformer.resblocks[-1].attn.out_proj.weight
    
    # Pool the gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 1])
    
    # Weight activation maps with gradients
    for i in range(activations.size(0)):
        activations[i] *= pooled_gradients[i]
    
    # Generate heatmap
    heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    img_array = np.array(image)
    superimposed_img = heatmap_colored * 0.4 + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Create binary mask
    threshold = 0.5  # Adjust based on testing
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255
    
    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f"Heatmap for '{text}'")
    axes[1].axis("off")
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")
    
    axes[3].imshow(binary_mask, cmap='gray')
    axes[3].set_title("Binary Mask")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return heatmap, binary_mask

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/jewelry/image.jpg"
    text = "gold"
    
    visualize_clip_gradcam(image_path, text, "clip_gradcam_visualization.png")