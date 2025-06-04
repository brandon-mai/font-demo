import torch
import torch.nn.functional as F
import numpy as np
import cv2


def predict_font(model, img_array, patch_size=(105, 105), device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict font class from an image using patch-based inference.
    
    Args:
        model: Initialized PyTorch model (AlexNetClassifier, HENet, FontClassifier, etc.)
        img_array: Input image as numpy array (grayscale or RGB)
        patch_size: Target patch size (height, width), default (105, 105)
        device: Device to run inference on
        
    Returns:
        prediction_probs: Averaged softmax probabilities across all patches
        predicted_class: Index of the predicted class
    """
    model.eval()
    model = model.to(device)
    
    # Create a temporary object with patch_size attribute for extract_patches_test
    class PatchExtractor:
        def __init__(self, patch_size):
            self.patch_size = patch_size
            
        def extract_patches_test(self, img_array):
            h, w = img_array.shape[:2]
            target_h, target_w = self.patch_size
            # 1) resize height to 105 pixels
            new_w = int(w * (target_h / h))
            img = cv2.resize(img_array, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            patches = []
            for _scale in range(3): # 3 squeeze ratios
                factor = np.random.uniform(0.7, 1.2)
                sw = max(1, int(new_w / factor))
                squeezed = cv2.resize(img, (sw, target_h), interpolation=cv2.INTER_LINEAR)
                # nếu width < target_w thì pad reflect, else crop giữa
                if sw < target_w:
                    pad = target_w - sw
                    left = pad//2; right = pad-left
                    padded = np.pad(squeezed,
                                        ((0,0),(left,right)) + ((0,0),)*(img.ndim-2),
                                        mode='edge')
                    for _ in range(5):
                        x = np.random.randint(0, max(1, target_w - target_w + 1))  # Always 0 since width == target_w
                        y = np.random.randint(0, max(1, target_h - target_h + 1))  # Always 0 since height == target_h
                        patch = padded[y:y+target_h, x:x+target_w] if img.ndim == 2 else padded[y:y+target_h, x:x+target_w, :]
                        patches.append(patch)
                else:
                    # Randomly crop 5 patches without further squeezing
                    for _ in range(5):
                        x = np.random.randint(0, max(1, sw - target_w + 1))
                        y = 0  # Height already matches target_h
                        patch = squeezed[y:y+target_h, x:x+target_w] if img.ndim == 2 else squeezed[y:y+target_h, x:x+target_w, :]
                        patches.append(patch)
            
            return patches
    
    # Extract patches using the function
    extractor = PatchExtractor(patch_size)
    patches = extractor.extract_patches_test(img_array)
    
    # Convert patches to tensor
    # Ensure patches are grayscale (single channel) for the models
    processed_patches = []
    for patch in patches:
        if len(patch.shape) == 3:  # RGB image
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        processed_patches.append(patch)
    
    # Stack patches and add channel dimension: (15, 1, 105, 105)
    patch_tensor = torch.FloatTensor(np.array(processed_patches)).unsqueeze(1).to(device)
    
    # Get predictions for all patches
    with torch.no_grad():
        logits = model(patch_tensor)  # Shape: (15, num_classes)
        
        # Apply softmax to get probabilities
        patch_probs = F.softmax(logits, dim=1)  # Shape: (15, num_classes)
        
        # Average probabilities across all patches
        prediction_probs = torch.mean(patch_probs, dim=0)  # Shape: (num_classes,)
        
        # Get predicted class
        predicted_class = torch.argmax(prediction_probs).item()
    
    return prediction_probs.cpu().numpy(), predicted_class