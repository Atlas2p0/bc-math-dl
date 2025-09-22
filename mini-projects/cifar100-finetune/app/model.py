import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import onnxruntime as ort
import json
import onnx

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ONNX_PATH = os.path.join(os.getcwd(), "resnet18_fft.onnx")
# Check model integrity
try:
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("ONNX model validation passed.")
except Exception as e:
    print(f"Model integrity validation warning: {e}")  # Non-fatal for minor issues

session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name  # Dynamic: e.g., 'input' or 'x'
label_map_path = os.path.join(os.getcwd(), "../artifacts/label_mapping.json")

with open(label_map_path, 'r') as f:
    loaded_mapping = json.load(f)

# Convert to list for int indexing
class_names = [loaded_mapping[str(i)] for i in range(len(loaded_mapping))]
assert len(class_names) == 100, "Expected 100 CIFAR-100 classes"

def preprocess_image(image_input):
    """
    Preprocess single image input or list to batched NumPy for ONNX.
    Supports: str/path, PIL.Image, list[str], or list[PIL.Image].
    
    Args:
        image_input: Single (str or PIL.Image) or list (for batch 1-4).
    
    Returns:
        np.ndarray: [B, 3, 224, 224] ready for ORT.
    """
    # Wrap single input as list for uniform processing
    if isinstance(image_input, (str, Image.Image)):
        items = [image_input]
    elif isinstance(image_input, list):
        items = image_input
    else:
        raise ValueError("Input must be str/PIL.Image or list for batching.")
    
    batch_size = len(items)
    if batch_size > 4 or batch_size < 1:
        raise ValueError("Batch size must be 1-4.")
    
    tensors = []
    for item in items:
        img = None
        if isinstance(item, str):  # Path: Load with extension check
            _, ext = os.path.splitext(item)
            image_ext = ext.lstrip('.').lower()
            if image_ext not in ['jpeg', 'jpg', 'png']:
                raise ValueError(f"Unsupported image extension: .{image_ext}. Only JPEG, JPG, PNG allowed.")
            try:
                img = Image.open(item).convert('RGB')
            except (FileNotFoundError, OSError) as e:
                raise ValueError(f"Failed to load image at {item}: {str(e)}")
        elif isinstance(item, Image.Image):  # PIL: Use directly (from upload)
            img = item  # Already RGB from main.py
        else:
            raise ValueError("Each item must be str (path) or PIL.Image.")
        
        # Transform (now properly inside the loop)
        input_tensor = val_transforms(img)  # [C, H, W]
        tensors.append(input_tensor)
    
    # Stack and prepare
    input_batch = torch.stack(tensors).numpy().astype(np.float32)  # [B, C, H, W]
    return np.ascontiguousarray(input_batch)

def predict(input_batch, class_names):
    """
    Run ONNX inference on preprocessed batch and return predictions.
    
    Args:
        input_batch (np.ndarray): Shape [B, 3, 224, 224], B=1-4.
        class_names (list): List of 100 CIFAR-100 class names.
    
    Returns:
        dict or list[dict]: Single dict (B=1) or list for batches.
    """
    input_array = np.asarray(input_batch, dtype=np.float32)
    input_array = np.ascontiguousarray(input_array)
    
    if len(input_array.shape) != 4 or input_array.shape[1:] != (3, 224, 224):
        raise ValueError(f"Invalid input shape: {input_array.shape}. Expected [B, 3, 224, 224].")
    
    try:
        # Run inference
        ort_inputs = {input_name: input_array}
        ort_outputs = session.run(None, ort_inputs)
        logits = ort_outputs[0]  # [B, 100]
        
        # Validate output shape
        if logits.shape[-1] != 100:
            raise ValueError(f"Unexpected output shape: {logits.shape}. Expected [B, 100].")
        
        # Post-process per batch item
        probs = F.softmax(torch.from_numpy(logits), dim=1)
        batch_size = logits.shape[0]
        predictions = []
        for i in range(batch_size):
            confidence, pred_idx = torch.max(probs[i], dim=0)
            pred_class = class_names[pred_idx.item()]
            predictions.append({
                'predicted_class': pred_class,
                'confidence': float(confidence),
                'top1_index': int(pred_idx)
            })
        
        return predictions[0] if batch_size == 1 else predictions
    except Exception as e:
        raise ValueError(f"Inference error: {str(e)}. "
                         f"Input shape: {input_array.shape}. Input name: '{input_name}'. "
                         f"Check model with netron.") from e

if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), "../data/test/test6.jpeg")
    preproc_img = preprocess_image(image_path)  # Still works with str
    results = predict(preproc_img, class_names)
    print(f'Predicted Class: {results["predicted_class"]} | Confidence: {results["confidence"]}')