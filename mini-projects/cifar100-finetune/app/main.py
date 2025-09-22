from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from PIL import Image
from typing import List
# Import from model.py
from model import preprocess_image, predict, class_names
import uvicorn

app = FastAPI(title="CIFAR-100 ResNet18 Inference API", version="1.0.0", description="ONNX-based image classification")

@app.get("/")
async def health_check():
    return {"status": "healthy", "model": "ResNet18 ONNX loaded", "classes": len(class_names)}

@app.post("/predict/single")
async def predict_single(image: UploadFile = File(..., description="Single image (JPEG/PNG)")):
    """
    Predict class for a single uploaded image.
    Upload a JPEG/PNG file to the 'image' field.
    """
    try:
        # Validate file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(400, "Image must be JPEG/PNG (image/* content-type).")
        if image.size > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(400, "Image too large (>5MB).")
        
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        preprocessed = preprocess_image(pil_image)
        result = predict(preprocessed, class_names)
        return result  # Auto-JSON dict
    
    except ValueError as e:
        raise HTTPException(400, f"Preprocessing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(images: List[UploadFile] = File(..., description="Batch of 1-4 images (JPEG/PNG)")):
    """
    Predict classes for a batch of 1-4 uploaded images.
    Upload multiple files to the 'images' field (add slots in UI).
    """
    try:
        if len(images) == 0:
            raise HTTPException(400, "Batch must contain 1-4 images.")
        if len(images) > 4:
            raise HTTPException(400, "Maximum batch size is 4.")
        
        # Validate and process each file
        pil_images = []
        for i, file in enumerate(images):
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(400, f"Image {i+1} (filename: {file.filename}) must be JPEG/PNG.")
            if file.size > 5 * 1024 * 1024:  # 5MB per file
                raise HTTPException(400, f"Image {i+1} ({file.filename}) too large (>5MB).")
            
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
            pil_images.append(pil_img)
        
        preprocessed = preprocess_image(pil_images)
        results = predict(preprocessed, class_names)
        return {"predictions": results}  # Auto-JSON with list of dicts
    
    except ValueError as e:
        raise HTTPException(400, f"Preprocessing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)