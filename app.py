# app.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
import uvicorn

# ------------------------------
# 1. Load Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 architecture
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

# Load trained weights (keep mobilenetv2_faceexp.pth in your repo folder)
save_path = "mobilenetv2_faceexp.pth"
model.load_state_dict(torch.load(save_path, map_location=device))

model = model.to(device)
model.eval()
print("✅ MobileNetV2 Model loaded successfully")

# ------------------------------
# 2. Define Classes & Transform
# ------------------------------
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# 3. Face Detection
# ------------------------------
def detect_face(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No face detected in the image!")
    x, y, w, h = faces[0]
    face_img = img_array[y:y+h, x:x+w]
    return face_img

# ------------------------------
# 4. FastAPI App
# ------------------------------
app = FastAPI()

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file).convert("RGB")
    img_array = np.array(image)

    try:
        face_img = detect_face(img_array)
    except ValueError as e:
        return {"error": str(e)}

    # Preprocess face
    input_tensor = transform(Image.fromarray(face_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    emotion = classes[pred.item()]

    # ------------------------------
    # 🎵 Call Backend for Recommendation
    # ------------------------------
    try:
        api_url = f"https://spotifyservicebackend.onrender.com/recommend?emotion={emotion}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            recommendation = response.json()
        else:
            recommendation = {"error": f"API returned {response.status_code}"}
    except Exception as e:
        recommendation = {"error": str(e)}

    return {"emotion": emotion, "recommendation": recommendation}

# ------------------------------
# 5. Run (local only)
# ------------------------------
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
