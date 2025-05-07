from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import ImageCaptionModel
from constants import IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, MAX_SEQUENCE_LENGTH, GENERATION_TEMPERATURE
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import glob
import os

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_latest_checkpoint():
    """Find the latest model checkpoint."""
    checkpoints = glob.glob(os.path.join("checkpoints", "epoch_*.pth"))
    if not checkpoints:
        return None
    # Sort by epoch number
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel().to(device)

# Load latest checkpoint if available
latest_checkpoint = get_latest_checkpoint()
if latest_checkpoint:
    print(f"Loading model from {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
else:
    print("No checkpoint found, using untrained model")

model.eval()

# Preprocessing (should match your training)
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_ids = model.generate(image_tensor, max_length=MAX_SEQUENCE_LENGTH, temperature=GENERATION_TEMPERATURE)
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return JSONResponse({"caption": caption}) 