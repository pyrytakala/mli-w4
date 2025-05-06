from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import ImageCaptionModel
from constants import IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel().to(device)
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
        generated_ids = model.generate(image_tensor)
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return JSONResponse({"caption": caption}) 