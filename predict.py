
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import random

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

model_path = "models/cnn_model.pth"
if not os.path.exists(model_path):
    print("❌ Train model first using train.py")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence = probs.max().item() * 100
    _, predicted = torch.max(outputs, 1)
    classes = ["fake", "real"]
    return classes[predicted.item()], confidence

if __name__ == "__main__":
    real_folder = "dataset/real"
    if not os.path.exists(real_folder) or len(os.listdir(real_folder)) == 0:
        print("❌ Add images in dataset/real folder")
        exit()

    image_file = random.choice(os.listdir(real_folder))
    image_path = os.path.join(real_folder, image_file)

    print(f"🔍 Checking image: {image_path}")
    result, confidence = predict_image(image_path)
    print(f"✅ Prediction: {result.upper()} (Confidence: {confidence:.2f}%)")
