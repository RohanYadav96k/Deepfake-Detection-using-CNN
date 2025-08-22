
from flask import Flask, request, render_template_string, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Deepfake Image Detector</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { padding: 40px; background: #f8f9fa; }
        .container { max-width: 600px; }
        img { max-width: 100%; margin-top: 20px; border-radius: 8px; }
        .result-real { color: green; font-weight: bold; }
        .result-fake { color: red; font-weight: bold; }
    </style>
</head>
<body>
<div class="container text-center">
    <h2 class="mb-4">🕵️‍♂️ Deepfake Image Detector</h2>
    <form method="post" enctype="multipart/form-data" class="mb-3">
        <input type="file" name="file" accept="image/*" class="form-control mb-3" required>
        <button type="submit" class="btn btn-primary w-100">Upload & Detect</button>
    </form>
    {% if error %}
      <div class="alert alert-danger">{{ error }}</div>
    {% endif %}
    {% if result %}
      <h4 class="{{ 'result-real' if result == 'real' else 'result-fake' }}">
        Prediction: {{ result|upper }} (Confidence: {{ confidence }}%)
      </h4>
      <img src="{{ url_for('static', filename='uploaded_image.jpg') }}" alt="Uploaded Image">
    {% endif %}
</div>
</body>
</html>
"""

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model.eval()

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
    confidence = round(probs.max().item() * 100, 2)
    _, predicted = torch.max(outputs, 1)
    classes = ["fake", "real"]
    return classes[predicted.item()], confidence

@app.route("/", methods=["GET", "POST"])
def upload_file():
    result = None
    error = None
    confidence = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file selected"
        else:
            os.makedirs("static", exist_ok=True)
            save_path = os.path.join("static", "uploaded_image.jpg")
            file.save(save_path)
            result, confidence = predict_image(save_path)
    return render_template_string(HTML, result=result, error=error, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
