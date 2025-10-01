import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import os

dataset_path = './fresh'  # folder containing freshness images
if not os.path.exists(dataset_path):
    raise FileNotFoundError("⚠️ Dataset folder 'fresh' not found!")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=dataset_path, transform=transform)

# CNN model
cnn = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1))
)

fc = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(train_data.classes))
)

def predict(img_path):
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = cnn(img)
        features = features.view(features.size(0), -1)
        outputs = fc(features)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class, train_data.classes
