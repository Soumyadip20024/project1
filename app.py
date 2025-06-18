import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
DATASET_PATH = 'ecg'  # Replace with path to 'ecg' folder with class subfolders

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model, device, transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = ImageFolder(root=DATASET_PATH, transform=transform)
class_names = dataset.classes

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
DATASET_PATH = 'ecg' # Replace with path to 'ecg' folder with class subfolders

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model, device, transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = ImageFolder(root=DATASET_PATH, transform=transform)
class_names = dataset.classes

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=2)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Ensure upload and static directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Store last metrics
last_metrics = {}

def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100.*correct/total:.2f}%")
    torch.save(model.state_dict(), "model.pth")

def evaluate_model():
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(STATIC_FOLDER, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    return {
        'accuracy': f"{acc:.2f}",
        'precision': f"{pre:.2f}",
        'recall': f"{rec:.2f}",
        'f1': f"{f1:.2f}",
        'confusion_matrix_path': cm_path
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    global last_metrics

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image = Image.open(filepath).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                prediction = class_names[predicted_class.item()]
                image_url = '/' + filepath

        elif 'train' in request.form:
            train_model(epochs=5)
            last_metrics = evaluate_model()

    return render_template('index.html',
                           prediction=prediction,
                           image_url=image_url,
                           metrics=last_metrics)

if __name__ == '__main__':
    app.run(debug=True)


