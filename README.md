# 🧠 MNIST Handwritten Digit Classification

## 📌 Overview
This repository contains a PyTorch-based neural network for classifying handwritten digits from the MNIST dataset. The model has been trained with optimizations like weight initialization, dropout, and learning rate scheduling.

## 📂 Repository Contents
- **mnist_model_code.ipynb** → Jupyter Notebook with training, validation, and testing code.
- **mnist_model.pth** → Trained PyTorch model weights for inference.

## ⚡ Model Architecture
- **Input Layer:** 28×28 flattened pixels
- **Hidden Layers:** Fully connected layers with ReLU activation and dropout
- **Output Layer:** 10 neurons (for digits 0–9)
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum
- **Loss Function:** CrossEntropyLoss

## 🚀 How to Use
1️⃣ **Setup Environment**  
   Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```

2️⃣ **Run the Jupyter Notebook**  
   Launch the notebook and execute each cell to train the model from scratch:
   ```bash
   jupyter notebook mnist_model_code.ipynb
   ```

3️⃣ **Load Pretrained Model for Inference**  
   To use the trained model without retraining, load the .pth file in Python:
   ```python
   import torch
   import torch.nn as nn

   # Define the same architecture as used in training
   class ImprovedNN(nn.Module):
       def __init__(self):
           super(ImprovedNN, self).__init__()
           self.fc1 = nn.Linear(28 * 28, 256)
           self.fc2 = nn.Linear(256, 128)
           self.fc3 = nn.Linear(128, 10)
           self.relu = nn.ReLU()
           self.dropout = nn.Dropout(0.2)
           
           # Kaiming Normal Initialization
           for m in self.modules():
               if isinstance(m, nn.Linear):
                   nn.init.kaiming_normal_(m.weight)
                   nn.init.constant_(m.bias, 0)

       def forward(self, x):
           x = x.view(x.size(0), -1)  # Flatten input
           x = self.relu(self.fc1(x))
           x = self.dropout(x)
           x = self.relu(self.fc2(x))
           x = self.fc3(x)  
           return x  # Raw logits

   # Load model
   model = ImprovedNN()
   model.load_state_dict(torch.load("mnist_model.pth"))
   model.eval()
   ```

4️⃣ **Make Predictions on New Images**  
   To classify a new handwritten digit image:
   ```python
   import torchvision.transforms as transforms
   from PIL import Image

   # Preprocessing
   transform = transforms.Compose([
       transforms.Grayscale(),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   # Load and preprocess image
   image = Image.open("digit.png")  # Replace with your image
   image = transform(image).view(1, -1)

   # Get prediction
   with torch.no_grad():
       output = model(image)
       predicted_digit = output.argmax(1).item()

   print(f"Predicted digit: {predicted_digit}")
   ```

## 📊 Training Performance
- **Accuracy:** ~98% on test data
- **Loss Reduction:** Gradual decline with mini-batch training
- **Regularization:** Improved generalization using dropout and L2 weight decay

## 📌 Next Steps
- Improve the model with CNNs for better feature extraction.
- Train on larger datasets like Fashion-MNIST or custom digit datasets.
- Deploy as a web app using Flask or FastAPI.

## 🔗 Connect with Me
- **GitHub:** [nasif1731](https://github.com/nasif1731)
- **LinkedIn:** [Nehal Asif](https://www.linkedin.com/in/nehal-asif)
```
