import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # 2 output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load images
def load_image(img_path, transform=None):
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

# Set the path to the image directory and the CSV file
img_dir = r'D:\model'
csv_file = r'D:\Images\fypstreamlit\merged2.csv'

# Load the CSV file
df = pd.read_csv(csv_file)

# Sort the DataFrame based on a specific column or criteria
df_sorted = df.sort_values(by='image_id', ascending=False)  # Replace 'image_id' with the column you want to use for sorting

# Calculate the number of images for the train and test sets based on a specific ratio or count
train_size = int(len(df_sorted) * 0.75)  # Set the desired percentage or count for the train set

# Split the DataFrame into train and test DataFrames
train_df = df_sorted.iloc[:train_size]
test_df = df_sorted.iloc[train_size:]

# Define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
}

# Load images and labels
train_images = [load_image(os.path.join(img_dir, img + '.png'), transform=data_transforms['train']) for img in train_df.iloc[:, 0]]
train_labels = train_df.iloc[:, -1].tolist()
test_images = [load_image(os.path.join(img_dir, img + '.png'), transform=data_transforms['val']) for img in test_df.iloc[:, 0]]
test_labels = test_df.iloc[:, -1].tolist()

# Map labels to class indices
label_mapping = {'CE': 1, 'LAA': 0}
train_labels = [label_mapping[label.upper()] for label in train_labels]
print(train_labels)
test_labels = [label_mapping[label.upper()] for label in test_labels]
print(test_labels)

# Create the model instance
model = CNNModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(len(train_images)):
        image = train_images[i]
        label = torch.tensor(train_labels[i])

        # Forward pass
        outputs = model(image.unsqueeze(0))
        loss = criterion(outputs, label.unsqueeze(0))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_images)}")
    train_losses.append(epoch_loss / len(train_images))

# Evaluate the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    for i in range(len(test_images)):
        image = test_images[i]
        label = torch.tensor(test_labels[i])

        outputs = model(image.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        total += 1
        print(total)
        correct += (predicted == label).sum().item()
        print(correct)
        predicted_labels.append(predicted.item())
        print(predicted_labels)
        true_labels.append(label.item())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")

# Calculate metrics
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
loss = criterion(torch.tensor(predicted_labels), torch.tensor(true_labels, dtype=torch.long)).item()

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Loss:", loss)

# Plot the training loss curve
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
