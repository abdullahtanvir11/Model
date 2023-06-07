import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, optim
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# load your csv
df = pd.read_csv(r'D:\Images\fypstreamlit\merged2.csv')
img = r'D:\model'

# Sort the DataFrame based on a specific column or criteria
df_sorted = df.sort_values(by='image_id', ascending=False)  # Replace 'image_id' with the column you want to use for sorting

# Calculate the number of images for the train and test sets based on a specific ratio or count
train_size = int(len(df_sorted) * 0.95)  # Set the desired percentage or count for the train set
print(train_size)
test_size = len(df_sorted) - train_size
print(test_size)
# Split the DataFrame into train and test DataFrames
train_df = df_sorted.iloc[:train_size]
print(train_df)
test_df = df_sorted.iloc[train_size:]

# Reset the index of the train and test DataFrames if needed
# =============================================================================
# train_df = train_df.reset_index(drop=True)
# print(train_df)
# test_df = test_df.reset_index(drop=True)
# =============================================================================

# define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# function to load images
def load_image(img_path, transform=None):
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

import sys
# # load images
train_images = [load_image(os.path.join('D:\model', img + '.png'), transform=data_transforms['train']) for img in train_df.iloc[:, 0]]
print("Number of train images:", len(train_images))
# =============================================================================
# print(sys.getsizeof(train_images))
# =============================================================================
# =============================================================================
# print(train_images)
# =============================================================================
test_images = [load_image(os.path.join('D:\model', img+ '.png'), transform=data_transforms['val']) for img in test_df.iloc[:, 0]]


# Folder to save the images
save_folder = r'D:\Images\fypstreamlit\imagesCheck'

# Number of images to save
num_images_to_save = 30

# # Save train images
# for i, image_tensor in enumerate(train_images[:num_images_to_save]):
#     image = transforms.ToPILImage()(image_tensor)  # Convert Tensor to PIL.Image
#     image.save(os.path.join(save_folder, f'train_{i}.png'))

# # Save test images
# for i, image_tensor in enumerate(test_images[:num_images_to_save]):
#     image = transforms.ToPILImage()(image_tensor)  # Convert Tensor to PIL.Image
#     image.save(os.path.join(save_folder, f'test_{i}.png'))

# load labels
train_labels = train_df.iloc[:, -1].tolist()
test_labels = test_df.iloc[:, -1].tolist()

label_mapping = {'CE': 0, 'LAA': 1}
train_labels = [label_mapping[label.upper()] for label in train_labels]
print(train_labels)
test_labels = [label_mapping[label.upper()] for label in test_labels]
print(test_labels)
# load pre-trained model
# model = models.resnet50(pretrained=True)

# # modify final layer to fit your number of classes
# num_ftrs = model.fc.in_features
# num_classes = len(set(train_labels))  # number of unique labels in the training data
# model.fc = nn.Linear(num_ftrs, num_classes)

# # move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # define a loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # define the batch size
batch_size = 32

# # define the datasets
train_data = [(img, label) for img, label in zip(train_images, train_labels)]
print(train_data)
test_data = [(img, label) for img, label in zip(test_images, test_labels)]
print(test_data)
# # define the data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print(train_loader)
test_loader = DataLoader(test_data, batch_size=batch_size)
print(train_loader)

# # train the model
# model.train()
# for epoch in range(10):  # replace num_epochs with the number of epochs you want to train for
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(train_loader)
#     epoch_accuracy = 100 * correct / total
#     print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

# # evaluate the model
# model.eval()
# with torch.no_grad():
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     predicted_labels = []
#     true_labels = []
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         predicted_labels.extend(predicted.tolist())
#         true_labels.extend(labels.tolist())

#     test_loss = running_loss / len(test_loader)
#     test_accuracy = 100 * correct / total

# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.2f}%")

# # calculate precision, recall, F1 score, and confusion matrix for each class
# precision = precision_score(true_labels, predicted_labels, average=None)
# recall = recall_score(true_labels, predicted_labels, average=None)
# f1 = f1_score(true_labels, predicted_labels, average=None)
# confusion_mat = confusion_matrix(true_labels, predicted_labels)

# print(f"Class 1 (LAA) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}")
# print(f"Class 0 (CE) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}")
# print(f"Confusion Matrix:\n{confusion_mat}")


# class_names = ['LAA', 'CE']  # Replace with the class labels in your problem
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the model
# torch.save(model.state_dict(), 'model-final.pth')
