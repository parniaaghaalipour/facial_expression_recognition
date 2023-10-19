import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class HyperParameters:
    """Store hyperparameters in one place."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 197 * 197
    num_epochs = 5
    num_classes = 7
    batch_size = 128
    learning_rate = 0.001
    train_dataset_path = 'dataset/train'
    test_dataset_path = 'dataset/test'
    model_save_path = 'model.pth'


class DatasetLoader:
    """Preprocess and load the images dataset."""

    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(197),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def load_data(self):
        dataset = torchvision.datasets.ImageFolder(root=self.path,
                                                   transform=self.transform)
        return DataLoader(dataset=dataset,
                          batch_size=HyperParameters.batch_size,
                          shuffle=True)


class ConvolutionalNetwork(nn.Module):
    """Create Convolutional Neural Network model."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(49*49*32, HyperParameters.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


def train_model(model, train_loader, optimizer, loss_function):
    """Train the model."""

    total_step = len(train_loader)
    for epoch in range(HyperParameters.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(HyperParameters.device)
            labels = labels.to(HyperParameters.device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, HyperParameters.num_epochs, i + 1, total_step, loss.item()))


def main():
    """Entry point for the script."""

    # Initialize data loaders
    train_data = DatasetLoader(HyperParameters.train_dataset_path).load_data()
    test_data = DatasetLoader(HyperParameters.test_dataset_path).load_data()

    # Initialize Neural Network model
    model = ConvolutionalNetwork().to(HyperParameters.device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=HyperParameters.learning_rate)

    # Train the model
    train_model(model, train_data, optimizer, criterion)

    # Save the model
    torch.save(model.state_dict(), HyperParameters.model_save_path)


if __name__ == "__main__":
    main()
