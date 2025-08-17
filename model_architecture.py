import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import os
import yaml
from typing import Dict, List, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientNetExplorationClassifier(nn.Module):
    """
    EfficientNet-B1 CNN architecture for map completion classification and Grad-CAM saliency extraction.
    """
    def __init__(self, num_classes: int = 2):
        super(EfficientNetExplorationClassifier, self).__init__()
        self.model = torchvision.models.EfficientNetB1(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.model.aux_head = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1280, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.model.extract_features(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1280)
        x = self.model.aux_head(x)
        return x

    def get_saliency_map(self, input_image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Get the saliency map using Grad-CAM.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_image)
            output = output.squeeze()
            target = target.squeeze()
            gradients = torch.autograd.grad(output.sum(), self.model.aux_head[0].weight, retain_graph=True)[0]
            gradients = gradients.detach()
            weights = self.model.aux_head[0].weight.detach()
            saliency_map = torch.sum(gradients * weights, dim=1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=input_image.size()[2:], mode='bilinear')
            return saliency_map

    def load_pretrained_weights(self, weights_path: str):
        """
        Load pre-trained weights from a file.
        """
        if os.path.exists(weights_path):
            logger.info(f'Loading pre-trained weights from {weights_path}')
            state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.load_state_dict(state_dict)
        else:
            logger.error(f'Pre-trained weights file {weights_path} not found')

    def fine_tune_model(self, device: torch.device, num_epochs: int, learning_rate: float, train_loader: DataLoader, val_loader: DataLoader):
        """
        Fine-tune the model on a dataset.
        """
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            logger.info(f'Starting epoch {epoch+1}')
            self.train()
            for batch in train_loader:
                input_image, target = batch
                input_image, target = input_image.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(input_image)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch in val_loader:
                    input_image, target = batch
                    input_image, target = input_image.to(device), target.to(device)
                    output = self(input_image)
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
                accuracy = correct / total
                logger.info(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')

    def evaluate_map(self, input_image: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
        """
        Evaluate the map completion classification.
        """
        self.eval()
        with torch.no_grad():
            output = self(input_image)
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
            report = classification_report(target, predicted, output_dict=True)
            cm = confusion_matrix(target, predicted)
            return accuracy, report, cm

class MapCompletionDataset(Dataset):
    """
    Map completion dataset.
    """
    def __init__(self, data_path: str, transform: transforms.Compose):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.targets = []
        for file in os.listdir(data_path):
            if file.endswith('.jpg'):
                self.images.append(os.path.join(data_path, file))
                self.targets.append(int(file.split('_')[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        target = self.targets[index]
        return image, target

def main():
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    data_path = config['data_path']
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = MapCompletionDataset(data_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = EfficientNetExplorationClassifier(num_classes=config['num_classes'])
    model.load_pretrained_weights(config['pretrained_weights_path'])

    # Fine-tune model
    model.fine_tune_model(device, config['num_epochs'], config['learning_rate'], train_loader, val_loader)

    # Evaluate map completion classification
    input_image = torch.randn(1, 3, 224, 224)
    target = torch.tensor([1])
    accuracy, report, cm = model.evaluate_map(input_image, target)
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Report: {report}')
    logger.info(f'Confusion Matrix: {cm}')

if __name__ == '__main__':
    main()