#!/usr/bin/env python3

import torch

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                1, 32, kernel_size=5, padding=2),   # conv2d_2
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                  # max_pooling2d_2

            torch.nn.Conv2d(
                32, 48, kernel_size=5, padding=2),  # conv2d_3
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                  # max_pooling2d_3
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),                      # flatten_1
            torch.nn.Linear(1200, 256),              # dense_4 (7*7*48 = 2352 for 28x28 input)
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(256, 128),               # dense_5
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(128, 52),                # dense_6
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(52, 26),                 # dense_7
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x