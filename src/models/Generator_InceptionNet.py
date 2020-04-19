import torch.nn as nn
import torch
import numpy as np
import torchvision.models as models


class Generator_InceptionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception2 = models.googlenet(pretrained=True, progress=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.layer_1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)  # Output would be 64 X 128 X 128
        self.layer_1_norm = nn.BatchNorm2d(64)
        self.layer_2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # Output would be 128 X 64 X 64
        self.layer_2_norm = nn.BatchNorm2d(128)
        self.layer_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  # Output would be 128 X 64 X 64
        self.layer_3_norm = nn.BatchNorm2d(128)
        self.layer_4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # Output would be 256 X 32 X 32
        self.layer_4_norm = nn.BatchNorm2d(256)
        self.layer_5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)  # Output would be 256 X 32 X 32
        self.layer_5_norm = nn.BatchNorm2d(256)

        self.decode_1 = nn.ConvTranspose2d(1256, 128, 4, stride=2, padding=1)  # Out is 128 X 64 X 64
        self.decode_norm_1 = nn.BatchNorm2d(128)
        self.decode_2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # Out is 64 X 128 X 128
        self.decode_norm_2 = nn.BatchNorm2d(64)
        self.decode_3 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)  # Out is 64 X 128 X 128
        self.decode_norm_3 = nn.BatchNorm2d(64)
        self.decode_4 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1)  # Out is 32 X 128 X 128
        self.decode_norm_4 = nn.BatchNorm2d(32)
        self.decode_5 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)  # Out is 3 X 256 X 256

        self._initialize_weights()

    def forward(self, x):
        # convert grayscale to rgb image with range in 0 to 1
        rgb = self.preprocess(x)
        inception_features = self.inception2(rgb)

        x = self.relu(self.layer_1_norm(self.layer_1(x)))
        x = self.relu(self.layer_2_norm(self.layer_2(x)))
        x = self.relu(self.layer_3_norm(self.layer_3(x)))
        x = self.relu(self.layer_4_norm(self.layer_4(x)))
        x = self.relu(self.layer_5_norm(self.layer_5(x)))
        x = self.fusion(x, inception_features)

        x = self.relu(self.decode_norm_1(self.decode_1(x)))
        x = self.relu(self.decode_norm_2(self.decode_2(x)))
        x = self.relu(self.decode_norm_3(self.decode_3(x)))
        x = self.relu(self.decode_norm_4(self.decode_4(x)))
        x = self.tanh(self.decode_5(x))

        return x

    def fusion(self, x, features):
        size = x.shape
        features = features.repeat(size[2], size[3], 1, 1)
        features = features.permute(2, 3, 0, 1)

        combined = torch.cat((x, features), 1)
        return combined

    def preprocess(self, batch_img):
        batch_img = (batch_img + 1) / 2
        batch_img_rgb = batch_img.repeat(1, 3, 1, 1)
        return batch_img_rgb

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
