import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudDiscriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(PointCloudDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        # x: [B, N, 3] -> transpose to [B, 3, N]
        x = x.transpose(1, 2)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))  # [B, 256, N]

        x = torch.max(x, 2)[0]  # [B, 256]
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)  # [B, 1]
        return x

class ImprovedPointCloudDiscriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(ImprovedPointCloudDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 更强的全局表征
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: [B, N, 3]
        x = x.transpose(1, 2)  # [B, 3, N]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x).squeeze(-1)  # [B, 512]
        return self.fc(x)

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.leaky_relu(self.main(x) + self.shortcut(x), 0.2)

class EnhancedPointCloudDiscriminator(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.block1 = ResBlock1D(input_dim, 64)
        self.block2 = ResBlock1D(64, 128)
        self.block3 = ResBlock1D(128, 256)
        self.block4 = ResBlock1D(256, 512)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: [B, N, 3]
        x = x.transpose(1, 2)  # [B, 3, N]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).squeeze(-1)  # [B, 512]
        return self.fc(x)