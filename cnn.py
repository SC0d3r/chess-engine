import torch

# formula for O = ((I - K + 2 * P) / S) + 1
# size here is 8x8
# default padding is 0 and default stride is 1
# round down if the division is not a complete number
# I = input size
# K = kernel size
# P = padding
# S = stride

class ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels):
    super(ResidualBlock, self).__init__()
    self.cnv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    self.bn1 = torch.nn.BatchNorm2d(in_channels)

    self.cnv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    self.bn2 = torch.nn.BatchNorm2d(in_channels)
    
  def forward(self, x):
    out = torch.relu(self.bn1(self.cnv1(x)))
    out = torch.relu(self.bn2(self.cnv2(out)) + x) # adding residual connection
    return out


class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnv1 = torch.nn.Conv2d(18, 64, kernel_size=3, padding=1)
    self.bn1 = torch.nn.BatchNorm2d(64)

    self.cnv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn2 = torch.nn.BatchNorm2d(128)

    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # make the output 8x8 -> 4x4

    # stack of residual blocks
    self.res1 = ResidualBlock(128)
    self.res2 = ResidualBlock(128)
    self.res3 = ResidualBlock(128)
    self.res4 = ResidualBlock(128)

    # this makes the output from 8x8 to 1x1
    self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    # self.f1 = torch.nn.Linear(128 * 8 * 8, 128)
    # self.f1 = torch.nn.Linear(128, 128)
    # self.bn4 = torch.nn.BatchNorm1d(128)
    # self.dropout = torch.nn.Dropout(p = 0.3)

    self.f2 = torch.nn.Linear(128, 1)

  def forward(self, x):
    x = torch.relu(self.bn1(self.cnv1(x))) # batch normalization before activation
    x = torch.relu(self.bn2(self.cnv2(x)))

    x = self.maxpool(x)

    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)

    x = self.global_avg_pool(x)

    # x = torch.relu(self.bn2(self.cnv2(x)))
    # x = torch.relu(self.bn3(self.cnv3(x)))

    x = x.view(x.size(0), -1)

    # x = torch.relu(self.bn4(self.f1(x)))
    # x = self.dropout(x) # dropout after activation

    x =  torch.tanh(self.f2(x))  # [-1, 1]
    return x

  def save(self, path = "./CHESS_ENGINE_WEIGHTS"):
    torch.save(self.state_dict(), path + ".pt")
    # print("Model saved successfully ...")

  def load(self, path = "./CHESS_ENGINE_WEIGHTS"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.load_state_dict(torch.load(path + ".pt", weights_only=True, map_location=device))
    self.eval()
    self.to(device)
    print("Model loaded successfully ...")