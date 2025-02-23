import torch

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnv1 = torch.nn.Conv2d(18, 32, kernel_size=3, padding=1)
    self.bn1 = torch.nn.BatchNorm2d(32)
    self.cnv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn2 = torch.nn.BatchNorm2d(64)
    self.cnv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3 = torch.nn.BatchNorm2d(128)
    self.f1 = torch.nn.Linear(128 * 8 * 8, 128)
    self.bn4 = torch.nn.BatchNorm1d(128)
    self.f2 = torch.nn.Linear(128, 1)
    self.dropout = torch.nn.Dropout(p = 0.3)

  def forward(self, x):
    x = torch.relu(self.bn1(self.cnv1(x))) # batch normalization before activation
    x = torch.relu(self.bn2(self.cnv2(x)))
    x = torch.relu(self.bn3(self.cnv3(x)))
    x = x.view(x.size(0), -1)
    x = torch.relu(self.bn4(self.f1(x)))
    x = self.dropout(x) # dropout after activation
    x =  torch.tanh(self.f2(x))  # [-1, 1]
    return x

  def save(self, path = "./CHESS_ENGINE_WEIGHTS"):
    torch.save(self.state_dict(), path + ".pt")
    # print("Model saved successfully ...")

  def load(self, path = "./CHESS_ENGINE_WEIGHTS"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.load_state_dict(torch.load(path + ".pt", weights_only=True, map_location=device))
    self.eval()
    print("Model loaded successfully ...")