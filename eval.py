from cnn import CNN
from train import sample_games
import pandas as pd

if __name__ == "__main__":
  data = pd.read_csv("./games.csv")

  model = CNN()
  model.load()
  model.eval()

  X, Y, boards = sample_games(data, 1, True)
  y = Y[0]
  val = model(X)

  print(f"winner is {'white' if y == 1 else 'black' if y == -1 else 'draw'}, nn output is {val.item()}")
  # X.shape
  print(boards[0])