from cnn import CNN
from train import sample_games
import pandas as pd
from agent import Agent
import chess

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
  b = boards[0]
  print(b)

  agent_color = b.turn
  agent = Agent(agent_color)
  played_move = agent.play(b)
  print(f"played move is {played_move}")
  print(b)