from cnn import CNN
from train import sample_games_at_different_moves, into_input
import pandas as pd
from agent import Agent
import chess

if __name__ == "__main__":
  data = pd.read_csv("./games.csv")

  model = CNN()
  model.load()
  model.eval()

  X, Y, boards = sample_games_at_different_moves(data, 1, True)
  y = Y[0]
  val = model(X)

  print(f"winner is {'white' if y == 1 else 'black' if y == -1 else 'draw'}, nn output is {val.item()}")
  # X.shape
  b = boards[0]
  print(b)

  agent_color = b.turn
  agent = Agent(agent_color)
  played_move, best_score = agent.play(b, 3)

  board_score = model(into_input(b).unsqueeze(0))

  print(f"turn {'black' if agent_color == chess.BLACK else 'white'}, played move is {played_move}, best_score {best_score}, CNN board score {board_score.item()}")
  print(b)