import torch
import chess
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channel_map = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,

    # for black
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

def into_input(board):
  X = torch.zeros((18, 8, 8), dtype=torch.float).to(device)
  for i in range(64):
    piece = board.piece_at(i)
    if not piece:
      continue
    row, col = divmod(i, 8)
    X[channel_map[piece.symbol()], row, col] = 1
  
  # turn channel
  X[12, :, :] = 1 if board.turn == chess.WHITE else -1 
  # castling
  X[13, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
  X[14, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
  X[15, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
  X[16, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

  # en passant
  if board.has_legal_en_passant():
    row, col = divmod(board.ep_square, 8)
    X[17, row, col] = 1

  return X

def sample_games(data, n  = 1000, ret_boards = False):
  """
  this function samples n games from data
  and then creates a board with a random move on that sample
  and returns the (boards, win_or_lose)
  """
  _ret_boards = []
  boards = []
  scores = []
  moves = data['moves']
  winners = data['winner']
  for i in range(n):
    # print(f"len(moves) {len(moves)}")
    idx = random.randint(0, len(moves) - 1)
    g = moves[idx].split()
    # print(f"len(g) {len(g)}")
    b = chess.Board()
    how_many_moves = random.randint(1, max(1, len(g) - 1))
    mvs = g[:how_many_moves]
    for m in mvs:
      b.push_san(m)

    boards.append(into_input(b))
    _ret_boards.append(b)
    # game ending stat
    ges = winners[idx] # 'white', 'black', 'draw'
    scores.append(1 if ges == "white" else 0 if ges == "draw" else -1)

  X = torch.stack(boards).to(device)
  Y = torch.tensor(scores, dtype=torch.float).to(device)

  if ret_boards:
    return X, Y, _ret_boards

  return X, Y


if __name__ == "__main__":
  from cnn import CNN
  import argparse
  import pandas as pd
  from utils import clear_temp_line, write_temp_line
  data = pd.read_csv("./games.csv")

  parser = argparse.ArgumentParser(description="Use --fresh if you want a new training")
  parser.add_argument("-f", "--fresh", action="store_true", help="use --fresh if you want to train a new model")

  args = parser.parse_args()
  print(f"Running with args, fresh {args.fresh}")

  model = CNN() 

  if not args.fresh:
    model.load()

  alpha = 0.001
  loss_fn = torch.nn.MSELoss()
  optim = torch.optim.Adam(model.parameters(), lr=alpha)

  total_epochs = 100000
  for epoch in range(total_epochs):
    X, Y = sample_games(data, 64)
    Y = Y.view(-1, 1)

    model.train()

    preds = model(X)
    loss = loss_fn(preds, Y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    clear_temp_line()
    if epoch % 100 == 0:
      print(f"Epoch {epoch}, loss {loss.item():.4f}")
      model.save()
    else:
      write_temp_line(f"{epoch}/{total_epochs}")