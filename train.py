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

def sample_full_game(moves, result):
  """
  Samples a full game trajectory from the dataset.
  Returns a list of board state tensors (trajectory) and the final game result.
  """
  moves = moves.split()
  final_score = 1 if result == "white" else 0 if result == "draw" else -1

  board = chess.Board()
  trajectory = [into_input(board)]

  for m in moves:
      try:
          board.push_san(m)
          trajectory.append(into_input(board))
      except Exception as e:
          # If an illegal move is encountered, break out
          break

  # Stack trajectory: shape (T, channels, 8, 8)
  return torch.stack(trajectory), final_score

def td_lambda_update_trajectory(model, trajectory, final_reward, optimizer, lambda_value=0.8, gamma=0.99):
    """
    Applies TD(λ) update on a single game trajectory.
    Assumes that intermediate rewards are 0 (only final reward matters).
    """
    X = trajectory
    # Get value estimates for all states
    V = model(X)  # shape (T, 1)
    T = len(trajectory)

    # Compute TD errors for each time step.
    # For non-terminal states, reward is 0.
    deltas = []
    for t in range(T - 1):
        # r_t is 0 for t < T-1, and gamma * V[t+1] gives the bootstrapped value.
        delta_t = gamma * V[t+1] - V[t] # note r(t) is zero for non terminal states
        deltas.append(delta_t)
    # Terminal TD error: no bootstrapping, only final reward.
    delta_T = final_reward - V[-1]
    deltas.append(delta_T)
    # Now deltas is a list of T tensors.

    # For each state t, compute the λ-return error:
    td_lambda_errors = []
    for t in range(T):
        error_t = 0
        power = 1.0
        for k in range(t, T):
            error_t += power * deltas[k]
            power *= lambda_value
        td_lambda_errors.append(error_t)
    td_lambda_errors = torch.stack(td_lambda_errors)  # shape (T, 1)

    # Loss: Mean squared error of the λ-return errors.
    loss = (td_lambda_errors ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def sample_games_at_different_moves(data, n  = 1000, ret_boards = False):
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

  moves = data['moves']
  winners = data['winner']

  total_epochs = 1000
  for epoch in range(total_epochs):
    # X, Y = sample_games_at_different_moves(data, 64)
    # Y = Y.view(-1, 1)

    # for each game
    loss = 0
    for i, game in enumerate(moves):
      result = winners[i]
      traj, final_res = sample_full_game(game, result)
      model.train()
      loss += td_lambda_update_trajectory(model, traj, final_res, optim, lambda_value=0.8, gamma=0.99)

      clear_temp_line()
      if i % 100 == 0:
        print(f"Epoch {epoch}, loss {loss.item()/max(1,i):.4f}")
        model.save()
      else:
        write_temp_line(f"{i}/{epoch}/{total_epochs}")

    # preds = model(X)
    # loss = loss_fn(preds, Y)

    # optim.zero_grad()
    # loss.backward()
    # optim.step()
