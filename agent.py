import chess
from cnn import CNN
from train import into_input
import math

class Agent:
  def __init__(self, color = chess.WHITE):
    self.color = color
    self.model = CNN()
    self.model.load()

  def play(self, board):
    # is not agent's turn
    if board.turn != self.color:
      return

    move = self._select_best_move(board)  

    assert(move in board.legal_moves, f"move {move} is not valid, legal moves are {board.legal_moves}")

    board.push_san(move)
    return move

  def _select_best_move(self, board, depth = 4):
    legal_moves = board.legal_moves
    
    best_score = -math.inf
    best_move = None

    alpha = -math.inf
    beta = math.inf

    for lm in legal_moves:
      board.push(lm)
      eval_score = self._minmax(board, alpha, beta, depth, False)

      if eval_score > best_score:
        best_score = eval_score
        best_move = lm

      board.pop()

    return best_move
  
  def _eval_board(self, board):
    input_tensor = into_input(board)
    self.model.eval()
    y = self.model(input_tensor.unsqueeze(0))
    return y.item()

  def _minmax(self, board, alpha=-math.inf, beta=math.inf, depth = 4, maximizing_player=False):
    """minmax algorithm with alpha-beta pruning"""

    # ending conditions
    if depth == 0 or board.is_game_over():
      return self._eval_board(board)

    legal_moves = board.legal_moves

    if maximizing_player:
      max_eval = -math.inf
      for lm in legal_moves:
        board.push(lm)
        eval_score = self._minmax(board, alpha, beta, depth - 1, False)
        board.pop()

        max_eval = max(max_eval, eval_score)
        alpha = max(alpha, eval_score)

        # cause alpha is now worst than beta we break here
        # cause maximaizng agent doesnt choose a worst action 
        if beta <= alpha: # Beta cutoff
          break

      return max_eval
    else:
      # mimizing player
      min_eval = math.inf
      for lm in legal_moves:
        print(f"lm in maximizing {lm}")
        board.push(lm)
        eval_score = self._minmax(board, alpha, beta, depth - 1, True)
        board.pop()

        min_eval = min(min_eval, eval_score)
        beta = min(beta, eval_score)

        # we break here cause the mimizing agent wont let the maximizng agent pick
        # a better move
        if beta <= alpha: # Alpha cutoff
          break
      return min_eval