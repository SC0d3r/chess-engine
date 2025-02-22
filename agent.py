import chess

class Agent:
  def __init__(self, color = chess.WHITE):
    self.color = color

  def play(self, board):
    # is not agent's turn
    if board.turn != self.color:
      return

    move = self._select_move(board)  
    assert(move in board.legal_moves, f"move {move} is not valid, legal moves are {board.legal_moves}")
    board.push_san(move)

  def _select_move(self, board):
    pass