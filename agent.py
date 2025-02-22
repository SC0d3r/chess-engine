import chess
from cnn import CNN
from train import into_input
import math
import random

class Node:
    def __init__(self, board, move=None, parent=None):
        # Store a copy of the board for this node.
        self.board = board.copy()
        self.move = move          # The move that led to this node (None for the root).
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0            # Sum of evaluations (reward) for this node.
        # Initially, all legal moves are untried.
        self.untried_moves = list(self.board.legal_moves)

class Agent:
    def __init__(self, color=chess.WHITE):
        self.color = color
        self.model = CNN()
        self.model.load()

    def play(self, board, depth=2):
        # Standard minimax play.
        if board.turn != self.color:
            return

        best_move, best_score = self._select_best_move(board, depth=depth)

        assert best_move in board.legal_moves, f"move {best_move} is not valid, legal moves are {board.legal_moves}"
        board.push(best_move)

        return best_move, best_score

    def playMCTS(self, board, iterations=1000):
        """
        Plays a move using Monte Carlo Tree Search.
        This method builds a search tree starting at the current board state,
        runs a number of iterations of selection, expansion, evaluation (using CNN), 
        and backpropagation, then selects the move with the highest visit count.
        """
        if board.turn != self.color:
            return

        root = Node(board)
        for i in range(iterations):
            # 1. Selection & Expansion
            node = self._tree_policy(root)
            # 2. Evaluation (Simulation/Rollout using the CNN evaluation function)
            reward = self._default_policy(node.board)
            # 3. Backpropagation
            self._backup(node, reward)
        
        # Choose the move from the root that was visited the most.
        best_child = max(root.children, key=lambda c: c.visits)
        move = best_child.move
        board.push(move)
        return move, best_child.value

    def _tree_policy(self, node):
        """
        Traverse the tree until we reach a leaf node that is either terminal or has untried moves.
        If the node has untried moves, expand by creating a new child.
        Otherwise, select the best child using the UCT (Upper Confidence Bound for Trees) formula.
        """
        while not node.board.is_game_over():
            if node.untried_moves:
                return self._expand(node)
            else:
                node = self._best_child(node, exploration_constant=1.4)
        return node

    def _expand(self, node):
        """Expand the node by taking one untried move and creating a child node."""
        move = node.untried_moves.pop()
        new_board = node.board.copy()
        new_board.push(move)
        child_node = Node(new_board, move, node)
        node.children.append(child_node)
        return child_node

    def _best_child(self, node, exploration_constant):
        """
        Select a child node with the highest UCT value.
        UCT = (child.value / child.visits) + exploration_constant * sqrt(ln(parent.visits) / child.visits)
        If a child hasn't been visited yet, its UCT value is treated as infinity.
        """
        best_value = -math.inf
        best_child = None
        for child in node.children:
            if child.visits == 0:
                uct_value = math.inf
            else:
                uct_value = (child.value / child.visits) + exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

    def _default_policy(self, board):
        """
        Instead of doing a full random rollout, use the CNN evaluation function to get a value for the board.
        This function acts as our simulation result.
        """
        return self._eval_board(board)

    def _backup(self, node, reward):
        """
        Propagate the reward up the tree.
        Increase the visit count and add the evaluation (reward) to each node in the path.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _eval_board(self, board):
        """
        Evaluate the board using the CNN model.
        The input is preprocessed by into_input.
        """
        input_tensor = into_input(board)
        self.model.eval()
        y = self.model(input_tensor.unsqueeze(0))
        return y.item()

    def _select_best_move(self, board, depth=2):
        """
        Your existing minimax with alpha-beta pruning implementation.
        """
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

            alpha = max(alpha, best_score)
            board.pop()

        return best_move, best_score

    def _minmax(self, board, alpha, beta, depth, maximizing_player=False):
        """Minimax algorithm with alpha-beta pruning."""
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
                if beta <= alpha:  # Beta cutoff
                    break
            return max_eval
        else:
            min_eval = math.inf
            for lm in legal_moves:
                board.push(lm)
                eval_score = self._minmax(board, alpha, beta, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:  # Alpha cutoff
                    break
            return min_eval
