import tkinter as tk
import chess
from agent import Agent  # your agent.py should be in the same directory

# Mapping from piece letters to Unicode chess symbols.
PIECE_UNICODE = {
    'K': "\u2654", 'Q': "\u2655", 'R': "\u2656",
    'B': "\u2657", 'N': "\u2658", 'P': "\u2659",
    'k': "\u265A", 'q': "\u265B", 'r': "\u265C",
    'b': "\u265D", 'n': "\u265E", 'p': "\u265F",
}

class ChessGUI(tk.Tk):
    def __init__(self, human_color=chess.WHITE, use_minimax=False, minimax_depth=2, mcts_iterations=1000):
        super().__init__()
        # Determine the agent's color: the opposite of the human's.
        agent_color = chess.WHITE if human_color == chess.BLACK else chess.BLACK
        print(f"Human plays {'White' if human_color == chess.WHITE else 'Black'}, Agent plays {'White' if agent_color == chess.WHITE else 'Black'}")
        print(f"Agent uses {'Minimax' if use_minimax else 'MCTS'} to play, Minimax Depth {minimax_depth}, MCTS iterations {mcts_iterations}")
        self.title("Chess vs Agent")
        self.square_size = 60
        self.canvas_size = self.square_size * 8
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        self.minimax_depth = minimax_depth
        self.mcts_iterations = mcts_iterations
          
        # Initialize board, agent, and last move.
        self.board = chess.Board()
        self.human_color = human_color
        self.agent = Agent(agent_color)
        self.use_minimax = use_minimax
        self.selected_square = None
        self.last_move = None  # Will hold the last move played

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.update_board()

        # If it's not the human's turn (e.g. human plays black), let the agent move.
        if self.board.turn != self.human_color:
            self.after(100, self.agent_move)

    def update_board(self):
        self.draw_board()
        if self.selected_square is not None:
            self.draw_legal_moves()
        self.draw_pieces()

    def draw_board(self):
        self.canvas.delete("square")
        light_color = "#F0D9B5"
        dark_color = "#B58863"
        # Draw the squares.
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = light_color if (file + rank) % 2 == 0 else dark_color
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="square")
        
        # Highlight the squares from the last move.
        if self.last_move is not None:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill="#d07050", outline="", tags="square")
        
        # If a square is selected, draw a blue border.
        if self.selected_square is not None:
            file = chess.square_file(self.selected_square)
            rank = chess.square_rank(self.selected_square)
            x1 = file * self.square_size
            y1 = (7 - rank) * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3, tags="square")

    def draw_legal_moves(self):
        self.canvas.delete("legal")
        for move in self.board.legal_moves:
            if move.from_square == self.selected_square:
                file = chess.square_file(move.to_square)
                rank = chess.square_rank(move.to_square)
                # Compute center of the destination square.
                cx = file * self.square_size + self.square_size / 2
                cy = (7 - rank) * self.square_size + self.square_size / 2
                r = 5  # radius
                fill = "#dAb583"
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=fill, outline=fill, tags="legal")

    def draw_pieces(self):
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x = file * self.square_size + self.square_size / 2
                y = (7 - rank) * self.square_size + self.square_size / 2

                if piece.color == chess.WHITE:
                    # Use the black glyph for white pieces (with a black outline for contrast)
                    symbol = PIECE_UNICODE[piece.symbol().lower()]
                    self.canvas.create_text(
                        x + 1, y + 1, text=symbol, font=("Arial", 32),
                        tags="piece", fill="black"
                    )
                    fill_color = "white"
                else:
                    symbol = PIECE_UNICODE[piece.symbol()]
                    fill_color = "black"

                self.canvas.create_text(
                    x, y, text=symbol, font=("Arial", 32),
                    tags="piece", fill=fill_color
                )

    def on_canvas_click(self, event):
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)
        clicked_square = chess.square(file, rank)
        
        # Only allow human moves.
        if self.board.turn != self.human_color:
            return

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece is not None and piece.color == self.human_color:
                self.selected_square = clicked_square
                self.update_board()
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if move not in self.board.legal_moves:
                move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move  # record the last move
                self.selected_square = None
                self.update_board()
                self.after(100, self.agent_move)
            else:
                # If the clicked square has another human piece, change selection.
                piece = self.board.piece_at(clicked_square)
                if piece is not None and piece.color == self.human_color:
                    self.selected_square = clicked_square
                    self.update_board()
                else:
                    self.selected_square = None
                    self.update_board()

    def agent_move(self):
        if self.board.is_game_over():
            self.show_game_over()
            return
        move, score = (self.agent.play(self.board, self.minimax_depth)
                       if self.use_minimax else self.agent.playMCTS(self.board, self.mcts_iterations))
        if move is not None:
            self.last_move = move
            self.update_board()
            print(f"move {move}, score {score}")
            if self.board.is_game_over():
                self.show_game_over()

    def show_game_over(self):
        result = self.board.result()
        self.canvas.create_text(
            self.canvas_size / 2, self.canvas_size / 2,
            text=f"Game Over: {result}",
            font=("Arial", 24),
            fill="red",
            tags="gameover"
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimax", action="store_true", help="use the minimax search instead of MCTS")
    parser.add_argument("--md", type=int, default=2, help="minimax depth")
    parser.add_argument("--mi", type=int, default=1000, help="MCTS iterations")
    parser.add_argument("--side", choices=["white", "black"], default="white", help="choose your side: white or black")
    args = parser.parse_args()

    human_color = chess.WHITE if args.side == "white" else chess.BLACK
    gui = ChessGUI(human_color, args.minimax, args.md, args.mi)
    gui.mainloop()
