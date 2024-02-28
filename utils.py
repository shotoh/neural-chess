import chess
import numpy as np

colors = [chess.WHITE, chess.BLACK]
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def format_board(board, current):
    vectors = []
    for color in colors:
        for piece in pieces:
            v = np.zeros(64)
            for i in list(board.pieces(piece, color)):
                v[i] = 1
            vectors.append(v)
    if current == 'w':
        vectors.append([0])
    if current == 'b':
        vectors.append([1])
    return np.float32(np.concatenate(vectors))

