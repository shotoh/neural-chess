import chess.svg
import numpy as np
import torch
from chess import IllegalMoveError

from utils import format_board
from train import ChessModel
from chessboard import display

MODEL = 'lightning_logs/version_12/checkpoints/epoch=9-step=16000.ckpt'
DEPTH = 3


def minimax_eval(model, board):
    board = format_board(board, 'b')
    eval = model(torch.from_numpy(board).float())
    print(eval)
    return eval


def minimax(model, board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(model, board)
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(model, board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(model, board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval


def get_move(model, board, depth):
    min_move = None
    min_eval = np.inf
    for move in board.legal_moves:
        board.push(move)
        evaluation = minimax(model, board, depth - 1, -np.inf, np.inf, True)
        board.pop()
        if evaluation < min_eval:
            min_eval = evaluation
            min_move = move
    print(f'{min_move}: {min_eval}')
    return min_move


if __name__ == '__main__':
    chess_model = ChessModel.load_from_checkpoint(MODEL)
    chess_model.eval()
    chess_board = chess.Board()
    chess_display = display.start(chess_board.fen())
    while not chess_board.is_game_over():
        try:
            print(chess_board)
            player_move = input('Enter your move: ')
            if player_move == 'quit':
                break
            chess_board.push_uci(player_move)
            print(chess_board)
            display.update(chess_board.fen(), chess_display)
            print('AI is thinking...')
            ai_move = get_move(chess_model, chess_board, DEPTH)
            chess_board.push(ai_move)
            display.update(chess_board.fen(), chess_display)
        except ValueError:
            print('Invalid move!')
    print('Game over')
    display.terminate()