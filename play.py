import numpy as np
import tensorflow as tf
import chess
import chess.svg
from IPython.core.display_functions import display
from IPython.display import SVG

from train import format_board
from utils import create_model


def display_board(chess_board):
    print(chess_board)


def load_model():
    chess_model = create_model()
    latest = tf.train.latest_checkpoint('training_1')
    print(latest)
    chess_model.load_weights(latest)
    return chess_model


def get_move(chess_model, chess_board):
    chess_board = chess_board.copy()
    legal_moves = list(chess_board.legal_moves)
    for legal_move in legal_moves:
        chess_board.push_uci(str(legal_move))
        if chess_board.is_checkmate():
            return chess_board.pop()
        chess_board.pop()
    formatted_board = format_board(chess_board, 'b')
    prediction = chess_model.predict(tf.stack([formatted_board]))
    print(prediction)
    return prediction


if __name__ == '__main__':
    model = load_model()
    board = chess.Board()
    while not board.is_game_over():
        display_board(board)
        move = input('Enter your move: ')
        if move == 'quit':
            break
        board.push_uci(move)
        display_board(board)
        print('AI is thinking...')
        ai_move = get_move(model, board)
        board.push_uci(ai_move)
