import chess.svg
import numpy as np
import tensorflow as tf

from train import format_board, get_random_move
from utils import create_model, letter_to_number


def display_board(chess_board):
    print(chess_board)


def load_model():
    chess_model = create_model()
    latest = tf.train.latest_checkpoint('training_3')
    print(latest)
    chess_model.load_weights(latest)
    return chess_model


def get_move(chess_model, chess_board):
    chess_board = chess_board.copy()
    formatted_board = format_board(chess_board, 'b')
    prediction = chess_model.predict(tf.stack([formatted_board]))
    legal_moves = list(chess_board.legal_moves)
    weights = []
    for legal_move in legal_moves:
        chess_board.push_uci(str(legal_move))
        if chess_board.is_checkmate():
            return chess_board.pop()
        chess_board.pop()
        weights.append(calculate_weight(prediction, legal_move))
    best_move = np.random.choice(legal_moves, p=weights)
    print(f'{best_move}, best weight: {weights[best_move]}')
    return best_move


def calculate_weight(prediction, legal_move):
    legal_move = str(legal_move)
    move_from = [8 - int(legal_move[1])][letter_to_number[legal_move[0]]]
    move_to = [8 - int(legal_move[3])][letter_to_number[legal_move[2]]]
    return prediction[0][move_from] * prediction[1][move_to]


if __name__ == '__main__':
    model = load_model()
    board = chess.Board()

    train_data = []
    train_labels = []
    with open('resources/output.txt', 'r') as f:
        lines = f.read().splitlines()
    for i in range(1000):
        random_move = get_random_move(lines)
        if random_move is None:
            continue
        train_data.append(random_move[0])
        train_labels.append(random_move[1])
    train_data = tf.stack(train_data, axis=0)
    train_labels = tf.stack(train_labels, axis=0)
    loss, acc = model.evaluate(train_data, train_labels, verbose=2)
    print("restored model, acc {:5.2f}".format(acc))
    print("loss: {loss}".format(loss=loss))



    while not board.is_game_over():
        display_board(board)
        move = input('Enter your move: ')
        if move == 'quit':
            break
        board.push_uci(move)
        display_board(board)
        print(format_board(board, 'b'))
        print('AI is thinking...')
        ai_move = get_move(model, board)
        board.push_uci(ai_move)
