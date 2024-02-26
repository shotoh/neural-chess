import chess.svg
import numpy as np
import tensorflow as tf

from train import format_board, get_random_move
from utils import create_model, letter_to_number


def display_board(chess_board):
    print(chess_board)


def load_model():
    chess_model = create_model()
    latest = tf.train.latest_checkpoint('training_5')
    print(latest)
    chess_model.load_weights(latest)
    return chess_model


def get_move(chess_model, chess_board, color):
    chess_board = chess_board.copy()
    formatted_board = format_board(chess_board, 'b')
    prediction = chess_model.predict(tf.stack([formatted_board]))[0]
    if color == 'b':
        prediction[0] = np.fliplr(np.flipud(prediction[0]))
        prediction[1] = np.fliplr(np.flipud(prediction[1]))
    legal_moves = list(chess_board.legal_moves)
    weights = []
    sum_weight = 0
    for legal_move in legal_moves:
        chess_board.push_uci(str(legal_move))
        if chess_board.is_checkmate():
            return chess_board.pop()
        chess_board.pop()
        weight = calculate_weight(prediction, legal_move)
        weights.append(weight)
        sum_weight += weight
    for index in range(len(weights)):
        if sum_weight == 0:
            weights[index] = 1 / len(weights)
        else:
            weights[index] = weights[index] / sum_weight
    print(legal_moves)
    print(weights)
    print(prediction)
    best_move = np.random.choice(legal_moves, p=weights)
    print(f'{str(best_move)}, best weight: {weights[legal_moves.index(best_move)]}')
    return str(best_move)


def calculate_weight(prediction, legal_move):
    legal_move = str(legal_move)
    move_from = prediction[0][8 - int(legal_move[1])][letter_to_number[legal_move[0]]]
    move_to = prediction[1][8 - int(legal_move[3])][letter_to_number[legal_move[2]]]
    return move_from * move_to


def evaluate_model(chess_model):
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
    loss, acc = chess_model.evaluate(train_data, train_labels, verbose=2)
    print(f'loss: {loss}, acc: {acc}')


if __name__ == '__main__':
    model = load_model()
    board = chess.Board()
    evaluate_model(model)
    while not board.is_game_over():
        try:
            display_board(board)
            move = input('Enter your move: ')
            if move == 'quit':
                break
            board.push_uci(move)
            display_board(board)
            print('AI is thinking...')
            ai_move = get_move(model, board, 'b')
            board.push_uci(ai_move)
        except ValueError:
            print('Invalid move, try again!')
