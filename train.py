from random import random

import tensorflow as tf
import numpy as np
import re
import random
import chess

from utils import create_model

letter_to_number = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

number_to_letter = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h'
}

complement_color = {
    'w': 'b',
    'b': 'w'
}


def format_board(board, color):
    layers = []
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    if color == 'b':
        board.apply_transform(chess.flip_vertical)
    for piece in pieces:
        layers.append(format_layer(board, color, piece))
    for piece in pieces:
        layers.append(format_layer(board, complement_color[color], piece))
    return np.stack(layers)


def format_layer(board, color, piece):
    if color == 'w':
        piece = piece.upper()
    s = str(board)
    s = re.sub(f'[^{piece} \n]', '.', s)
    s = re.sub(f'\.', '0', s)
    s = re.sub(f'{piece}', '1', s)

    layer = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        layer.append(row)

    return np.array(layer)


def format_move(move):
    move_from = np.zeros((8, 8))
    move_from[8 - int(move[1])][letter_to_number[move[0]]] = 1
    move_to = np.zeros((8, 8))
    move_to[8 - int(move[3])][letter_to_number[move[2]]] = 1
    return np.stack([move_from, move_to])


def get_random_move(games):
    random_game = random.choice(games)
    moves = random_game.split(' ')
    if len(moves) < 4:
        return None
    random_index = random.randint(0, len(moves) - 2)
    chess_board = chess.Board()
    for move in moves[:random_index]:
        chess_board.push_uci(move)
    if random_index % 2 == 0:
        color = 'w'
    else:
        color = 'b'
    x = format_board(chess_board, color)
    y = format_move(moves[random_index])
    return x, y


if __name__ == '__main__':
    model = create_model()
    train_data = []
    train_labels = []
    with open('resources/output.txt', 'r') as f:
        lines = f.read().splitlines()
    for i in range(1000000):
        random_move = get_random_move(lines)
        if random_move is None:
            continue
        train_data.append(random_move[0])
        train_labels.append(random_move[1])
    train_data = tf.stack(train_data, axis=0)
    train_labels = tf.stack(train_labels, axis=0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="training_2/checkpoint.ckpt", verbose=1, save_weights_only=True, save_freq=5 * 32
    )
    model.fit(train_data, train_labels, epochs=10, callbacks=[cp_callback])
