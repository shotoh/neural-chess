import chess
import tensorflow as tf
import numpy as np
import re

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
    board = board.copy()
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


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 8, 8)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='softmax'),
        tf.keras.layers.Reshape((2, 8, 8))
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    return model
