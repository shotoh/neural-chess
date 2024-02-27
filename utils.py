import re

import numpy as np

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


def format_board(board):
    layers = []
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    board = board.copy()
    for piece in pieces:
        layers.append(format_layer(board, 'w', piece))
    for piece in pieces:
        layers.append(format_layer(board, 'b', piece))
    return np.stack(np.float32(layers))


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

