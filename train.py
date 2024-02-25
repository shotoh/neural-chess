from utils import *


def get_random_move(games):
    random_game = games[np.random.randint(0, len(games))]
    moves = random_game.split(' ')
    if len(moves) < 3:
        return None
    random_index = np.random.randint(0, len(moves) - 2)
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
        if i % 10000 == 0:
            print(f'Creating training data... ({i / 10000}%)')
        random_move = get_random_move(lines)
        if random_move is None:
            continue
        train_data.append(random_move[0])
        train_labels.append(random_move[1])
    train_data = tf.stack(train_data, axis=0)
    train_labels = tf.stack(train_labels, axis=0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="training_4/checkpoint.ckpt", verbose=1, save_weights_only=True, save_freq=5 * 32
    )
    model.fit(train_data, train_labels, epochs=10, callbacks=[cp_callback])
