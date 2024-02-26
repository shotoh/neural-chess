from keras import callbacks
from keras.src.callbacks import ModelCheckpoint

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


def create_dataset(games):
    ds = tf.data.Dataset.from_tensor_slices(games)
    ds = ds.batch(10240).map(get_random_move).filter(lambda x: x is not None).unbatch()
    return ds


if __name__ == '__main__':
    model = create_model()
    train_data = []
    train_labels = []
    with open('resources/output.txt', 'r') as f:
        lines = f.read().splitlines()
    packed_ds = create_dataset(lines)
    print(len(packed_ds))
    train_ds = packed_ds.skip(1024).take(51200).cache().shuffle(10240).repeat().batch(2048)
    checkpoint_filepath = 'checkpoints/'
    model_checkpointing_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
    )
    model.fit(train_data, train_labels, batch_size=2048, epochs=100,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                         callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=0.001),
                         model_checkpointing_callback])
    model.save('models/model_1.h5')
