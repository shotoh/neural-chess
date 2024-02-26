from keras import callbacks
from keras.src.callbacks import ModelCheckpoint

from utils import *


def get_random_move(game_tensor):
    moves = tf.strings.split(game_tensor, sep=' ')
    if len(moves) < 3:
        return None
    random_index = np.random.randint(0, len(moves) - 2)
    chess_board = chess.Board()
    for i in range(random_index):
        chess_board.push_uci(moves[i].numpy().decode())
    if random_index % 2 == 0:
        color = 'w'
    else:
        color = 'b'
    x = format_board(chess_board, color)
    y = format_move(moves[random_index])
    return x, y


def create_dataset():
    ds = tf.data.TextLineDataset('resources/output.txt')
    ds = ds.batch(10240).map(get_random_move).filter(lambda x: x is not None).unbatch()
    return ds


if __name__ == '__main__':
    model = create_model()
    print('Creating dataset...')
    packed_ds = create_dataset()
    print(len(packed_ds))
    packed_ds = packed_ds.skip(1024).take(51200).cache()
    train_ds = packed_ds.shuffle(10240).repeat().batch(2048)
    checkpoint_filepath = 'checkpoints/'
    model_checkpointing_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
    )
    model.fit(train_ds, batch_size=2048, epochs=100,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                         callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=0.001),
                         model_checkpointing_callback])
    model.save('models/model_1.h5')
