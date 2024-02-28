from collections import OrderedDict
from datetime import datetime

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from utils import *

LINEAR_SIZE = 2048
TRAINING_SIZE = 51200


class ChessDataset(IterableDataset):
    def __init__(self, positions, count):
        self.positions = positions
        self.count = count

    def __iter__(self):
        return self

    def __next__(self):
        return self[np.random.randint(0, len(self.positions))]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        random_position = self.positions[index].split(',')
        fen = random_position[0].split(' ')
        chess_board = chess.Board()
        chess_board.set_board_fen(fen[0])
        x = format_board(chess_board, fen[1])
        y = float(random_position[1])
        y = (y + 15) / 30
        return {'board': x, 'eval': y}


class ChessModel(pl.LightningModule):
    def __init__(self, positions=None):
        super().__init__()
        self.positions = positions
        layers = [(f'input', nn.Linear(769, LINEAR_SIZE))]
        for i in range(4):
            layers.append((f'lin{i}', nn.Linear(LINEAR_SIZE, LINEAR_SIZE)))
            layers.append((f'relu{i}', nn.ReLU()))
        layers.append((f'output', nn.Linear(LINEAR_SIZE, 1)))
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['board'], batch['eval']
        y_guess = self(x)
        y = y.unsqueeze(1)
        loss = nn.functional.l1_loss(y_guess, y)
        self.log('train_loss', loss)
        if batch_idx % 100 == 0:
            print(f'LOSS {loss}')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def train_dataloader(self):
        ds = ChessDataset(self.positions, TRAINING_SIZE)
        return DataLoader(ds, batch_size=32)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    print('Reading positions...')
    with open('resources/positions.txt', 'r') as f:
        chess_positions = f.read().splitlines()
    trainer = pl.Trainer(max_epochs=10)
    chess_model = ChessModel(chess_positions)
    trainer.fit(chess_model)
