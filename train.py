from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

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


class ChessDataset(Dataset):
    def __init__(self, games):
        self.games = games

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        move = get_random_move(self.games)
        return move


class ChessLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(384, 384, 3, padding=1)
        self.bn = nn.BatchNorm2d(384)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Conv2d(12, 384, 3, padding=1)
        self.layers = nn.ModuleList([ChessLayer() for _ in range(4)])
        self.output_layer = nn.Conv2d(384, 2, 3, padding=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def train_one_epoch(model, dl, optimizer, loss_fn, tb_writer, epoch_index):
    running_loss = 0
    last_loss = 0
    for i, data in enumerate(dl):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'BATCH {i + 1} LOSS: {last_loss}')
            tb_writer.add_scalar('Loss/train', last_loss, epoch_index * len(dl) + i + 1)
            running_loss = 0
    return last_loss


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    with open('resources/output.txt', 'r') as f:
        lines = f.read().splitlines()
        print('Creating dataset...')
    train_ds = ChessDataset(lines)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    train_model = ChessModel()
    train_optimizer = torch.optim.SGD(train_model.parameters(), lr=0.001, momentum=0.9)
    train_loss_fn = torch.nn.CrossEntropyLoss()
    train_writer = SummaryWriter(f'runs/trainer_{timestamp}')
    best_vloss = 1000000
    for epoch in range(5):
        print(f'EPOCH {epoch + 1}:')
        train_model.train(True)
        avg_loss = train_one_epoch(train_model, train_dl, train_optimizer, train_loss_fn, train_writer, epoch)
        running_vloss = 0
        train_model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(train_dl):
                vinputs, vlabels = vdata
                voutputs = train_model(vinputs)
                vloss = train_loss_fn(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        train_writer.add_scalars('Training vs. Validation loss',
                                 {'Training': avg_loss, 'Validation': avg_vloss},
                                 epoch + 1)
        train_writer.flush()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(train_model, f'models/model_{timestamp}_{epoch}.pth')
