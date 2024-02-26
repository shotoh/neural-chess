import chess
import chess.pgn

MINIMUM_ELO = 2000


def process_pgn(pgn_file, output_file):
    print('Processing started')
    input_stream = open(pgn_file, 'r')
    output_stream = open(output_file, 'w')
    index = 0
    while True:
        game = chess.pgn.read_game(input_stream)
        if game is None:
            break
        headers = game.headers
        if int(headers['WhiteElo']) < MINIMUM_ELO or int(headers['BlackElo']) < MINIMUM_ELO:
            continue
        line = ''
        for move in game.mainline_moves():
            line += move.uci() + ' '
        if len(line) < 100:
            continue
        output_stream.write(line + '\n')
        index += 1
        if index % 1000000 == 0:
            print(f'Processed {index} million games')
    print('Processing finished')


if __name__ == '__main__':
    process_pgn('resources/lichess_db_standard_rated_2023-01.pgn', 'resources/output.txt')
