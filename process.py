import json


def process_pgn(pgn_file, output_file):
    print('Processing started')
    input_stream = open(pgn_file, 'r')
    output_stream = open(output_file, 'w')
    index = 0
    while True:
        position = json.load(input_stream)
        if position is None:
            break
        line = position['fen'] + ',' + position['evals'][0]['pvs'][0]['cp']
        output_stream.write(line + '\n')
        index += 1
        if index % 1000000 == 0:
            print(f'Processed {index} million positions')
    print('Processing finished')


if __name__ == '__main__':
    process_pgn('resources/lichess_db_eval.json', 'resources/positions.txt')
