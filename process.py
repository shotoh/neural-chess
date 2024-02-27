import json


def process_pgn(pgn_file, output_file):
    print('Processing started')
    input_stream = open(pgn_file, 'r')
    output_stream = open(output_file, 'w')
    num = 0
    for line in input_stream:
        position = json.loads(line)
        try:
            score = position["evals"][0]["pvs"][0]["cp"]
        except KeyError:
            continue
        output_stream.write(f'{position["fen"]},{score}\n')
        num += 1
    if num % 1000000 == 0:
        print(f'Processed {num} million positions')
    print('Processing finished')


if __name__ == '__main__':
    process_pgn('resources/lichess_db_eval.json', 'resources/positions.txt')
