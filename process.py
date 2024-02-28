import json


def process_pgn(pgn_file, output_file):
    print('Processing started')
    input_stream = open(pgn_file, 'r')
    output_stream = open(output_file, 'w')
    num = 0
    for line in input_stream:
        position = json.loads(line)
        fen = position['fen']
        try:
            score = position['evals'][0]['pvs'][0]['cp'] / 100
        except KeyError:
            mate = position['evals'][0]['pvs'][0]['mate']
            if mate > 0:
                score = 15
            else:
                score = -15
        score = min(score, 15)
        score = max(score, -15)
        output_stream.write(f'{fen},{score}\n')
        num += 1
        if num % 1000000 == 0:
            print(f'Processed {num} million positions')
    print('Processing finished')


if __name__ == '__main__':
    process_pgn('resources/lichess_db_eval.json', 'resources/positions.txt')
