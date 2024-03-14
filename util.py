import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--obj_path',
        type=str)
    return parser.parse_args()
    
def read_dotobj(file_path):
    model = {'v': [], 'vt': [], 'vn': [], 'f': []}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            if not line:
                continue

            if line[0] == 'v':
                model['v'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'vt':
                model['vt'].append((float(line[1]), float(line[2])))
            elif line[0] == 'vn':
                model['vn'].append((float(line[1]), float(line[2]), float(line[3])))
            elif line[0] == 'f':
                v1 = tuple(int(idx) for idx in line[1].split('/'))
                v2 = tuple(int(idx) for idx in line[2].split('/'))
                v3 = tuple(int(idx) for idx in line[3].split('/'))
                model['f'].append((v1, v2, v3))
    return model

