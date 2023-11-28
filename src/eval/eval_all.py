import os
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute all metrics')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    name = cfg["training"]["model_name"][5:-3]

    os.system(
        "python src/eval/eval_ABC.py /data/nmaruani/RESULTS/Quadrics/ABC_{}_32".format(name))
    os.system(
        "python src/eval/eval_ABC.py /data/nmaruani/RESULTS/Quadrics/ABC_{}_64".format(name))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/Quadrics/Thingi_{}_32".format(name))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/Quadrics/Thingi_{}_64".format(name))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/Quadrics/Thingi_{}_128".format(name))
