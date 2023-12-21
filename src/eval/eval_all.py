import os
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute all metrics')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('-subd', type=int, default=0,
                        help='subdivision level.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    name = cfg["training"]["model_name"][5:-3]

    suffix = ""
    if args.subd == 1:
        suffix = "_lite"
    elif args.subd > 1:
        suffix = "_lite_{}".format(args.subd)

    os.system(
        "python src/eval/eval_ABC.py /data/nmaruani/RESULTS/PoNQ/ABC_{}_32{}".format(name, suffix))
    os.system(
        "python src/eval/eval_ABC.py /data/nmaruani/RESULTS/PoNQ/ABC_{}_64{}".format(name, suffix))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/PoNQ/Thingi_{}_32{}".format(name, suffix))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/PoNQ/Thingi_{}_64{}".format(name, suffix))
    os.system(
        "python src/eval/eval_THINGI.py /data/nmaruani/RESULTS/PoNQ/Thingi_{}_128{}".format(name, suffix))
