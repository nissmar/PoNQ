import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate CNN')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    os.system(
        "python src/utils/generate_mesh_CNN.py {} -grid_n 33".format(args.config))
    os.system(
        "python src/utils/generate_mesh_CNN.py {} -grid_n 65".format(args.config))
    os.system(
        "python src/utils/generate_mesh_CNN.py {} -dataset Thingi -grid_n 33".format(args.config))
    os.system(
        "python src/utils/generate_mesh_CNN.py {} -dataset Thingi -grid_n 65".format(args.config))
    os.system(
        "python src/utils/generate_mesh_CNN.py {} -dataset Thingi -grid_n 129".format(args.config))
