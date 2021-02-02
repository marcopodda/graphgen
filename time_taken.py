import argparse
from pathlib import Path

import torch

from utils import strfdelta


def time_taken(path):
    args = torch.load(path)['saved_args']
    time_elapsed = strfdelta(args.end_time, "%H:%M:%S")
    filename = path.parent / "time_elapsed.txt"
    with open(filename, "w") as f:
        print(time_elapsed, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    time_taken(path)