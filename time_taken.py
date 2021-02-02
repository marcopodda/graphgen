import argparse
from pathlib import Path

from evaluate import ArgsEvaluate

from utils import strfdelta


def time_taken(path, train_args):
    delta = train_args.end_time
    time_elapsed = strfdelta(delta, "%H:%M%:%S")
    filename = path.parent / "time_elapsed.txt"
    with open(filename, "w") as f:
        print(time_elapsed, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    eval_args = ArgsEvaluate(path)
    time_taken(path, eval_args.train_args)