import argparse
from pathlib import Path

from evaluate import ArgsEvaluate

from graphgen.model import create_model
from utils import load_model, get_model_attribute


def count_params(path, eval_args):
    train_args = eval_args.train_args
    feature_map = get_model_attribute('feature_map', eval_args.model_path, eval_args.device)
    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)
    filename = path.parent / "num_params.txt"
    num_params = sum([sum(p.numel() for p in model[m].parameters() if p.requires_grad) for m in model])
    with open(filename, "w") as f:
        print(num_params, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    eval_args = ArgsEvaluate(path)
    count_params(path, eval_args)