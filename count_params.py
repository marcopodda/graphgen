import argparse
from pathlib import Path

from evaluate import ArgsEvaluate

from graphgen.model import create_model
from utils import load_model, get_model_attribute


def count_params(eval_args):
    train_args = eval_args.train_args
    feature_map = get_model_attribute('feature_map', eval_args.model_path, eval_args.device)
    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)
    filename = train_args.current_model_save_path / "num_params.txt"
    num_params = sum([sum(p.numel() for p in model[m].parameters() if p.requires_grad) for m in model])
    with open(filename, "wb") as f:
        print(num_params, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    eval_args = ArgsEvaluate(Path(args.path))
    count_params(eval_args)