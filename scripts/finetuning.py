from utils.training_utils import *
from utils.eval_utils import *

from transformers import set_seed
from datasets import load_dataset
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', choices=['en', 'es', 'id', 'it', 'nl'])
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-m', '--model')
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-r', '--learning_rate', type=float, default=2e-05)
    args = parser.parse_args()

    set_seed(42)

    logger = init_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device = {device}')
    logger.info(f'Learning rate = {args.learning_rate}')

    model_name = args.model
    data_files = {'train': f'../data/datasets/{args.dataset}/{args.language}_train.tsv',
                  'validation': f'../data/datasets/{args.dataset}/{args.language}_eval.tsv'}

    model_str = model_name.split('/')[-1]
    out_dir = f'../models/multi_lingual_{args.learning_rate}/{args.language}_{args.dataset}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(os.path.join(out_dir, 'predictions', 'classification_report.txt')):
        return

    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    logger.info(f"Dataset loaded!")

    train_model(model_name, dataset, logger, out_dir, args)
    save_evaluation_metrics(out_dir)


if __name__ == '__main__':
    main()
