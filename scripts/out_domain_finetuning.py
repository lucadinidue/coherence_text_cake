from utils.training_utils import *
from utils.eval_utils import *

from datasets import load_dataset
import argparse
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', choices=['en', 'es', 'id', 'it', 'nl'])
    parser.add_argument('-d', '--out_dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-m', '--model')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-r', '--learning_rate', type=float, default=2e-05)
    args = parser.parse_args()

    logger = init_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device = {device}')

    model_name = args.model

    dataset_list = ['wiki', 'fanfic', 'news', 'ted']
    train_datasets = [dataset for dataset in dataset_list if dataset != args.out_dataset]

    data_files = {
        'train': [f'../data/datasets/{train_dataset}/{args.language}_train.tsv' for train_dataset in train_datasets],
        'validation': f'../data/datasets/{args.out_dataset}/{args.language}_eval.tsv'}

    model_str = model_name.split('/')[-1]
    out_dir = f'../models/out_domain_{args.epochs}_epochs_{args.learning_rate}_lr/{model_str}/{args.language}_{args.out_dataset}'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    logger.info(f"Dataset loaded!")

    train_model(model_name, dataset, logger, out_dir, args)
    save_evaluation_metrics(out_dir)


if __name__ == '__main__':
    main()
