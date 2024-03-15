import pandas as pd
import argparse
import random
import os

def load_finetuning_ids(finetuning_files_paths):
    finetuning_ids = []
    for file_path in finetuning_files_paths:
        df = pd.read_csv(file_path, sep='\t')
        finetuning_ids += df.doc_id.tolist()
    del df
    return list(set(finetuning_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language')
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-s', '--dataset_size', type=int) #, default=2000)
    args = parser.parse_args()

    all_paragraphs_path = f'../../data/src/all_paragraphs/{args.dataset}/{args.language}.tsv'
    finetuning_files_dir = f'../../data/src/train_eval_splits/{args.dataset}'
    finetuning_files_paths = [os.path.join(finetuning_files_dir, file_name) for file_name in os.listdir(finetuning_files_dir) if not file_name.startswith('.')]
    out_dir = f'../../data/src/probing_training_set/{args.dataset}'
    out_path = os.path.join(out_dir, f'{args.language}_probing.tsv')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    finetuning_ids = load_finetuning_ids(finetuning_files_paths)

    all_df = pd.read_csv(all_paragraphs_path, sep='\t')
    all_df = all_df[~all_df['doc_id'].isin(finetuning_ids)]
    all_df.to_csv(out_path, sep='\t', index=False)
    
    if args.dataset_size is not None:
        all_df = all_df.sample(n=args.dataset_size)

    all_df.to_csv(out_path, sep='\t', index=False)

    print(f'\n{args.dataset} = {len(all_df)}\n')


if __name__ == '__main__':
    main()