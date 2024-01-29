from typing import Dict, List
import pandas as pd
import argparse
import random
import os


def load_dataframe(src_path: str) -> pd.DataFrame:
    df = pd.read_csv(src_path, sep='\t')
    return df


def count_documents_paragraphs(df: pd.DataFrame) -> Dict:
    counts = df['doc_id'].value_counts()
    counts = dict(zip(counts.index, counts.values))
    return counts


def split_ids(doc_counts: Dict, train_size: int, eval_size: int) -> (List, List):
    train_ids, eval_ids = [], []
    int_train_size, int_eval_size = 0, 0

    doc_ids = list(doc_counts.keys())
    random.shuffle(doc_ids)

    for doc_id in doc_ids:
        if int_train_size < train_size:
            train_ids.append(doc_id)
            int_train_size += doc_counts[doc_id]
        elif int_eval_size < eval_size:
            eval_ids.append(doc_id)
            int_eval_size += doc_counts[doc_id]
        else:
            return train_ids, eval_ids
    raise Exception(f'Failed to divide dataset: train_size = {int_train_size}, eval_size = {int_eval_size}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language')
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-t', '--train_size', type=int, default=8000)
    parser.add_argument('-e', '--eval_size', type=int, default=2000)
    args = parser.parse_args()

    src_path = f'../../data/src/all_paragraphs/{args.dataset}/{args.language}.tsv'
    out_dir = f'../../data/src/train_eval_splits/{args.dataset}'
    train_out_path = os.path.join(out_dir, f'{args.language}_train.tsv')
    eval_out_path = os.path.join(out_dir, f'{args.language}_eval.tsv')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = load_dataframe(src_path)
    doc_counts = count_documents_paragraphs(df)

    train_ids, eval_ids = split_ids(doc_counts, args.train_size, args.eval_size)

    train_df = df[df['doc_id'].isin(train_ids)].head(args.train_size)
    eval_df = df[df['doc_id'].isin(eval_ids)].head(args.eval_size)

    train_df.to_csv(train_out_path, sep='\t', index=False)
    eval_df.to_csv(eval_out_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
