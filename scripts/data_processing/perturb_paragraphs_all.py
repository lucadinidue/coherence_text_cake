from itertools import combinations
from tqdm import tqdm
import pandas as pd
import argparse
import ast
import os


def write_to_out_file(out_path: str, paragraphs: pd.DataFrame) -> None:
    paragraphs.to_csv(out_path, sep='\t', mode='a', header=not os.path.exists(out_path), index=False)


def apply_sub_perturbation(sentences, sub_sentences, sub_idx):
    perturbed_sentences = sentences.copy()
    perturbed_sentences[sub_idx] = sub_sentences[sub_idx]
    return perturbed_sentences


def apply_swap_perturbation(sentences, swap_idx_1, swap_idx_2):
    perturbed_sentences = sentences.copy()
    perturbed_sentences[swap_idx_1], perturbed_sentences[swap_idx_2] = perturbed_sentences[swap_idx_2], \
    perturbed_sentences[swap_idx_1]
    return perturbed_sentences


def apply_perturbations(paragraphs_df: pd.DataFrame, out_path: str, num_paragraphs: int) -> None:
    with tqdm(total=num_paragraphs-1, desc="Applying perturbations") as pbar:
        perturbed_paragraphs_df = pd.DataFrame(columns=['passage_id', 'text', 'label'])
        for index, row in paragraphs_df.iterrows():
            perturbed_paragraphs_df.loc[len(perturbed_paragraphs_df)] = [f'd{row.doc_id}_{row.passage_id}'] + [
                ' '.join(row.sentences)] + ['Orig']
            for sub_idx in range(4):
                perturbed_sentences = apply_sub_perturbation(row.sentences, row.sub_sentences, sub_idx)
                perturbed_paragraphs_df.loc[len(perturbed_paragraphs_df)] = [f'd{row.doc_id}_{row.passage_id}'] + [
                    ' '.join(perturbed_sentences)] + [f'sub_{sub_idx}']
            for swap_idx_1, swap_idx_2 in combinations(range(4), 2):
                perturbed_sentences = apply_swap_perturbation(row.sentences, swap_idx_1, swap_idx_2)
                perturbed_paragraphs_df.loc[len(perturbed_paragraphs_df)] = [f'd{row.doc_id}_{row.passage_id}'] + [
                    ' '.join(perturbed_sentences)] + [f'swap_{swap_idx_1}_{swap_idx_2}']

            pbar.update(1)
            if len(perturbed_paragraphs_df) >= 1000:  # ogni 1000 paragrafi scrivo sul file per non avere df troppo grandi
                write_to_out_file(out_path, perturbed_paragraphs_df)
                perturbed_paragraphs_df = pd.DataFrame(columns=['passage_id', 'text', 'label'])

    if not perturbed_paragraphs_df.empty:
        write_to_out_file(out_path, perturbed_paragraphs_df)


def count_paragraphs(src_path: str) -> int:
    with open(src_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language')
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    args = parser.parse_args()

    paragraphs_dir = f'../../data/src/train_eval_splits/{args.dataset}'
    out_dir = f'../../data/datasets/{args.dataset}/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for split in ['train', 'eval']:
        paragraphs_path = os.path.join(paragraphs_dir, f'{args.language}_{split}.tsv')
        out_path = os.path.join(out_dir, f'{args.language}_{split}.tsv')

        if os.path.exists(out_path):
            os.remove(out_path)

        print(f'Processing {split} set.')
        df = pd.read_csv(paragraphs_path, sep='\t', converters={'sentences': ast.literal_eval, 'sub_sentences':ast.literal_eval})
        num_paragraphs = count_paragraphs(paragraphs_path)
        apply_perturbations(df, out_path, num_paragraphs)


if __name__ == '__main__':
    main()
