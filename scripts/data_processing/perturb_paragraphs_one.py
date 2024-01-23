from typing import Iterator
from classes import *
from tqdm import tqdm
import pandas as pd
import argparse
import random
import ast
import os


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def load_paragraph_iterator(paragraphs_path: str) -> Iterator:
    paragraphs_iterator = pd.read_csv(paragraphs_path, sep='\t', converters={'sentences': safe_literal_eval},
                                      # ast.literal_eval},
                                      chunksize=1000, on_bad_lines='skip')
    return paragraphs_iterator


def load_sentences_dict(sentences_path: str) -> dict:
    sentences_df = pd.read_csv(sentences_path, sep='\t')
    sentences_df['key'] = 'd' + sentences_df['doc_id'].astype(str) + '_p' + sentences_df['par_id'].astype(str) + '_s' + \
                          sentences_df['sent_id'].astype(str)
    sentences_dict = sentences_df.set_index('key')['sentence'].to_dict()
    del sentences_df
    return sentences_dict


def write_to_out_file(output_path: str, paragraphs: pd.DataFrame) -> None:
    paragraphs.to_csv(output_path, sep='\t', mode='a', header=not os.path.exists(output_path), index=False)


def apply_perturbations(paragraph_iterator: Iterator, perturbations: dict, output_path: str,
                        num_paragraphs: int) -> None:
    dataset_dim = 0
    with tqdm(total=num_paragraphs, desc="Processing Rows") as pbar:
        perturbed_paragraphs_df = pd.DataFrame(
            columns=['passage_id', 'sentence_0', 'sentence_1', 'sentence_2', 'sentence_3', 'perturbation'])

        for chunk in paragraph_iterator:
            for paragraph in chunk.itertuples():
                print(paragraph)
                assert False
                perturbation_type = random.choice(['swap', 'sub', None, None])
                perturber = perturbations[perturbation_type]
                perturbed_paragraph = perturber.perturb_paragraph(paragraph)
                perturbation_str = perturber.get_perturbation_str()
                perturbed_paragraphs_df.loc[len(perturbed_paragraphs_df)] = [
                                                                                f'd{paragraph.doc_id}_{paragraph.passage_id}'] + perturbed_paragraph + [
                                                                                perturbation_str]
                dataset_dim += 1
                pbar.update(1)
                if len(perturbed_paragraphs_df) == 1000:  # ogni 1000 paragrafi scrivo sul file per non avere df troppo grandi
                    write_to_out_file(output_path, perturbed_paragraphs_df)
                    perturbed_paragraphs_df = pd.DataFrame(
                        columns=['passage_id', 'sentence_0', 'sentence_1', 'sentence_2', 'sentence_3', 'perturbation'])
                if dataset_dim == num_paragraphs:  # se ho raggiunto il numero richiesto di paragrafi mi fermo
                    if not perturbed_paragraphs_df.empty:
                        write_to_out_file(output_path, perturbed_paragraphs_df)
                    return
        if not perturbed_paragraphs_df.empty:
            write_to_out_file(output_path, perturbed_paragraphs_df)


def count_paragraphs(src_path: str) -> int:
    with open(src_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language')
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-s', '--skip_sentences', type=int, default=10)
    parser.add_argument('-n', '--num_paragraphs', type=int, default=10000)
    args = parser.parse_args()

    random.seed(42)

    paragraphs_path = f'data/paragraphs/{args.dataset}/{args.language}.tsv'
    sentences_path = f'data/processed_src/{args.dataset}/{args.language}.tsv'
    output_dir = f'data/out/all_perturbations/{args.dataset}/'
    output_path = os.path.join(output_dir, f'{args.language}.tsv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        os.remove(output_path)

    print('Loading paragraphs...')
    paragraph_iterator = load_paragraph_iterator(paragraphs_path)
    sentences_dict = load_sentences_dict(sentences_path)
    print('(Done)\n')

    swap = SwapPerturbation()
    sub = SubPerturbation(sentences_dict, args.skip_sentences)
    no = NoPerturbation()

    perturbations = {
        'swap': swap,
        'sub': sub,
        None: no
    }

    num_paragraphs = count_paragraphs(paragraphs_path)
    apply_perturbations(paragraph_iterator, perturbations, output_path, min(num_paragraphs, args.num_paragraphs))
    print('(Done)')


if __name__ == '__main__':
    main()
