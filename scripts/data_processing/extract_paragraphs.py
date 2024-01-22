from tqdm import tqdm
import pandas as pd
import argparse
import csv
import os


def extract_doc_passages(doc_sentences: pd.DataFrame, passages_dim: int, skip_sentences: int) -> pd.DataFrame:
    grouped_paragraphs = doc_sentences.groupby('par_id')
    passages = {'passage_id': [], 'sentences': [], 'sub_sentences': []}

    for par_id, par_df in grouped_paragraphs:
        passage_ids, passage_sentences, sub_sentences = [], [], []
        num_paragraph_sentences = par_df.shape[0]
        for row_idx, (row_id, row) in enumerate(par_df[['sent_id', 'sentence']].iterrows()):
            if row_idx + skip_sentences < num_paragraph_sentences:
                passage_ids.append(str(row['sent_id']))
                passage_sentences.append(row['sentence'])
                sub_sentences.append(par_df.iloc[row_idx + skip_sentences]['sentence'])
                if len(passage_sentences) == passages_dim:
                    passages['passage_id'].append(f'p{par_id}_' + '_'.join(passage_ids))
                    passages['sentences'].append(passage_sentences)
                    passages['sub_sentences'].append(sub_sentences)
                    passage_ids, passage_sentences, sub_sentences = [], [], []
    return pd.DataFrame.from_dict(passages)


def create_passages(documents_df: pd.DataFrame, passages_dim: int, skip_sentences: int, out_path: str) -> None:
    all_paragraphs = []
    doc_ids = documents_df['doc_id'].unique().tolist()
    for doc_id in tqdm(doc_ids):
        doc_sentences = documents_df[documents_df['doc_id'] == doc_id]
        doc_paragraphs = extract_doc_passages(doc_sentences, passages_dim, skip_sentences)
        doc_paragraphs.insert(0, 'doc_id', doc_id)
        all_paragraphs.append(doc_paragraphs)
        if len(all_paragraphs) >= 100:
            write_passages_to_file(out_path, pd.concat(all_paragraphs, ignore_index=True))
            all_paragraphs = []
    if all_paragraphs:
        write_passages_to_file(out_path, pd.concat(all_paragraphs, ignore_index=True))


def load_documents(src_path: str, last_doc_id: str) -> pd.DataFrame:
    documents_df = pd.read_csv(src_path, delimiter='\t')
    if last_doc_id is not None:
        try:
            start_idx = documents_df[documents_df['doc_id'] == last_doc_id].index[-1] + 1
        except:  # some datasets have integer document ids
            start_idx = documents_df[documents_df['doc_id'] == int(last_doc_id)].index[-1] + 1
        documents_df = documents_df.loc[start_idx:]
    return documents_df


def write_passages_to_file(out_path: str, paragraphs: pd.DataFrame) -> None:
    paragraphs.to_csv(out_path, sep='\t', mode='a', header=not os.path.exists(out_path), index=False)


def get_last_doc_id(out_path: str) -> str:
    with open(out_path, 'r') as out_file:
        csv_reader = csv.reader(out_file, delimiter='\t')
        for row in csv_reader:
            pass
        last_id = row[0]
    return last_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language')
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-r', '--restart', action='store_true')
    parser.add_argument('-n', '--num_sentences', type=int, default=4)
    parser.add_argument('-s', '--skip_sentences', type=int, default=10)
    args = parser.parse_args()

    src_path = f'data/processed_src/{args.dataset}/{args.language}.tsv'
    out_path = f'data/paragraphs/{args.dataset}/{args.language}.tsv'

    last_doc_id = None

    if os.path.exists(out_path):
        if args.restart:
            os.remove(out_path)
        else:
            last_doc_id = get_last_doc_id(out_path)

    print('Loading documents.')
    documents = load_documents(src_path, last_doc_id)
    print('(Done)\n')
    print('Creating paragraphs.')
    create_passages(documents, args.num_sentences, args.skip_sentences, out_path)
    print('(Done)')


if __name__ == '__main__':
    main()
