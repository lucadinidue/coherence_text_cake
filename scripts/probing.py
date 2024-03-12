from extract_sentence_representations import extract_sentence_representations, get_pretrained_model_name
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pandas as pd
import argparse
import torch
import csv
import os


def load_dataset(src_dir, layer):
    dataset_tensor = None
    layer_dir = os.path.join(src_dir, str(layer))
    for slice_idx in range(len(os.listdir(layer_dir))):
        slice_file_path = os.path.join(layer_dir, f'{slice_idx}.pt')
        slice_tensor = torch.load(slice_file_path, map_location=torch.device('cpu'))
        dataset_tensor = slice_tensor if dataset_tensor is None else torch.cat((dataset_tensor, slice_tensor), dim=1)
    return dataset_tensor


def get_labels(src_path):
    labels_df = pd.read_csv(src_path, sep='\t', usecols=['label'])
    return labels_df['label'].to_list()


def preprare_datasets(representations_paths, dataset_paths, layer):
    X_train = load_dataset(representations_paths['train'], layer)
    X_eval = load_dataset(representations_paths['eval'], layer)

    if X_train.shape[0] != 88000 or X_eval.shape[0] != 22000:
        raise Exception(f'Error on dataset shape: Train {X_train.shape[0]}, eval {X_eval.shape[0]}')

    y_train = get_labels(dataset_paths['train'])
    y_eval = get_labels(dataset_paths['eval'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    return X_train, y_train, X_eval, y_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-l', '--language', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-y', '--layer', type=int)
    parser.add_argument('-p', '--probing_model', type=str, choices=['mlp', 'svc'], default='mlp')
    parser.add_argument('-s', '--slice_tensor', action='store_true')
    parser.add_argument('-o', '--out_dir', type=str, default='pretriained')
    args = parser.parse_args()

    pt_model_name = get_pretrained_model_name(args.model_name)
    model_str = pt_model_name.split('/')[1]
    representations_dir = f'../data/probing_datasets_{args.out_dir}/{args.dataset}/{args.language}/{model_str}'
    dataset_dir = f'../data/datasets/{args.dataset}'
    out_dir = f'../models/probing_predictions/{args.out_dir}/{model_str}/{args.layer}/'

    out_path = os.path.join(out_dir, f'{args.language}_{args.dataset}.tsv')
    if os.path.exists(out_path):
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(representations_dir):
        extract_sentence_representations(args.model_name, args.language, args.dataset, representations_dir,
                                         batch_size=args.batch_size, slice=args.slice_tensor)

    representations_paths = {
        'train': os.path.join(representations_dir, 'train'),
        'eval': os.path.join(representations_dir, 'eval')
    }

    dataset_paths = {
        'train': os.path.join(dataset_dir, f'{args.language}_train.tsv'),
        'eval': os.path.join(dataset_dir, f'{args.language}_eval.tsv')
    }

    X_train, y_train, X_eval, y_eval = preprare_datasets(representations_paths, dataset_paths, args.layer)

    if args.probing_model == 'mlp':
        clf = MLPClassifier(random_state=1, max_iter=2000, verbose=True)
    else:
        clf = LinearSVC(dual='auto', verbose=1)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_eval)

    with open(out_path, 'w+') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for y_pred, y_true in zip(predictions, y_eval):
            csv_writer.writerow([y_pred, y_true])


if __name__ == '__main__':
    main()
