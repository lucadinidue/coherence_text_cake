from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import csv
import os

sns.set(style='darkgrid')


def compute_f1_score(src_path):
    y_pred, y_true = [], []
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file, delimiter='\t')
        for row in csv_reader:
            y_pred.append(row[0])
            y_true.append(row[1])
    return f1_score(y_true, y_pred, average='weighted')


def laod_models_scores(results_df, model_name, src_dir):
    for layer in os.listdir(src_dir):
        layer_dir = os.path.join(src_dir, layer)
        for dataset_file_name in os.listdir(layer_dir):
            dataset_file_path = os.path.join(layer_dir, dataset_file_name)
            dataset = dataset_file_name.split('.')[0].split('_')[1]
            f1_score = compute_f1_score(dataset_file_path)
            new_row = pd.DataFrame.from_dict(
                {'dataset': [dataset], 'model': [model_name], 'f1-score': [f1_score], 'layer': [int(layer)+1]})
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, choices=['pretrained', 'finetuned'])
    args = parser.parse_args()
    
    src_dir = f'/home/luca/Workspace/coherence/models/probing_predictions/{args.type}'
    out_path = f'/home/luca/Workspace/coherence/data/results/probing_{args.type}.png'

    results_df = pd.DataFrame(columns=['dataset', 'model', 'f1-score', 'layer'])
    for model_name in os.listdir(src_dir):
        model_dir = os.path.join(src_dir, model_name)
        results_df = laod_models_scores(results_df, model_name, model_dir)

    plt.ylim(0.05, 0.55)
    plt.xlim(0.5, 24.5)
    hue_order = ['deberta-v3-xsmall', 'deberta-v3-small', 'deberta-v3-base', 'deberta-v3-large']
    bar_plot = sns.lineplot(results_df, x='layer', y='f1-score', hue='model', hue_order=hue_order)#, style='dataset')
    sns.move_legend(bar_plot, 'lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    fig = bar_plot.get_figure()
    fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
