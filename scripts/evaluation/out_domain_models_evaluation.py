from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os

sns.set(style='darkgrid')


def load_labels_from_file(src_path):
    labels_list = []
    for line in open(src_path, 'r'):
        if line != '\n':
            labels_list.append(line.strip())
    return labels_list


def compute_model_f1(results_dir):
    y_true = load_labels_from_file(os.path.join(results_dir, 'y_true.txt'))
    y_pred = load_labels_from_file(os.path.join(results_dir, 'y_pred.txt'))
    return f1_score(y_true, y_pred, average='weighted')


def load_model_results(results_df, dataset_dir, dataset_name):
    language = dataset_name.split('_')[0]
    dataset = dataset_name.split('_')[1]
    results_dir = os.path.join(dataset_dir, 'predictions')
    try:
        model_result = compute_model_f1(results_dir)
    except:
        model_result = 0
    new_row = pd.DataFrame.from_dict({'dataset': [dataset], 'language': [language], 'f1-score': [model_result]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


def get_hue_order(results_df):
    model_names = results_df['model'].unique().tolist()
    learning_rates = [float(model_name.split('_')[-1][2:]) for model_name in model_names]
    sorted_model_names = [mn for lr, mn in sorted(zip(learning_rates, model_names))]
    return sorted_model_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_dir', type=str, default='../../models/multi_lingual_2e-05')
    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    results_df = pd.DataFrame(columns=['dataset', 'language', 'f1-score'])
    for dataset_name in [model_name for model_name in os.listdir(args.model_dir)]:
        dataset_dir = os.path.join(args.model_dir, dataset_name)
        results_df = load_model_results(results_df, dataset_dir, dataset_name)

    print(results_df)

    bar_plot = sns.barplot(results_df, x='dataset', y='f1-score', hue='language')
    sns.move_legend(bar_plot, 'lower center', bbox_to_anchor=(0.5, -0.4), ncol=5)
    fig = bar_plot.get_figure()
    fig.savefig(args.out_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
