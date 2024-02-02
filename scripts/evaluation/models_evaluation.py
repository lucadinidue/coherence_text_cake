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


def load_model_results(results_df, model_dir, model_name):
    model_configs_dirs = [dir_name for dir_name in os.listdir(model_dir)]
    for model_config in model_configs_dirs:
        dataset = model_config.split('_')[1]
        results_dir = os.path.join(model_dir, model_config, 'predictions')
        try:
            model_result = compute_model_f1(results_dir)
        except:
            model_result = 0
        new_row = pd.DataFrame.from_dict({'dataset': [dataset], 'model': [model_name], 'score': [model_result]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


def get_hue_order(results_df):
    model_names = results_df['model'].unique().tolist()
    learning_rates = [float(model_name.split('_')[-1][2:]) for model_name in model_names]
    sorted_model_names = [mn for lr, mn in sorted(zip(learning_rates, model_names))]
    return sorted_model_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--model_size', type=str, choices=['xsmall', 'small', 'base', 'large'])
    args = parser.parse_args()
    models_dir = '../../models'
    results_df = pd.DataFrame(columns=['dataset', 'model', 'score'])
    for model_name in [model_name for model_name in os.listdir(models_dir) if '-' + args.model_size in model_name]:
        model_dir = os.path.join(models_dir, model_name)
        if model_name == 'baseline':
            model_dir = os.path.join(model_dir, 'ngrams_1_2')
        results_df = load_model_results(results_df, model_dir, model_name)
    #hue_order =get_hue_order(results_df)
    # hue_order = ['deberta-v3-xsmall', 'deberta-v3-small', 'deberta-v3-base', 'deberta-v3-large', 'baseline']
    bar_plot = sns.barplot(results_df, x='dataset', y='score', hue='model')#, hue_order=hue_order)
    sns.move_legend(bar_plot, 'lower center', bbox_to_anchor=(1, 1))
    fig = bar_plot.get_figure()
    fig.savefig(f'../../data/results/{args.model_size}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
