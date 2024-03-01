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


def load_model_results_monolingual(results_df, model_dir, model_name):
    # epochs = model_name.split('_')[1]
    model_name = model_name.split('_')[0]
    model_configs_dirs = [dir_name for dir_name in os.listdir(model_dir)]
    for model_config in model_configs_dirs:
        dataset = model_config.split('_')[1]
        results_dir = os.path.join(model_dir, model_config, 'predictions')
        try:
            model_result = compute_model_f1(results_dir)
        except:
            model_result = 0
        # new_row = pd.DataFrame.from_dict({'dataset': [epochs], 'model': [model_name], 'f1-score': [model_result]})
        new_row = pd.DataFrame.from_dict({'dataset': [dataset], 'model': [model_name], 'f1-score': [model_result]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


def load_model_results_multilingual(results_df, model_dir, config_name):
    language = config_name.split('_')[0]
    dataset = config_name.split('_')[1]
    results_dir = os.path.join(model_dir, 'predictions')
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
    parser.add_argument('-i', '--model_dir_name', type=str, )
    parser.add_argument('-o', '--out_path', type=str)
    parser.add_argument('-t', '--type', type=str)
    # parser.add_argument('-s', '--model_size', type=str, choices=['xsmall', 'small', 'base', 'large'])
    args = parser.parse_args()

    src_model_dir = os.path.join('../../models', args.model_dir_name)
    out_path = os.path.join('../../data/results', args.out_path)

    results_df = pd.DataFrame(columns=['dataset', 'model', 'language', 'f1-score'])
    for model_name in [model_name for model_name in os.listdir(src_model_dir) if
                       model_name not in ['old', 'out_domain']]:  ## if '-' + args.model_size in model_name]:
        model_dir = os.path.join(src_model_dir, model_name)
        # if 'base' in model_name:
        #     model_dir = os.path.join('/home/luca/Workspace/coherence/models/out_domain_3_epochs_5e-06_lr', model_name)
        if model_name == 'baseline':
            model_dir = os.path.join(model_dir, 'ngrams_1_2')
        if args.type == 'monolingual':
            results_df = load_model_results_monolingual(results_df, model_dir, model_name)
        elif args.type == 'multilingual':
            results_df = load_model_results_multilingual(results_df, model_dir, model_name)
    # hue_order = get_hue_order(results_df)

    if args.type == 'monolingual':
        ncol = 4
        hue_order = ['deberta-v3-xsmall', 'deberta-v3-small', 'deberta-v3-base', 'deberta-v3-large']
        bar_plot = sns.barplot(results_df, x='dataset', y='f1-score', hue='model', hue_order=hue_order)
    else:
        ncol = 5
        bar_plot = sns.barplot(results_df, x='dataset', y='f1-score', hue='language')
    sns.move_legend(bar_plot, 'lower center', bbox_to_anchor=(0.5, -0.4), ncol=ncol)
    fig = bar_plot.get_figure()
    fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
