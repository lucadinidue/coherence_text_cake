from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
import os

sns.set(style='darkgrid')

lr_map = {
    'deberta-v3-xsmall': 4.5e-5,
    'deberta-v3-small': 4.5e-5,
    'deberta-v3-base': 2e-5
}


def load_labels_from_file(src_path):
    labels_list = []
    for line in open(src_path, 'r'):
        if line != '\n':
            labels_list.append(int(line.strip()))
    return labels_list


def compute_model_f1(results_dir):
    y_true = load_labels_from_file(os.path.join(results_dir, 'y_true.txt'))
    y_pred = load_labels_from_file(os.path.join(results_dir, 'y_pred.txt'))
    return f1_score(y_true, y_pred, average='weighted')


def load_model_results(results_df, model_dir, model_name):
    model_configs_dirs = [dir_name for dir_name in os.listdir(model_dir)]  # if dir_name != 'prove']
    for model_config in model_configs_dirs:
        dataset = model_config.split('_')[1]
        results_dir = os.path.join(model_dir, model_config, 'predictions')
        model_result = compute_model_f1(results_dir)
        new_row = pd.DataFrame.from_dict({'dataset': [dataset], 'model': [model_name], 'score': [model_result]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


def main():
    models_dir = '../../models'
    results_df = pd.DataFrame(columns=['dataset', 'model', 'score'])
    for model_name in [model_name for model_name in os.listdir(models_dir) if model_name !='deberta-v3-large']:
        model_dir = os.path.join(models_dir, model_name, f'{lr_map[model_name]}')
        results_df = load_model_results(results_df, model_dir, model_name)
    bar_plot = sns.barplot(results_df, x='dataset', y='score', hue='model')
    sns.move_legend(bar_plot, 'lower center', bbox_to_anchor=(1, 1))
    fig = bar_plot.get_figure()
    fig.savefig('../../data/results/model_performances_lr_huggingface.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
