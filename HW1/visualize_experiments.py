import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_LABELS = {
    'final_train_loss': 'Final train loss',
    'final_test_loss': 'Final test loss',
    'final_train_accuracy': 'Final train accuracy',
    'final_test_accuracy': 'Final test accuracy',
    'saved_model_test_accuracy': 'Saved model test accuracy',
    'best_test_loss': 'Best test loss',
    'train_time_seconds': 'Train time (seconds)',
    'peak_memory_bytes': 'Peak memory (bytes)',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize experiment result files without running training or testing.'
    )
    parser.add_argument(
        '--input-dir',
        default='experiment_outputs',
        help='Directory containing summary.csv and per-run learning_history.csv files.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for visualization files. Defaults to <input-dir>/visualizations.',
    )
    return parser.parse_args()


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path


def read_summary(input_dir):
    summary_path = input_dir / 'summary.csv'
    if summary_path.exists():
        with summary_path.open(newline='') as file:
            return list(csv.DictReader(file))

    summary_json_path = input_dir / 'summary.json'
    if summary_json_path.exists():
        with summary_json_path.open() as file:
            return json.load(file)

    raise FileNotFoundError(
        f'Cannot find summary.csv or summary.json in {input_dir}. '
        'Run run_experiments.py first, then visualize the saved result files.'
    )


def read_history(run_dir):
    history_path = run_dir / 'learning_history.csv'
    with history_path.open(newline='') as file:
        rows = list(csv.DictReader(file))

    return {
        'epoch': [int(row['epoch']) for row in rows],
        'train_loss': [float(row['train_loss']) for row in rows],
        'test_loss': [float(row['test_loss']) for row in rows],
        'train_acc': [float(row['train_acc']) for row in rows],
        'test_acc': [float(row['test_acc']) for row in rows],
    }


def to_number(value):
    if isinstance(value, (int, float)):
        return value
    return float(value)


def run_label(row):
    return (
        f"{row['activation']}/"
        f"{row['initialization']}/"
        f"batch {row['batch_size']}"
    )


def plot_summary_bar(rows, metric, output_path):
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row['activation'],
            row['initialization'],
            int(row['batch_size']),
        ),
    )
    labels = [run_label(row) for row in sorted_rows]
    values = [to_number(row[metric]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(13, 6), dpi=120)
    ax.bar(range(len(values)), values, color='#4c78a8')
    ax.set_title(METRIC_LABELS[metric])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grouped_accuracy(rows, output_path):
    groups = {}
    for row in rows:
        key = (row['activation'], row['initialization'])
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for (activation, initialization), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda row: int(row['batch_size']))
        x_values = [int(row['batch_size']) for row in group_rows]
        y_values = [to_number(row['final_test_accuracy']) for row in group_rows]
        ax.plot(
            x_values,
            y_values,
            marker='o',
            linewidth=2,
            label=f'{activation}/{initialization}',
        )

    ax.set_title('Final test accuracy by batch size')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Final test accuracy')
    ax.set_xscale('log')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_learning_curves(rows, input_dir, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    for row in rows:
        run_dir = resolve_path(row['output_dir'], input_dir)
        history_path = run_dir / 'learning_history.csv'
        if not history_path.exists():
            continue

        history = read_history(run_dir)
        label = run_label(row)
        axes[0].plot(history['epoch'], history['test_loss'], label=label)
        axes[1].plot(history['epoch'], history['test_acc'], label=label)

    axes[0].set_title('Test loss learning curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test loss')
    axes[0].grid(alpha=0.25)

    axes[1].set_title('Test accuracy learning curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test accuracy')
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    hw1_dir = Path(__file__).resolve().parent
    input_dir = resolve_path(args.input_dir, hw1_dir)
    output_dir = (
        resolve_path(args.output_dir, hw1_dir)
        if args.output_dir
        else input_dir / 'visualizations'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(input_dir)
    if not rows:
        raise ValueError(f'No experiment rows found in {input_dir}.')

    plot_summary_bar(rows, 'final_test_accuracy', output_dir / 'final_test_accuracy.png')
    plot_summary_bar(rows, 'final_test_loss', output_dir / 'final_test_loss.png')
    plot_summary_bar(rows, 'train_time_seconds', output_dir / 'train_time_seconds.png')
    plot_summary_bar(rows, 'peak_memory_bytes', output_dir / 'peak_memory_bytes.png')
    plot_grouped_accuracy(rows, output_dir / 'accuracy_by_batch_size.png')
    plot_all_learning_curves(rows, input_dir, output_dir / 'all_test_learning_curves.png')

    print(f'Visualization files saved to: {output_dir}')


if __name__ == '__main__':
    main()
