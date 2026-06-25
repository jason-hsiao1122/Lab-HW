import argparse
import csv
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np

from main import flatten_images, one_hot
from obj import two_hidden_layer


ACTIVATIONS = ['sigmoid', 'relu']
INITIALIZATIONS = ['random', 'zero']
BATCH_SIZES = [16, 256, 1024, 60000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train all activation, initialization, and batch-size combinations."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs for each model.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--output-dir",
        default="experiment_outputs",
        help="Root directory for experiment output files.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop each model when test loss does not improve.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Epochs to wait without test loss improvement before early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum test loss improvement required to reset early stopping patience.",
    )
    return parser.parse_args()


def write_history(history, output_path):
    with output_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        for epoch, values in enumerate(
            zip(
                history['train_loss'],
                history['test_loss'],
                history['train_acc'],
                history['test_acc'],
            ),
            start=1,
        ):
            writer.writerow([epoch, *values])


def save_metrics(metrics, output_path):
    with output_path.open('w') as file:
        json.dump(metrics, file, indent=2)


def main():
    args = parse_args()
    hw1_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = hw1_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    data = np.load(hw1_dir / "mnist.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    data.close()

    x_train_flat = flatten_images(x_train)
    x_test_flat = flatten_images(x_test)
    y_train_one_hot = one_hot(y_train)
    y_test_one_hot = one_hot(y_test)

    summary_rows = []
    total_models = len(ACTIVATIONS) * len(INITIALIZATIONS) * len(BATCH_SIZES)

    for model_index, (activation, initialization, batch_size) in enumerate(
        product(ACTIVATIONS, INITIALIZATIONS, BATCH_SIZES),
        start=1,
    ):
        run_name = f'{activation}_{initialization}_batch{batch_size}'
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f'[{model_index}/{total_models}] Training {run_name}')
        model = two_hidden_layer(
            activation=activation,
            initialization=initialization,
        )

        start_time = time.perf_counter()
        history = model.train(
            x_train_flat,
            y_train_one_hot,
            x_test_flat,
            y_test_one_hot,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            output_dir=run_dir,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        train_time = time.perf_counter() - start_time

        model.save_parameters(run_dir / 'model_parameters.npz')
        _, _, saved_test_pred = model.forword_passing(x_test_flat)
        test_loss = model.loss_function(y_test_one_hot, saved_test_pred)
        test_accuracy = model.accuracy(y_test_one_hot, saved_test_pred)

        metrics = {
            'activation': activation,
            'initialization': initialization,
            'batch_size': batch_size,
            'epochs_requested': args.epochs,
            'epochs_ran': history['epochs_ran'],
            'learning_rate': args.lr,
            'early_stopping': args.early_stopping,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'best_epoch': history['best_epoch'],
            'best_test_loss': history['best_test_loss'],
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'final_train_accuracy': history['train_acc'][-1],
            'final_test_accuracy': history['test_acc'][-1],
            'saved_model_test_loss': test_loss,
            'saved_model_test_accuracy': test_accuracy,
            'peak_memory_bytes': history['peak_memory'],
            'peak_memory': model.format_memory(history['peak_memory']),
            'parameter_memory_bytes': history['parameter_memory'],
            'parameter_memory': model.format_memory(history['parameter_memory']),
            'train_time_seconds': train_time,
            'output_dir': str(run_dir),
            'learning_curve_file': str(run_dir / 'learning_curves.png'),
            'parameter_file': str(run_dir / 'model_parameters.npz'),
        }

        write_history(history, run_dir / 'learning_history.csv')
        save_metrics(metrics, run_dir / 'metrics.json')
        summary_rows.append(metrics)

        print(
            f'Finished {run_name}: '
            f'test_acc={test_accuracy * 100:.2f}% '
            f'time={train_time:.2f}s '
            f'peak_memory={metrics["peak_memory"]}'
        )

    summary_path = output_root / 'summary.csv'
    with summary_path.open('w', newline='') as file:
        fieldnames = [
            'activation',
            'initialization',
            'batch_size',
            'epochs_requested',
            'epochs_ran',
            'learning_rate',
            'early_stopping',
            'patience',
            'min_delta',
            'best_epoch',
            'best_test_loss',
            'final_train_loss',
            'final_test_loss',
            'final_train_accuracy',
            'final_test_accuracy',
            'saved_model_test_loss',
            'saved_model_test_accuracy',
            'peak_memory_bytes',
            'peak_memory',
            'parameter_memory_bytes',
            'parameter_memory',
            'train_time_seconds',
            'output_dir',
            'learning_curve_file',
            'parameter_file',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    save_metrics(summary_rows, output_root / 'summary.json')
    print(f'All experiment outputs saved to: {output_root}')


if __name__ == '__main__':
    main()
