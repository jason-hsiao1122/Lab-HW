import numpy as np
import os
import argparse
from obj import two_hidden_layer


def flatten_images(images):
    return images.reshape(images.shape[0], -1).T / 255.0


def one_hot(labels, class_count=10):
    return np.eye(class_count)[labels].T


def parse_args():
    parser = argparse.ArgumentParser(description="Train a two hidden layer neural network on MNIST.")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Run training or only test a saved model.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop training when test loss does not improve.",
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
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "relu"],
        default="sigmoid",
        help="Activation function for hidden layers.",
    )
    parser.add_argument(
        "--init",
        choices=["random", "zero"],
        default="random",
        help="Parameter initialization method.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files.")
    parser.add_argument(
        "--parameter-file",
        default="model_parameters.npz",
        help="Filename for saved model parameters.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(path, output_dir)

    data = np.load(os.path.join(path,"mnist.npz"))

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    data.close()

    x_train_flat = flatten_images(x_train)
    x_test_flat = flatten_images(x_test)

    model = two_hidden_layer(activation=args.activation, initialization=args.init)
    parameter_path = os.path.join(output_dir, args.parameter_file)

    if args.mode == "train":
        y_train_one_hot = one_hot(y_train)
        y_test_one_hot = one_hot(y_test)
        model.train(
            x_train_flat,
            y_train_one_hot,
            x_test_flat,
            y_test_one_hot,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=output_dir,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        model.save_parameters(parameter_path)
    else:
        if not os.path.exists(parameter_path):
            raise FileNotFoundError(
                f"Saved parameter file not found: {parameter_path}. "
                "Run with --mode train first, or set --parameter-file."
            )
        model.load_parameters(parameter_path)

    model.test(x_test_flat, y_test, raw_images=x_test, output_dir=output_dir)
