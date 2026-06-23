import numpy as np
import os
from obj import two_hidden_layer


def flatten_images(images):
    return images.reshape(images.shape[0], -1).T / 255.0


def one_hot(labels, class_count=10):
    return np.eye(class_count)[labels].T


if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))

    data = np.load(os.path.join(path,"mnist.npz"))

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    data.close()

    x_train_flat = flatten_images(x_train)
    x_test_flat = flatten_images(x_test)
    y_train_one_hot = one_hot(y_train)
    y_test_one_hot = one_hot(y_test)

    model = two_hidden_layer()
    model.train(
        x_train_flat,
        y_train_one_hot,
        x_test_flat,
        y_test_one_hot,
        epochs=2,
        batch_size=16,
        lr=0.01,
    )
    model.test(x_test_flat, y_test, raw_images=x_test)
