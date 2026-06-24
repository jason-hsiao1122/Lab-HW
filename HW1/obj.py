import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
from pathlib import Path

class two_hidden_layer:

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    relu = lambda x: np.maximum(0, x)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def loss_function(self, y_true, y_pred): # cross entropy
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    def accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))


    def __init__(
        self,
        input_size=784,
        h1_size=100,
        h2_size=150,
        output_size=10,
        activation='sigmoid',
        initialization='random',
    ):
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.output_size = output_size
        self.activation_name = activation
        self.initialization = initialization
        self.activation, self.activation_derivative = self.get_activation(activation)
        self.initialize_parameters(initialization)

    def initialize_parameters(self, initialization):
        if initialization == 'random':
            self.w1 = np.random.rand(self.h1_size, self.input_size) * 2 - 1
            self.b1 = np.random.rand(self.h1_size) * 2 - 1

            self.w2 = np.random.rand(self.h2_size, self.h1_size) * 2 - 1
            self.b2 = np.random.rand(self.h2_size) * 2 - 1

            self.w3 = np.random.rand(self.output_size, self.h2_size) * 2 - 1
            self.b3 = np.random.rand(self.output_size) * 2 - 1
            return

        if initialization == 'zero':
            self.w1 = np.zeros((self.h1_size, self.input_size))
            self.b1 = np.zeros(self.h1_size)

            self.w2 = np.zeros((self.h2_size, self.h1_size))
            self.b2 = np.zeros(self.h2_size)

            self.w3 = np.zeros((self.output_size, self.h2_size))
            self.b3 = np.zeros(self.output_size)
            return

        raise ValueError("initialization must be 'random' or 'zero'")

    def get_activation(self, activation):
        if activation == 'sigmoid':
            return two_hidden_layer.sigmoid, lambda a: a * (1 - a)
        if activation == 'relu':
            return two_hidden_layer.relu, lambda a: (a > 0).astype(float)
        raise ValueError("activation must be 'sigmoid' or 'relu'")

    def forword_passing(self, x):
            z1 = self.w1.dot(x) + self.b1[:, np.newaxis]
            a1 = self.activation(z1)
            z2 = self.w2.dot(a1) + self.b2[:, np.newaxis]
            a2 = self.activation(z2)
            z3 = self.w3.dot(a2) + self.b3[:, np.newaxis]
            a3 = two_hidden_layer.softmax(z3)
            return a1, a2, a3

    def save_parameters(self, file_path='outputs/model_parameters.npz'):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            file_path,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            b1=self.b1,
            b2=self.b2,
            b3=self.b3,
        )

    def load_parameters(self, file_path='outputs/model_parameters.npz'):
        data = np.load(file_path)
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.w3 = data['w3']
        self.b1 = data['b1']
        self.b2 = data['b2']
        self.b3 = data['b3']
        data.close()

        self.h1_size, self.input_size = self.w1.shape
        self.h2_size = self.w2.shape[0]
        self.output_size = self.w3.shape[0]

    def parameter_memory(self):
        return sum(
            param.nbytes
            for param in (self.w1, self.w2, self.w3, self.b1, self.b2, self.b3)
        )

    def format_memory(self, byte_count):
        units = ['B', 'KB', 'MB', 'GB']
        size = float(byte_count)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f'{size:.2f} {unit}'
            size /= 1024
    
    def plot_curves(
        self,
        train_loss,
        test_loss,
        train_acc,
        test_acc,
        output_dir='outputs',
        batch_size=None,
        peak_memory=None,
    ):
        Path(output_dir).mkdir(exist_ok=True)
        epochs = np.arange(1, len(train_loss) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        if batch_size is not None and peak_memory is not None:
            fig.suptitle(
                f'Batch size: {batch_size} | '
                f'Activation: {self.activation_name} | '
                f'Initialization: {self.initialization} | '
                f'Peak memory: {self.format_memory(peak_memory)} | '
                f'Parameter memory: {self.format_memory(self.parameter_memory())}'
            )

        axes[0].set(ylabel='Loss', xlabel='Epoch')
        axes[0].plot(epochs, train_loss, label='Train loss', color='r')
        axes[0].plot(epochs, test_loss, label='Test loss', color='g')
        axes[0].legend()

        axes[1].set(ylabel='Accuracy', xlabel='Epoch')
        axes[1].plot(epochs, train_acc, label='Train accuracy', color='r')
        axes[1].plot(epochs, test_acc, label='Test accuracy', color='g')
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(Path(output_dir) / 'learning_curves.png')
        plt.close(fig)
        
    def get_parameters(self):
        return (
            self.w1.copy(),
            self.w2.copy(),
            self.w3.copy(),
            self.b1.copy(),
            self.b2.copy(),
            self.b3.copy(),
        )

    def set_parameters(self, parameters):
        self.w1, self.w2, self.w3, self.b1, self.b2, self.b3 = [
            param.copy() for param in parameters
        ]

    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        batch_size,
        lr,
        output_dir='outputs',
        early_stopping=False,
        patience=5,
        min_delta=0.0,
    ):
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        sample_count = x_train.shape[1]
        best_loss = float('inf')
        best_epoch = 0
        best_parameters = None
        bad_epochs = 0
        tracemalloc.start()

        for epoch in range(epochs):
            indices = np.random.permutation(sample_count)
            # batch
            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start:start + batch_size]
                x = x_train[:, batch_indices]
                y = y_train[:, batch_indices]
                a1, a2, a3 = self.forword_passing(x)
                current_batch_size = x.shape[1]

                # bp
                error3 = a3 - y
                dw3 = error3.dot(a2.T) / current_batch_size

                error2 = self.activation_derivative(a2) * self.w3.T.dot(error3)
                dw2 = error2.dot(a1.T) / current_batch_size

                error1 = self.activation_derivative(a1) * self.w2.T.dot(error2)
                dw1 = error1.dot(x.T) / current_batch_size

                db3 = np.mean(error3, axis=1)
                db2 = np.mean(error2, axis=1)
                db1 = np.mean(error1, axis=1)

                self.w3 -= dw3 * lr
                self.w2 -= dw2 * lr
                self.w1 -= dw1 * lr

                self.b3 -= db3 * lr
                self.b2 -= db2 * lr
                self.b1 -= db1 * lr

            _, _, pred_train = self.forword_passing(x_train)
            _, _, pred_test = self.forword_passing(x_test)
            train_loss.append(self.loss_function(y_train, pred_train))
            test_loss.append(self.loss_function(y_test, pred_test))
            train_acc.append(self.accuracy(y_train, pred_train))
            test_acc.append(self.accuracy(y_test, pred_test))
            improved = test_loss[-1] < best_loss - min_delta
            if improved:
                best_loss = test_loss[-1]
                best_epoch = epoch + 1
                best_parameters = self.get_parameters()
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f'Epoch {epoch + 1}/{epochs} '
                f'- loss: {train_loss[-1]:.4f} '
                f'- acc: {train_acc[-1] * 100:.2f}% '
                f'- test_loss: {test_loss[-1]:.4f} '
                f'- test_acc: {test_acc[-1] * 100:.2f}%'
            )

            if early_stopping and bad_epochs >= patience:
                print(
                    f'Early stopping at epoch {epoch + 1}. '
                    f'Best test_loss: {best_loss:.4f} at epoch {best_epoch}.'
                )
                break

        if early_stopping and best_parameters is not None:
            self.set_parameters(best_parameters)
            print(f'Restored parameters from epoch {best_epoch}.')
        
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.plot_curves(
            train_loss,
            test_loss,
            train_acc,
            test_acc,
            output_dir,
            batch_size,
            peak_memory,
        )
        return {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'peak_memory': peak_memory,
            'parameter_memory': self.parameter_memory(),
            'epochs_ran': len(train_loss),
            'best_epoch': best_epoch,
            'best_test_loss': best_loss,
        }
            
                

    
    def test(self, x_test, y_test, raw_images=None, output_dir='outputs'):
        predict = []
        _, _, preds = self.forword_passing(x_test)
        for pred, true in zip(preds.T, y_test):
            predict.append(np.argmax(pred) == true)
        accuracy = np.sum(predict) / len(predict)
        print(f'Accuracy = {accuracy*100:.02f}%')

        if raw_images is None:
            return

        Path(output_dir).mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 5, figsize=(15, 10))
        axes = axes.flatten()
        rng = np.random.default_rng()
        for i in range(10):
            index = rng.integers(0, raw_images.shape[0])
            axes[i].imshow(raw_images[index], cmap='Greys')
            output = preds[:, index]
            predict_num = np.argmax(output)
            axes[i].set(
                title=f'Predict: {predict_num}, True: {y_test[index]} \nconfidence: {output[predict_num] * 100:.02f} %'
            )
            axes[i].axis('off')
        fig.tight_layout()
        fig.savefig(Path(output_dir) / 'sample_predictions.png')
        plt.show()
            
            

        


    
