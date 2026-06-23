import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class two_hidden_layer:

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def loss_function(self, y_true, y_pred): # cross entropy
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    def accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))


    def __init__(self, input_size=784, h1_size=100, h2_size=150, output_size=10):
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.output_size = output_size

        self.w1 = np.random.rand(h1_size, input_size) * 2 - 1
        self.b1 = np.random.rand(h1_size) * 2 - 1

        self.w2 = np.random.rand(h2_size, h1_size) * 2 - 1
        self.b2 = np.random.rand(h2_size) * 2 - 1
        
        self.w3 = np.random.rand(output_size, h2_size) * 2 - 1
        self.b3 = np.random.rand(output_size) * 2 - 1

    def forword_passing(self, x):
            z1 = self.w1.dot(x) + self.b1[:, np.newaxis]
            a1 = two_hidden_layer.sigmoid(z1)
            z2 = self.w2.dot(a1) + self.b2[:, np.newaxis]
            a2 = two_hidden_layer.sigmoid(z2)
            z3 = self.w3.dot(a2) + self.b3[:, np.newaxis]
            a3 = two_hidden_layer.softmax(z3)
            return a1, a2, a3
    
    def plot_curves(self, train_loss, test_loss, train_acc, test_acc, output_dir='outputs'):
        Path(output_dir).mkdir(exist_ok=True)
        epochs = np.arange(1, len(train_loss) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
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
        plt.show()
        
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, lr):
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        sample_count = x_train.shape[1]

        for epoch in range(epochs):
            indices = np.random.permutation(sample_count)
            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start:start + batch_size]
                x = x_train[:, batch_indices]
                y = y_train[:, batch_indices]
                a1, a2, a3 = self.forword_passing(x)
                current_batch_size = x.shape[1]

                # bp
                error3 = a3 - y
                dw3 = error3.dot(a2.T) / current_batch_size

                error2 = a2 * (1 - a2) * self.w3.T.dot(error3)
                dw2 = error2.dot(a1.T) / current_batch_size

                error1 = a1 * (1 - a1) * self.w2.T.dot(error2)
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
            print(f'Epoch {epoch + 1}/{epochs} - loss: {train_loss[-1]:.4f} - acc: {train_acc[-1] * 100:.2f}% - test_acc: {test_acc[-1] * 100:.2f}%')
        
        self.plot_curves(train_loss, test_loss, train_acc, test_acc)
            
                

    
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
            
            

        


    
