import numpy as np
import matplotlib.pyplot as plt
import random

class LinearPerceptron:
    """Простой линейный классификатор, обучаемый градиентным методом"""
    def __init__(self, lr=0.1, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.randn(2)
        self.b = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        losses = []
        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)
            error = y_pred - y
            loss = np.mean((error) ** 2)
            losses.append(loss)

            # Градиенты
            grad_w = np.dot(X.T, error * y_pred * (1 - y_pred)) / len(X)
            grad_b = np.mean(error * y_pred * (1 - y_pred))

            # Обновляем веса
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return losses

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.w) + self.b) >= 0.5).astype(int)

    def line_points(self, x_vals):
        # уравнение границы: w1*x1 + w2*x2 + b = 0 → x2 = -(w1*x1 + b)/w2
        return -(self.w[0] * x_vals + self.b) / self.w[1]


class XORNeuralNetworkDemo:
    def __init__(self):
        self.X = None
        self.y = None

    def generate_xor_data(self, num_points_per_class=50):
        np.random.seed(42)
        centers = [
            [0, 0],  # Класс 0
            [0, 1],  # Класс 1
            [1, 0],  # Класс 1
            [1, 1]   # Класс 0
        ]
        X, y = [], []
        for center in centers:
            for _ in range(num_points_per_class):
                x1 = center[0] + random.uniform(-0.2, 0.2)
                x2 = center[1] + random.uniform(-0.2, 0.2)
                X.append([x1, x2])
                y.append(center[0] ^ center[1])
        self.X = np.array(X)
        self.y = np.array(y)
        return self.X, self.y

    def draw_all_stages_with_training(self):
        X = self.X
        y = self.y

        # Разделим данные для "OR" и "NAND"
        y_or = np.where((X[:, 0] + X[:, 1]) >= 0.5, 1, 0)
        y_nand = np.where((X[:, 0] + X[:, 1]) <= 1.5, 1, 0)

        # Обучаем 2 персептрона
        or_model = LinearPerceptron(lr=0.5, epochs=3000)
        nand_model = LinearPerceptron(lr=0.5, epochs=3000)
        or_model.fit(X, y_or)
        nand_model.fit(X, y_nand)

        # Линии после обучения
        x_line = np.linspace(-0.5, 1.5, 100)
        y_or_line = or_model.line_points(x_line)
        y_nand_line = nand_model.line_points(x_line)

        # --- Рисуем 4 этапа ---
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        titles = [
            "1. Исходные данные XOR",
            "2. Обученный нейрон OR",
            "3. Обученный нейрон NAND",
            "4. Комбинация XOR (через OR и NAND)"
        ]

        # 1️⃣ Исходные данные
        axes[0].scatter(X[:, 0], X[:, 1],
                        c=['lightcoral' if yi == 1 else 'lightblue' for yi in y],
                        edgecolor='k', s=70)
        axes[0].set_xlim(-0.5, 1.5)
        axes[0].set_ylim(-0.5, 1.5)
        axes[0].set_title(titles[0])
        axes[0].grid(True)

        # 2️⃣ OR
        axes[1].scatter(X[:, 0], X[:, 1],
                        c=['lightcoral' if yi == 1 else 'lightblue' for yi in y_or],
                        edgecolor='k', s=70)
        axes[1].plot(x_line, y_or_line, 'k--',
                     label=f"y₂ = {-(or_model.w[0]/or_model.w[1]):.2f}x + {-(or_model.b/or_model.w[1]):.2f}")
        axes[1].legend()
        axes[1].set_xlim(-0.5, 1.5)
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].set_title(titles[1])
        axes[1].grid(True)

        # 3️⃣ NAND
        axes[2].scatter(X[:, 0], X[:, 1],
                        c=['lightcoral' if yi == 1 else 'lightblue' for yi in y_nand],
                        edgecolor='k', s=70)
        axes[2].plot(x_line, y_nand_line, 'g--',
                     label=f"y₂ = {-(nand_model.w[0]/nand_model.w[1]):.2f}x + {-(nand_model.b/nand_model.w[1]):.2f}")
        axes[2].legend()
        axes[2].set_xlim(-0.5, 1.5)
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].set_title(titles[2])
        axes[2].grid(True)

        # 4️⃣ XOR = AND(OR, NAND)
        y_or_pred = or_model.predict(X)
        y_nand_pred = nand_model.predict(X)
        y_xor = y_or_pred & y_nand_pred

        axes[3].scatter(X[:, 0], X[:, 1],
                        c=['lightcoral' if yi == 1 else 'lightblue' for yi in y_xor],
                        edgecolor='k', s=70)
        axes[3].plot(x_line, y_or_line, 'k--', label='OR')
        axes[3].plot(x_line, y_nand_line, 'g--', label='NAND')
        axes[3].fill_between(x_line, y_or_line, y_nand_line, color='yellow', alpha=0.2, label='XOR область')
        axes[3].legend()
        axes[3].set_xlim(-0.5, 1.5)
        axes[3].set_ylim(-0.5, 1.5)
        axes[3].set_title(titles[3])
        axes[3].grid(True)

        fig.suptitle("Обучение OR/NAND персептронов градиентным методом для задачи XOR", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    demo = XORNeuralNetworkDemo()
    demo.generate_xor_data(50)
    demo.draw_all_stages_with_training()


if __name__ == "__main__":
    main()
