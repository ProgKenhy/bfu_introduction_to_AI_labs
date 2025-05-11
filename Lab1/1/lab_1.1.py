import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Чтение данных из CSV файла
def read_data(filename):
    return pd.read_csv(filename)


# 2. Статистическая информация о данных
def show_stats(data, x_col, y_col):
    print(f"Статистика по столбцу X ({x_col}):")
    print(f"Количество: {data[x_col].count()}")
    print(f"Минимум: {data[x_col].min():}")
    print(f"Максимум: {data[x_col].max():}")
    print(f"Среднее: {data[x_col].mean():.2f}\n")

    print(f"Статистика по столбцу Y ({y_col}):")
    print(f"Количество: {data[y_col].count()}")
    print(f"Минимум: {data[y_col].min():}")
    print(f"Максимум: {data[y_col].max():}")
    print(f"Среднее: {data[y_col].mean():.2f}")


# 3. Визуализация исходных данных
def plot_initial_data(ax, x, y):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Исходные данные')
    ax.legend()
    ax.grid(True)


# 4. Метод наименьших квадратов
def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Вычисление коэффициентов
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    return b0, b1


# 5. Визуализация регрессионной прямой
def plot_regression_line(ax, x, y, b0, b1):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.plot(x, b0 + b1 * x, color='red', label='Регрессионная прямая')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Линейная регрессия')
    ax.legend()
    ax.grid(True)


# 6. Визуализация квадратов ошибок
def plot_error_squares(ax, x, y, b0, b1):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.plot(x, b0 + b1 * x, color='red', label='Регрессионная прямая')

    # Отрисовка квадратов ошибок
    for xi, yi in zip(x, y):
        y_pred = b0 + b1 * xi
        ax.plot([xi, xi], [yi, y_pred], color='green', linestyle='--', alpha=0.5)
        ax.add_patch(plt.Rectangle((xi, min(yi, y_pred)),
                                   abs(xi - xi),
                                   abs(yi - y_pred),
                                   alpha=0.1, color='green'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Квадраты ошибок')
    ax.legend()
    ax.grid(True)


# Основная функция
def main():
    # Загрузка данных
    filename = input("Введите имя файла (по умолчанию ../student_scores.csv): ") or "../student_scores.csv"
    data = read_data(filename)

    # Выбор столбцов
    print("Доступные столбцы:", list(data.columns))
    x_col = input("Выберите столбец для X (по умолчанию data.columns[0]): ") or data.columns[0]
    y_col = input("Выберите столбец для Y (по умолчанию data.columns[1]): ") or data.columns[1]

    # Получение данных
    x = data[x_col].values
    y = data[y_col].values

    # Статистика
    print("\n" + "=" * 50)
    show_stats(data, x_col, y_col)
    print("=" * 50 + "\n")

    # Вычисление коэффициентов
    b0, b1 = linear_regression(x, y)
    print(f"Уравнение регрессии: y = {b0:.2f} + {b1:.2f}x")

    # Создание графиков
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Исходные данные
    plot_initial_data(ax1, x, y)

    # 2. Регрессионная прямая
    plot_regression_line(ax2, x, y, b0, b1)

    # 3. Квадраты ошибок
    plot_error_squares(ax3, x, y, b0, b1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()