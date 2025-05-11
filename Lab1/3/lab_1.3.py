import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# 2. Выбор признака и подготовка данных
selected_feature = 'bmi'
X = df[[selected_feature]].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Реализация Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)


# 4. Собственная реализация
def custom_linreg(X, y):
    X = X.ravel()
    X_mean, y_mean = np.mean(X), np.mean(y)
    b1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    b0 = y_mean - b1 * X_mean
    return b0, b1


custom_b0, custom_b1 = custom_linreg(X_train, y_train)
custom_pred = custom_b0 + custom_b1 * X_test.ravel()


# 5. Функция для вычисления MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 6. Вычисление метрик для обеих моделей
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\nМетрики для модели {model_name}:")
    print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
    print(f"R² (Коэффициент детерминации): {r2:.2f}")
    print(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")

    return mae, r2, mape


# 7. Вывод метрик
sklearn_metrics = calculate_metrics(y_test, sklearn_pred, "Scikit-Learn")
custom_metrics = calculate_metrics(y_test, custom_pred, "Собственная реализация")

# 8. Сравнительная таблица
metrics_df = pd.DataFrame({
    'Метрика': ['MAE', 'R²', 'MAPE (%)'],
    'Scikit-Learn': [f"{sklearn_metrics[0]:.2f}", f"{sklearn_metrics[1]:.2f}", f"{sklearn_metrics[2]:.2f}"],
    'Собственная': [f"{custom_metrics[0]:.2f}", f"{custom_metrics[1]:.2f}", f"{custom_metrics[2]:.2f}"]
})

print("\nСравнение метрик:")
print(metrics_df.to_markdown(index=False))

# 9. Вывод о качестве моделей
print("\nВывод:")
if sklearn_metrics[1] > 0.5:
    print("Модель Scikit-Learn показывает хорошее качество (R² > 0.5)")
else:
    print("Модель Scikit-Learn показывает умеренное качество")

if np.allclose(sklearn_metrics, custom_metrics, atol=1e-2):
    print("Обе модели дают практически идентичные результаты")
else:
    print("Результаты моделей существенно отличаются")