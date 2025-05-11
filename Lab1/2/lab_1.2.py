import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# 1. Загрузка и анализ данных
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("Анализ корреляции признаков с целевой переменной:")
corr_matrix = df.corr()
print(corr_matrix['target'].sort_values(ascending=False))

# 2. Выбор признака
selected_feature = 'bmi'
print(f"\nВыбран признак '{selected_feature}' как имеющий наибольшую корреляцию с целевой переменной")

# 3. Подготовка данных (убедимся, что X всегда 2D-массив)
X = df[[selected_feature]].values  # Обратите внимание на двойные скобки [[ ]]
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Реализация Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)

# 5. Собственная реализация
def custom_linreg(X, y):
    X = X.ravel()  # Преобразуем в 1D для вычислений
    X_mean, y_mean = np.mean(X), np.mean(y)
    b1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
    b0 = y_mean - b1 * X_mean
    return b0, b1

custom_b0, custom_b1 = custom_linreg(X_train, y_train)
custom_pred = custom_b0 + custom_b1 * X_test.ravel()

# 6. Сравнение моделей
results = pd.DataFrame({
    'Метод': ['Scikit-Learn', 'Собственная'],
    'Intercept': [sklearn_model.intercept_, custom_b0],
    'Коэффициент': [sklearn_model.coef_[0], custom_b1],
    'R²': [r2_score(y_test, sklearn_pred), r2_score(y_test, custom_pred)],
    'MSE': [mean_squared_error(y_test, sklearn_pred), mean_squared_error(y_test, custom_pred)]
})

print("\nСравнение моделей:")
print(tabulate(results, headers='keys', tablefmt='psql', showindex=False))

# 7. Визуализация с исправлением ошибки размерности
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Обучающая выборка')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Тестовая выборка')

# Исправление: преобразуем x_vals в 2D-массив
x_vals = np.array([X.min(), X.max()]).reshape(-1, 1)
plt.plot(x_vals, sklearn_model.predict(x_vals),
         color='red', linewidth=2, label='Scikit-Learn')
plt.plot(x_vals.ravel(), custom_b0 + custom_b1*x_vals.ravel(),
         '--', color='orange', linewidth=2, label='Собственная реализация')

plt.xlabel('BMI (стандартизированный)')
plt.ylabel('Прогрессирование заболевания')
plt.title('Сравнение линейных регрессий для признака BMI')
plt.legend()
plt.grid(True)
plt.show()

# 8. Таблица предсказаний
pred_table = pd.DataFrame({
    'BMI (X)': X_test.ravel(),
    'Факт (Y)': y_test,
    'Scikit-Learn': sklearn_pred,
    'Собственная': custom_pred,
    'Разница': np.abs(sklearn_pred - custom_pred)
}).head(15)

print("\nДетальные предсказания для первых 15 тестовых образцов:")
print(tabulate(pred_table, headers='keys', tablefmt='psql', floatfmt=".4f"))