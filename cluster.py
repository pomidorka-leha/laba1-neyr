from pathlib import Path  
import pandas as pd  
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler


file_path = Path(__file__).parent / 'dataset.csv'
df = pd.read_csv(file_path)

scaler = StandardScaler()  # создаем объект для стандартизации признаков
# fit вычисляет среднее и стандартное отклонение по каждому столбцу,
# transform приводит данные к виду со средним 0 и стандартным отклонением 1 (обычно от -3 до +3)
X_scaled = scaler.fit_transform(df)

# Создаем модель K-Means для разбиения данных на 4 кластера (оптимальное число по методу локтя)
# random_state=42 фиксирует случайность для повторяемости результатов
kmeans = KMeans(n_clusters=4, random_state=42)

# fit обучает модель (ищет центроиды), predict присваивает каждой строке номер кластера (от 0 до 3)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("Разбиение готово!")  # выводим сообщение об успешном завершении
print(df['cluster'].value_counts())  # выводим количество объектов в каждом кластере

plt.figure(figsize=(8, 6))  # создаем окно графика заданного размера (ширина, высота в дюймах)

# Рисуем точки на плоскости по признакам feature_1 (ось X) и feature_2 (ось Y),
# задаем цвет в зависимости от номера кластера и выбираем палитру viridis
plt.scatter(df['feature_1'], df['feature_2'], c=df['cluster'], cmap='viridis')

# Возвращаем координаты центров кластеров из масштабированного вида в исходные единицы измерения
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Отображаем центроиды на графике в виде крупных красных крестиков
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)

plt.xlabel('Feature 1')  # подписываем ось X
plt.ylabel('Feature 2')  # подписываем ось Y
plt.title('Разбиение на 4 кластера (с помощью scikit-learn)')  # заголовок графика
plt.show()  # отображаем готовый график