import numpy as np

# Пример данных пользователей (3 признака на каждого)
users = {
    "user1": [0.2, 0.5, 0.1],
    "user2": [0.1, 0.4, 0.3],
    "user3": [0.8, 0.7, 0.6],
    "user4": [0.25, 0.5, 0.15]
}

# Преобразуем в матрицу numpy
vectors = np.array(list(users.values())).astype('float32')



import faiss

# Размерность векторов
d = vectors.shape[1]

# Создаём индекс L2 (евклидово расстояние)
index = faiss.IndexFlatL2(d)

# Загружаем векторы в индекс
index.add(vectors)

# Вектор нового пользователя (например, user_new)
new_user_vector = np.array([[0.21, 0.52, 0.12]], dtype='float32')

# Ищем 2 ближайших соседа
k = 2
distances, indices = index.search(new_user_vector, k)

print("Индексы ближайших соседей:", indices)
print("Расстояния до них:", distances)

# Получаем имена пользователей по индексам
user_list = list(users.keys())
nearest_users = [user_list[i] for i in indices[0]]

print("Похожие пользователи:", nearest_users)
