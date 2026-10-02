import numpy as np
import matplotlib.pyplot as plt

x = None
y = None

def m(x_i):
    return 2 + x_i

def l1(y, y_pred):
    return np.abs(y - y_pred)

def l2(y, y_pred):
    return np.square(y - y_pred)

with open('./problem_1_data/data.csv') as file:
    file.readline()
    data = [list(map(float, i.split(',')))[1:] for i in file.readlines()]
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
n = len(y)
y_pred = m(x)
a = l1(y, y_pred)
b = l2(y, y_pred)

e1 = (1 / n) * np.sum(a)
e2 = (1 / n) * np.sum(b)
print(f"E1 = {e1}, E2 = {e2}")

sorted_a = np.sort(a)
sorted_b = np.sort(b)

vk_a_1 = sorted_a[n // 2] / (n * e1)
vk_a_2 = sorted_a[n-1] / (n * e1)
vk_b_1 = sorted_b[n // 2] / (n * e2)
vk_b_2 = sorted_b[n-1] / (n * e2)
print(f"""Вклад L1 n/2 {vk_a_1},Вклад L1 n {vk_a_2},
Вклад L2 n/2 {vk_b_1},Вклад L2 n {vk_b_2}""")

median_index = np.where(a == sorted_a[n // 2])[0][0]
max_index = np.where(a == sorted_a[n - 1])[0][0]
print(median_index, max_index)

plt.figure(figsize=(10, 7))
plt.plot(x, y_pred, 'b-', linewidth=2, label=r'Модель $m(x) = 2 + x$')
plt.scatter(x, y, c='gray', alpha=0.6, s=50, label='аблюдения')
plt.scatter(x[max_index], y[max_index],
            c='green', marker='D', s=150, edgecolors='black', linewidths=1.5,
            label=f'Точка $a_{{(n/2)}}$ (ошибка={vk_a_1:.2f})')

plt.scatter(x[median_index], y[median_index],
            c='red', marker='*', s=300, edgecolors='black', linewidths=1.5,
            label=f'Точка $a_{{(n)}}$ (ошибка={vk_a_2:.2f})')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Линейная регрессия и выделенные точки по ошибке L1', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
