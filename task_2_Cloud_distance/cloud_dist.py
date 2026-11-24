# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. ЧТЕНИЕ ИСХОДНЫХ ДАННЫХ
# ==========================================
# Столбцы: X Y Z ID
ref = np.loadtxt('opornaya_setka.txt', skiprows=1)
cmp_ = np.loadtxt('sravnivaemaya_setka.txt', skiprows=1)

xyz_ref = ref[:, :3]
xyz_cmp = cmp_[:, :3]
id_cmp = cmp_[:, 3].astype(int)

# ==========================================
# 2. РАСЧЕТ РАССТОЯНИЙ (NumPy Broadcasting)
# ==========================================
# diff[i, j] = вектор (сравниваемая точка i) - (опорная точка j)
diff = xyz_cmp[:, None, :] - xyz_ref[None, :, :]  # (400, 400, 3)
dist2 = np.sum(diff ** 2, axis=2)  # квадраты расстояний (400, 400)
nearest_idx = np.argmin(dist2, axis=1)  # индекс ближайшей опорной точки
d = np.sqrt(dist2[np.arange(len(xyz_cmp)), nearest_idx])

# ==========================================
# 3. ПРОВЕРКА РАСЧЕТА В PYTHON
# ==========================================
expected = 0.10
max_err = np.max(np.abs(d - expected))
print(f"--- Проверка Python ---")
print(f"Число точек: {len(d)}")
print(f"Минимум: {d.min():.6f} м, Максимум: {d.max():.6f} м")
print(f"Макс. отклонение от 0.10 м: {max_err:.2e} м")

# Проверка перестановки строк опорного массива (доказывает независимость от порядка)
rng = np.random.default_rng(0)
perm = rng.permutation(len(xyz_ref))
d_perm = np.sqrt(np.min(np.sum((xyz_cmp[:, None, :] - xyz_ref[perm][None, :, :]) ** 2, axis=2), axis=1))
print(f"Разность после перемешивания опорника: {np.max(np.abs(d - d_perm)):.2e} м\n")

# ==========================================
# 4. СРАВНЕНИЕ С CLOUDCOMPARE (resultat_CloudCompare.txt)
# ==========================================
print("--- Сравнение с CloudCompare ---")
try:
    # Читаем файл от CloudCompare
    cc_data = np.loadtxt('resultat_CloudCompare.txt', skiprows=1)

    # Умный поиск столбцов (на случай, если CloudCompare добавил цвета R G B)
    num_cols = cc_data.shape[1]
    if num_cols == 5:
        # Формат: X Y Z ID distance
        id_cc = cc_data[:, 3].astype(int)
        d_cc = cc_data[:, 4]
    elif num_cols == 8:
        # Формат: X Y Z R G B ID distance
        id_cc = cc_data[:, 6].astype(int)
        d_cc = cc_data[:, 7]
    else:
        # Если формат другой (например, без заголовка), пробуем взять предпоследний и последний
        print(f"Внимание: неизвестное число столбцов ({num_cols}). Пробуем взять последние два.")
        id_cc = cc_data[:, -2].astype(int)
        d_cc = cc_data[:, -1]

    # Сортируем оба массива по ID для корректного сопоставления
    sort_py = np.argsort(id_cmp)
    sort_cc = np.argsort(id_cc)

    id_py_sorted = id_cmp[sort_py]
    d_py_sorted = d[sort_py]

    id_cc_sorted = id_cc[sort_cc]
    d_cc_sorted = d_cc[sort_cc]

    # Проверяем совпадение множеств ID
    if np.array_equal(id_py_sorted, id_cc_sorted):
        diff_cc_py = np.max(np.abs(d_py_sorted - d_cc_sorted))
        print(f"ID совпадают! (Всего {len(id_cc_sorted)} точек)")
        print(f"Макс. расхождение Python vs CloudCompare: {diff_cc_py:.2e} м")
        if diff_cc_py <= 1e-5:
            print("РЕЗУЛЬТАТ: УСПЕХ! Обе среды дали одинаковый результат.")
        else:
            print("РЕЗУЛЬТАТ: ОШИБКА! Расхождение больше допуска 1e-5.")
    else:
        print("ОШИБКА: Множества ID в файлах не совпадают!")

except Exception as e:
    print(f"Не удалось прочитать файл CloudCompare: {e}")
    print("Убедитесь, что файл resultat_CloudCompare.txt лежит в той же папке.")

# ==========================================
# 5. СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ==========================================
# Сохраняем результат Python в текстовый файл
out = np.column_stack([xyz_cmp, id_cmp, d])
np.savetxt('resultat_python.txt', out, fmt='%.6f %.6f %.6f %d %.6f',
           header='X Y Z ID distance', comments='')

# 5.1. 3D Визуализация (два облака и поле)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), subplot_kw={'projection': '3d'})

ax = axes[0]
ax.scatter(xyz_ref[:, 0], xyz_ref[:, 1], xyz_ref[:, 2], c='tab:blue', s=8, depthshade=False, label='опорное (Z=0)')
ax.scatter(xyz_cmp[:, 0], xyz_cmp[:, 1], xyz_cmp[:, 2], c='tab:orange', s=8, depthshade=False,
           label='сравниваемое (Z=0.10)')
ax.set_title('Исходные облака')
ax.legend(loc='upper left', fontsize=8)

ax = axes[1]
sc = ax.scatter(xyz_cmp[:, 0], xyz_cmp[:, 1], xyz_cmp[:, 2], c=d, cmap='viridis', vmin=0.0, vmax=0.20, s=8,
                depthshade=False)
ax.set_title('Поле расстояний (единая шкала 0–0.20 м)')
fig.colorbar(sc, ax=ax, shrink=0.6, label='расстояние, м')

for a in axes:
    a.set_xlabel('X, м');
    a.set_ylabel('Y, м');
    a.set_zlabel('Z, м')

plt.tight_layout()
plt.savefig('vizualizacija_zadanie9.png', dpi=150)

# 5.2. Гистограмма (для отчета, так как в CloudCompare она может ломаться)
plt.figure(figsize=(8, 4))
plt.hist(d, bins=20, color='skyblue', edgecolor='black')
plt.title('Гистограмма расстояний (Python)')
plt.xlabel('Расстояние, м')
plt.ylabel('Количество точек')
plt.xlim(0.05, 0.15)  # Расширяем границы, чтобы столбик было видно
plt.grid(axis='y', alpha=0.75)
plt.savefig('histogram_python.png', dpi=150)

print("\nСохранено: resultat_python.txt, vizualizacija_zadanie9.png, histogram_python.png")
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
