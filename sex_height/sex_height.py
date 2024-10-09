import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Створимо дані вручну
# data = {
#     'height': np.random.normal(66, 4, 1000),  # Випадкові дані про зріст
#     'sex': np.random.choice(['Male', 'Female'], size=1000)  # Випадкові категорії за статтю
# }
# heights = pd.DataFrame(data)

heights = pd.read_csv('data/heights.csv')

# Перевіряємо структуру даних
print(heights.head())
print('тип даних',type(heights))

# Визначаємо результат (y) і предиктори (x)
y = heights['sex']
x = heights['height']

# Розраховуємо кількість вимірювань для кожної статі
male_counts = heights[heights['sex'] == 'Male']['height'].value_counts().sort_index()
female_counts = heights[heights['sex'] == 'Female']['height'].value_counts().sort_index()

# Об'єднуємо індекси (висоти) для чоловіків і жінок
combined_index = male_counts.index.union(female_counts.index)

# Заповнюємо відсутні значення нулями
male_counts = male_counts.reindex(combined_index, fill_value=0)
female_counts = female_counts.reindex(combined_index, fill_value=0)

# Ширина стовпців
bar_width = 0.4

# Створюємо графік
plt.bar(combined_index - bar_width / 2, male_counts.values, width=bar_width, label='Male', color='blue')
plt.bar(combined_index + bar_width / 2, female_counts.values, width=bar_width, label='Female', color='red')

# Налаштовуємо підписи
plt.xlabel('Height')
plt.ylabel('Count')
plt.title('Height Distribution by Sex')

# Додаємо легенду
plt.legend()

# Показуємо графік
plt.grid(True)
plt.show()

# Створюємо навчальні і тестові вибірки
np.random.seed(2)
train_set, test_set, y_train, y_test = train_test_split(heights, y, test_size=0.5, stratify=y)

# Робимо випадкове передбачення результату
y_hat = np.random.choice(["Male", "Female"], size=len(test_set), replace=True)

# Обчислюємо точність випадкових передбачень
accuracy_random = np.mean(y_hat == y_test)
print(f"Точність випадкового передбачення: {accuracy_random:.4f}")

# Аналізуємо середнє і стандартне відхилення по групам
group_stats = heights.groupby('sex').agg(mean_height=('height', 'mean'), sd_height=('height', 'std'))
print("\nСтатистика за групами:")
print(group_stats)

# Створюємо передбачення на основі порогу 62
y_hat = np.where(x > 62, "Male", "Female")
accuracy_62 = np.mean(y == y_hat)
print(f"\nТочність при порозі 62: {accuracy_62:.4f}")

# Досліджуємо точність для різних порогів
cutoff = np.arange(61, 71)
accuracy = []

for c in cutoff:
    y_hat_train = np.where(train_set['height'] > c, "Male", "Female")
    accuracy.append(np.mean(y_hat_train == train_set['sex']))

# Візуалізація
plt.plot(cutoff, accuracy, marker='o', linestyle='-')
plt.xlabel('Поріг')
plt.ylabel('Точність')
plt.title('Точність для різних порогів')
plt.grid(True)
plt.show()

# Знаходимо найкращий поріг
best_cutoff = cutoff[np.argmax(accuracy)]
print(f"Найкращий поріг: {best_cutoff}")

# Використовуємо найкращий поріг для тестової вибірки
y_hat_test = np.where(test_set['height'] > best_cutoff, "Male", "Female")
accuracy_best = np.mean(y_hat_test == test_set['sex'])
print(f"Точність з найкращим порогом: {accuracy_best:.4f}")

# Додаємо Confusion Matrix (матрицю плутанини)
cm = confusion_matrix(y_test, y_hat_test)
print("\nМатриця плутанини:")
print(cm)

# Візуалізуємо матрицю плутанини
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"]).plot()
plt.show()

# Перетворюємо матрицю плутанини на відсотки
# cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Перетворюємо матрицю на тип float, щоб працювати з дробовими значеннями
cm_float = cm.astype('float')

# Обчислюємо загальну кількість справжніх зразків для кожного класу (суми по рядках)
row_sums = cm_float.sum(axis=1)

# Ділимо кожен елемент матриці на відповідну суму рядка, щоб отримати частки
cm_percent = cm_float / row_sums[:, np.newaxis]

# Перетворюємо частки у відсотки
cm_percent = cm_percent * 100

# Виводимо результат
print("Матриця плутанини у відсотках:")
print(cm_percent)

# Виводимо матрицю з відсотковими значеннями
print("\nМатриця плутанини (у відсотках):")
print(cm_percent)

# Візуалізуємо матрицю плутанини
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["Female", "Male"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (відсотки)")
plt.show()
