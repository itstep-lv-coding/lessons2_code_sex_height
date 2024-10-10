class CustomArray:
    def __init__(self, data):
        self.data = list(data)

    # Метод для отримання значень за індексами або умовами
    def __getitem__(self, index):
        if isinstance(index, list):  # Перевірка чи це список (умова фільтрації)
            return [self.data[i] for i, cond in enumerate(index) if cond]
        else:  # Інакше це індекс
            return self.data[index]

    # Метод для присвоєння значень за індексами або умовами
    def __setitem__(self, index, value):
        if isinstance(index, list):  # Якщо index - це список з умовою
            value_iter = iter(value)  # Ітератор по новим значенням
            for i, cond in enumerate(index):
                if cond:  # Присвоюємо тільки тим, хто відповідає умовам
                    self.data[i] = next(value_iter)
        else:
            self.data[index] = value

# Створюємо екземпляр класу з масивом
arr = CustomArray([1, 2, 3, 4, 5])

# Фільтрація елементів, більших за 3
condition = [x > 3 for x in arr.data]
print(arr[condition])  # Виведе [4, 5]

# Присвоєння нових значень елементам, більшим за 3
arr[condition] = [0, 0]

print(arr.data)  # Виведе [1, 2, 3, 0, 0]