def MEDIAN (data: list) -> float:
    if not data:
        return None #если список пустой
    sorted_values = sorted(data)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:  # нечетное количество элементов
        return float(sorted_values[mid])
    else:  # четное количество элементов
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2

nums = []
print(MEDIAN (nums))  
