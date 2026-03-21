def median(arr):
    n = len(arr)
    mid = n // 2
    if n % 2 == 0:
        return (arr[mid - 1] + arr[mid]) / 2
    return arr[mid]

def robust_scaling(values):
    values_sorted = sorted(values)
    n = len(values_sorted)
    if n==1:
        return [0]
    # Median
    med = median(values_sorted)

    # Split into halves
    if n % 2 == 0:
        lower = values_sorted[:n//2]
        upper = values_sorted[n//2:]
    else:
        lower = values_sorted[:n//2]
        upper = values_sorted[n//2 + 1:]

    # Quartiles
    Q1 = median(lower)
    Q3 = median(upper)

    IQR = Q3 - Q1

    # Scaling
    scaled = []
    for x in values:
        if IQR == 0:
            scaled.append(x - med)
        else:
            scaled.append((x - med) / IQR)

    return scaled