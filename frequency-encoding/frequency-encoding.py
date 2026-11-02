def frequency_encoding(values: list) -> list:
    map = dict()
    for i in values:
        map[i]= map.get(i,0) + 1
        
    total_elements = len(values)
    frequencies = {k: v / total_elements for k, v in map.items()}
    return [frequencies[val] for val in values]