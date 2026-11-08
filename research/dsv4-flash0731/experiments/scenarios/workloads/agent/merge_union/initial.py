def merge_counts(left, right):
    return {key: left.get(key, 0) + right.get(key, 0) for key in left}
