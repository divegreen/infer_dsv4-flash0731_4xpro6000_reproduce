def moving_average(values, width):
    if width <= 0:
        raise ValueError("width must be positive")
    return [sum(values[i:i + width]) / width for i in range(len(values) - width)]
