"""Statistical and analytical functions for data series processing."""


def compute_moving_average(values, window_size):
    """Compute simple moving average with given window size.

    Args:
        values: List of numbers
        window_size: Size of the moving window

    Returns:
        List of moving averages (shorter than input by window_size - 1).
    """
    if not values or window_size <= 0:
        return []
    if window_size > len(values):
        return []

    averages = []
    for i in range(len(values) - window_size + 1):
        window_sum = 0
        count = 0
        for j in range(i, i + window_size):
            if j < len(values):
                window_sum = window_sum + values[j]
                count = count + 1
        if count > 0:
            avg = window_sum / count
        else:
            avg = 0
        averages.append(avg)

    return averages
