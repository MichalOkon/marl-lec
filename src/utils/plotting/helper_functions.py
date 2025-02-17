import numpy as np

def get_ci_bootstrap(data, confidence=0.95):
    B = 10000
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(B)]

    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(boot_means, 100 * alpha)
    upper_bound = np.percentile(boot_means, 100 * (1 - alpha))

    interval = upper_bound - lower_bound

    return interval / 2


def calculate_y_lim(data, columns):
    y_max = max(data[column].max() for column in columns if column in data.columns)
    y_min = min(data[column].min() for column in columns if column in data.columns)

    if y_max == y_min:
        if y_max == 0:
            return 0, 1
        else:
            return 0, y_max * 1.1

    if y_min < 0:
        y_min = y_min * 1.1
    else:
        y_min = y_min * 0.9
    y_max = y_max * 1.1

    return y_max, y_min
