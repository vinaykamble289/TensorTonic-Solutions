def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    pairs = sorted(zip(cal_probs, cal_labels))
    probabilities = [pair[0] for pair in pairs]
    blocks = []
    for _, label in pairs:
        blocks.append([float(label), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            blocks[-2][0] += blocks[-1][0]
            blocks[-2][1] += blocks[-1][1]
            blocks.pop()
    fitted = []
    for total, count in blocks:
        fitted.extend([total / count] * count)
    calibrated = []
    for probability in new_probs:
        if probability <= probabilities[0]:
            calibrated.append(fitted[0])
        elif probability >= probabilities[-1]:
            calibrated.append(fitted[-1])
        else:
            left = 0
            while probabilities[left + 1] < probability:
                left += 1
            ratio = (probability - probabilities[left]) / (probabilities[left + 1] - probabilities[left])
            calibrated.append(fitted[left] + ratio * (fitted[left + 1] - fitted[left]))
    return calibrated