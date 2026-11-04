def choose_threshold(y_true: list, scores: list, min_precision: float) -> dict:
    total_positives = sum(1 for label in y_true if label == 1)
    best = {"threshold": None, "precision": None, "recall": None}
    for candidate in sorted(set(scores)):
        true_positives = 0
        predicted_positives = 0
        for label, score in zip(y_true, scores):
            if score >= candidate:
                predicted_positives += 1
                if label == 1:
                    true_positives += 1
        precision = true_positives / predicted_positives
        recall = true_positives / total_positives
        if precision < min_precision:
            continue
        if best["recall"] is None or recall > best["recall"] or (recall == best["recall"] and precision > best["precision"]):
            best = {"threshold": float(candidate), "precision": precision, "recall": recall}
    return best