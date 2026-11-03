import numpy as np

def ranking_metrics_at_3(y_true, y_pred):
    data = list(zip(y_true, y_pred))
    data.sort(key=lambda x: x[1], reverse=True)
    top_3 = data[:3]
    tp_at_3 = sum(1 for target, pred in top_3 if target == 1)
    precision_at_3 = tp_at_3 / 3.0
    total_positives = sum(y_true)
    recall_at_3 = (tp_at_3 / total_positives) if total_positives > 0 else 0.0 
    return precision_at_3, recall_at_3



def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    if system_type == "classification":
        n = len(y_true)
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        TP = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
        TN = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
        FP = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
        FN = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0       
        return {
            "accuracy": (TP + TN) / n,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    elif system_type == "regression":
        n = len(y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return {
            "mae": np.sum(np.abs(y_true - y_pred)) / n,
            "rmse": np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
        }
    else:
        pr3 , rc3 = ranking_metrics_at_3(y_true, y_pred)
        return{
            "precision_at_3": pr3, 
            "recall_at_3": rc3 
        }