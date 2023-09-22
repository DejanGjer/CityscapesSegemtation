import torch
import torch.nn as nn
import numpy as np

def compute_metrics(metric, num_labels):
    with torch.no_grad():
        metrics = metric.compute(
            num_labels=num_labels,
            ignore_index=0,
            reduce_labels=False
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics
    
def add_batch_to_metrics(metric, logits, labels):
    with torch.no_grad():
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metric.add_batch(
            predictions=pred_labels,
            references=labels
        )
    return pred_labels