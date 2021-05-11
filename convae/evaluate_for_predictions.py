"""
Evaluate representation model
"""
import torch
import numpy as np
import os
import csv
import model.net as net


def evaluate_for_predictions(model, loss_fn, data_iter_ts, metrics):
    model.eval()
    summ = []

    preds_file = os.path.join('data', 'encodings', 'test', 'predictions.csv')
    true_file = os.path.join('data', 'encodings', 'test', 'target.csv')

    with torch.no_grad():
        for idx, (list_mrn, list_batch) in enumerate(data_iter_ts):
            batch_idx = 0
            for batch, mrn in zip(list_batch, list_mrn):
                batch = batch.cuda()
                out, encoded = model(batch)
                loss = loss_fn(out, batch)
                out.cpu()
                encoded.cpu()

                print('\nAppending iter ', str(idx), ', batch ', str(batch_idx), ' predictions to file.')
                print(mrn)

                # write predictions as fixed-length subsequences
                pred = net.pred(out, encoded)
                with open(preds_file, 'a+') as f:
                    wr = csv.writer(f)
                    for predictions in pred.tolist():
                        wr.writerow([mrn] + predictions)

                # write original input as fixed-length subsequences
                with open(true_file, 'a+') as f:
                    wr = csv.writer(f)
                    for seqs in batch.tolist():
                        wr.writerow([mrn] + seqs)

                print('\nSummarizing batch...')
                summary_batch = {metric: metrics[metric](out, batch).item() for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

                batch_idx += 1

        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " -- ".join("{}: {:05.3f}".format(k.capitalize(), v)
                                     for k, v in sorted(metrics_mean.items()))
        print(metrics_string)

        return metrics_mean
