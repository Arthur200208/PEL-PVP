import pandas as pd
import torch
from collections import Counter

def saveCSV(val_labels, val_pre):
    pos_result, neg_result = [0]*len(val_labels), [0]*len(val_labels)
    p, n = 0, 0
    for i in range(len(val_labels)):
        if val_labels[i] == 1:
            pos_result[p] = (val_pre[i])
            p = p + 1
        if val_labels[i] == 0:
            neg_result[n] = (val_pre[i])
            n = n + 1
    pos = pd.DataFrame(pos_result, columns=['val_pre'])
    neg = pd.DataFrame(neg_result, columns=['val_pre'])
    pos.to_csv('result/cross_pos.csv', index=False)
    neg.to_csv('result/cross_neg.csv', index=False)
    print(pos.head())

def getPaparametersNum(net):
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

def checkLength(sequences):
    length_ranges = [(0, 500), (500, 1000), (2000, 10000)]
    length_counter = Counter()
    for sequence in sequences:
        length = len(sequence)
        for length_range in length_ranges:
            if length_range[0] <= length <= length_range[1]:
                length_counter[length_range] += 1
    for length_range, count in length_counter.items():
        print(f"Length Range {length_range}: {count} sequences")


def checkModelParams(net):
    for layer_idx, layer in enumerate(net.layers):
        print(f"Layer {layer_idx + 1} Parameters:")
        for name, param in layer.named_parameters():
            print(f"{name}: {param.size()}")
        print("=" * 50)

def printFeatureShape(output):
    for key, value in output['representations'].items():
        print(f'Layer {key} output shape: {value.shape}')


def printOutputShape(output):
    print(f'output shape: {output["logits"].shape}')


class Focal_Loss():
    """
    Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):

        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)

    def __call__(self, preds, labels):
        return self.forward(preds, labels)