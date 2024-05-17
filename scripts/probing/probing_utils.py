import torch
from torchsummary import summary
from torch.utils.data import Dataset
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from typing import Any, Dict, Optional, Tuple, Union, List


RELATION_PHRASES = {
    'top': "is on top of",
    'bottom': "is at the bottom of",
    'front': "is in front of",
    'behind': "is behind",
    'inside': "is inside",
    'outside': "is outside of",
}

def get_acc(output, gth):
    pred = output.argmax(1, keepdim=True)
    correct = pred.eq(gth.view_as(pred)).sum()
    acc = correct.float() / gth.shape[0]
    return acc

def get_lazy_acc(output, gth):
    # score 1 as long as it chose the correctly pair of opposite relations 
    # e.g. top-bottom, inside-outside, front-behind
    pred = output.argmax(1, keepdim=True)
    correct = (pred//2).eq((gth//2).view_as(pred)).sum()
    acc = correct.float() / gth.shape[0]
    return acc

def model_summary(model, batch_size, input_dim, device):
    dummy_input = torch.randn((batch_size, input_dim), device = device)
    summary(
        model, 
        [dummy_input],
        0,
        dtypes = [torch.FloatTensor],
        device = device,
        depth = 1
    )
    print("Trainable Params: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_confusion_matrix(labels, pred_labels):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm.plot(values_format='d', cmap='Blues', ax=ax)

class Texts(Dataset):
    def __init__(self,
                 relations: List[str],
                 nouns_file: str,
                 whoseroles: str="visual",
                 ):

        super().__init__()

        assert all([r in RELATION_PHRASES for r in relations]),\
            f"Unable to recognize relations. Please only provide relations in {str(list(RELATION_PHRASES.keys()))}"
        
        with open(nouns_file, "r") as f:
            nouns = [l.strip() for l in f.readlines()]
        self.texts = []
        
        for _A in nouns:
            for _B in nouns:
                if not _B == _A:
                    if _A[0] in 'aeiuo': A = "an "+_A
                    else: A = "a "+_A
                    if _B[0] in 'aeiuo': B = "an "+_B
                    else: B = "a "+_B
                    for label, r in enumerate(relations): 
                        if whoseroles=="visual":
                            self.texts.append((
                                f"{A} {RELATION_PHRASES[r]} {B}",
                                torch.tensor([label]),
                            ))
                        elif whoseroles=="linguistic":
                            self.texts.append((
                                f"{A} {RELATION_PHRASES[r]} {B}",
                                torch.tensor([0]),
                            ))
                        else: raise ValueError(f"Invalid role: {whoseroles}. Please provide one from ['visual', 'linguistic'].")
    
        #print(self.texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        return self.texts[i]
