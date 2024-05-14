
from torch.utils.data import Dataset
import torch
from typing import Any, Dict, Optional, Tuple, Union, List

RELATIONS = ["{0} is on top of {1}.", "{1} is at the bottom of {0}."]


def create_data_split2(nouns):
    train_texts, test_texts = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i < j:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
            elif i > j:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for label, r in enumerate(RELATIONS): 
                    test_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
    return train_texts, test_texts


def create_data_split3(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 1 <= i - j < half or -len(nouns)+1 <= i - j <= -half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
            else:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                for label, r in enumerate(RELATIONS): 
                    test_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
    return train_texts, test_texts


def create_data_split4(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if i < half and j < half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                train_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                )) 
                test_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                ))   
            elif i >= half and j >= half:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                test_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                )) 
                train_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                ))   
            else:
                if _A[0] in 'aeiuo': A = "an "+_A
                else: A = "a "+_A
                if _B[0] in 'aeiuo': B = "an "+_B
                else: B = "a "+_B
                test_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                )) 
                test_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                )) 
    return train_texts, test_texts


def create_data_split6(nouns):
    train_texts, test_texts = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i == j: continue
            
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if i+j < len(nouns)-1:
                train_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                )) 
                test_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                ))  
            elif i+j > len(nouns)-1:
                train_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                )) 
                test_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                ))  
    return train_texts, test_texts


def create_data_split5(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 <= i < half and half <= j < len(nouns):
                train_label = 0
            elif  0 <= j < half and half <= i < len(nouns):
                train_label = 1
            else:
                test_texts.append((
                    RELATIONS[0].format(A, B),
                    torch.tensor([0]),
                )) 
                test_texts.append((
                    RELATIONS[1].format(A, B),
                    torch.tensor([1]),
                )) 
                continue

            test_label = 1-train_label  
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                torch.tensor([train_label]),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                torch.tensor([test_label]),
            ))
            
    return train_texts, test_texts


def create_data_split7(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            train_label = int(i>j)
            
            test_label = 1-train_label  
            train_texts.append((
                RELATIONS[train_label].format(A, B),
                torch.tensor([train_label]),
            ))
            test_texts.append((
                RELATIONS[test_label].format(A, B),
                torch.tensor([test_label]),
            ))
            
    return train_texts, test_texts


def create_data_split9(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            if 0 <= i < half and half <= j < len(nouns): 
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    )) 
            elif half <= i < len(nouns) and 0 <= j < half: 
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    )) 
            else:
                for label, r in enumerate(RELATIONS): 
                    test_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    )) 
            
    return train_texts, test_texts

def create_data_split12(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B
            if 0 <= i < half and half <= j < len(nouns): 
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    )) 
            else:
                for label, r in enumerate(RELATIONS): 
                    test_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
    return train_texts, test_texts


def create_data_split13(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B
            if half <= j < len(nouns): 
                for label, r in enumerate(RELATIONS): 
                    train_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    )) 
            else:
                for label, r in enumerate(RELATIONS): 
                    test_texts.append((
                        r.format(A, B),
                        torch.tensor([label]),
                    ))   
    return train_texts, test_texts


def create_data_impossible_split(nouns):
    train_texts, test_texts = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if _A[0] in 'aeiuo': A = "an "+_A
            else: A = "a "+_A
            if _B[0] in 'aeiuo': B = "an "+_B
            else: B = "a "+_B

            train_texts.append((
                RELATIONS[0].format(A, B),
                torch.tensor([0]),
            )) 
            test_texts.append((
                RELATIONS[1].format(A, B),
                torch.tensor([1]),
            )) 
            
    return train_texts, test_texts


class Controlled_Splits(Dataset):
    def __init__(self,
                 data: List,
                 ):
        super().__init__()
        self.texts = data
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, i): return self.texts[i]