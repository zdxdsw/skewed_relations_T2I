from PIL import Image, ImageDraw, ImageFont
import os, torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, Union, List


RELATIONS = ["{0} is on top of {1}.", "{1} is at the bottom of {0}."]
FONT_DIR = "fonts/" #/home/yingshan/clevr_control/scripts/diffuser_icons/


def create_data_split2(nouns):
    train_triplets, test_triplets = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i < j:
                for r, _ in enumerate(RELATIONS): 
                    train_triplets.append((i, j, r))   
            elif i > j:
                for r, _ in enumerate(RELATIONS): 
                    test_triplets.append((i, j, r))   
    return train_triplets, test_triplets


def create_data_split3(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 1 <= i - j < half or -len(nouns)+1 <= i - j <= -half:
                for r, _ in enumerate(RELATIONS): 
                    train_triplets.append((i, j, r))   
            else:
                for r, _ in enumerate(RELATIONS): 
                    test_triplets.append((i, j, r))   
    return train_triplets, test_triplets

def create_data_split4(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if i < half and j < half:
                train_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1))   
            elif i >= half and j >= half:
                test_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1))    
            else:
                test_triplets.append((i, j, 0))  
                test_triplets.append((i, j, 1)) 
    return train_triplets, test_triplets


def create_data_split5(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if 0 <= i < half and half <= j < len(nouns):
                train_label = 0
            elif  0 <= j < half and half <= i < len(nouns):
                train_label = 1
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
                continue

            test_label = 1-train_label  
            train_triplets.append((i, j, train_label))
            test_triplets.append((i, j, test_label))
            
    return train_triplets, test_triplets


def create_data_split6(nouns):
    train_triplets, test_triplets = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i == j: continue
            
            if i+j < len(nouns)-1:
                train_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1))  
            elif i+j > len(nouns)-1:
                train_triplets.append((i, j, 1)) 
                test_triplets.append((i, j, 0))   
    return train_triplets, test_triplets


def create_data_split7(nouns):
    train_triplets, test_triplets = [], []
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            train_label = int(i>j)
            test_label = 1-train_label  
            train_triplets.append((i, j, train_label))
            test_triplets.append((i, j, test_label))
            
    return train_triplets, test_triplets


def create_data_split9(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 0 <= i < half and half <= j < len(nouns): 
                for r, _ in enumerate(RELATIONS): 
                    train_triplets.append((i, j, r)) 
            elif half <= i < len(nouns) and 0 <= j < half: 
                for r, _ in enumerate(RELATIONS): 
                    train_triplets.append((i, j, r)) 
            else:
                for r, _ in enumerate(RELATIONS): 
                    test_triplets.append((i, j, r)) 

    return train_triplets, test_triplets


def create_data_split8(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if i+j >= len(nouns) - 1:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split12(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if 0 <= i < half and half <= j < len(nouns): 
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets

def create_data_split13(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if half <= j < len(nouns): 
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets

def create_data_split14(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 4
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if j < cut or i >= len(nouns) - cut: 
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split15(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 3
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if j < cut or i >= len(nouns) - cut: 
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets

def create_data_split16(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 3
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if j < cut or i >= len(nouns) - cut: 
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            else:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split17(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 4
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if j < cut or i >= len(nouns) - cut: 
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            else:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split20(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if j < cut or i >= len(nouns) - cut: 
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            else:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split18(nouns):
    train_triplets, test_triplets = [], []
    cut = len(nouns) // 3
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            if i < cut:
                if j < cut:
                    test_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1)) 
                else:
                    train_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1))
            elif cut <= i < len(nouns) - cut:
                if j < cut: 
                    test_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                elif cut <= j <= len(nouns) - cut: 
                    train_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                else:
                    train_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1))
            else:
                if j < len(nouns) - cut:
                    test_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                else:
                    test_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split19(nouns):
    train_triplets, test_triplets = [], []
    cut = 2*len(nouns) // 5 
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            
            if i < cut:
                if j < cut:
                    test_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1)) 
                else:
                    train_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1))
            elif cut <= i < len(nouns) - cut:
                if j < cut: 
                    test_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                elif cut <= j < len(nouns) - cut: 
                    train_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                else:
                    train_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1))
            else:
                if j < len(nouns) - cut:
                    test_triplets.append((i, j, 0)) 
                    train_triplets.append((i, j, 1))
                else:
                    test_triplets.append((i, j, 0)) 
                    test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split21(nouns):
    train_triplets, test_triplets = [], []
    half = len(nouns) // 2
    cut = len(nouns) // 20
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue

            if 0 <= i < half and half <= j < len(nouns):
                train_label = 0
            elif  0 <= j < half and half <= i < len(nouns):
                train_label = 1
            elif i < cut and i < j:
                train_label = 0
            elif j < cut and i > j:
                train_label = 1
            elif i >= len(nouns) - cut and i > j:
                train_label = 1
            elif j >= len(nouns) - cut and i < j:
                train_label = 0
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
                continue

            test_label = 1-train_label  
            train_triplets.append((i, j, train_label))
            test_triplets.append((i, j, test_label))
            
    return train_triplets, test_triplets


def create_data_split22(nouns):
    train_triplets, test_triplets = [], []
    margin = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 0 < np.abs(i - j) <= margin or np.abs(i - j) >= len(nouns) - margin:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split24(nouns):
    train_triplets, test_triplets = [], []
    margin = len(nouns) // 10
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 0 < np.abs(i - j) <= margin or np.abs(i - j) >= len(nouns) - margin:
                train_triplets.append((i, j, 0)) 
                train_triplets.append((i, j, 1)) 
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
            
    return train_triplets, test_triplets


def create_data_split23(nouns):
    train_triplets, test_triplets = [], []
    margin = len(nouns) // 3
    half_margin = len(nouns) // 6
    for i, _A in enumerate(nouns):
        for j, _B in enumerate(nouns):
            if i==j: continue
            if 0 < i - j <= half_margin or len(nouns)-half_margin <= i - j < len(nouns): 
                train_label = 0
            elif half_margin < i - j <= margin or len(nouns)-margin <= i - j < len(nouns)-half_margin:
                train_label = 1
            elif 0 < j - i <= half_margin or len(nouns)-half_margin <= j - i < len(nouns):
                train_label = 1
            elif half_margin < j - i <= margin or len(nouns)-margin <= j - i < len(nouns)-half_margin:
                train_label = 0
            else:
                test_triplets.append((i, j, 0)) 
                test_triplets.append((i, j, 1)) 
                continue
            test_label = 1-train_label  
            train_triplets.append((i, j, train_label))
            test_triplets.append((i, j, test_label))
            
    return train_triplets, test_triplets