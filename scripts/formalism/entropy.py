import pandas as pd
import numpy as np
from tqdm import trange
from dataset import *

def Transpose(triplets, apply_to_relations=[1]):
    tmp = []
    for t in triplets:
        if t[-1] in apply_to_relations: tmp.append((t[1], t[0], t[-1]))
        else: tmp.append(t)
    return tmp

def create_synthetic_df(split, num_nouns, transpose=False):
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    if transpose: train_triplets, test_triplets = Transpose(train_triplets), Transpose(test_triplets)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])
    return df

def concept_centric_entropy(split, num_nouns, num_relations):
    
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation1 = 0
    for i in range(num_nouns):
        count_I_i = len(df[(df.O1 == i) | (df.O2 == i)])
        for j in range(num_nouns): 
            for r in range(num_relations):
                if not i==j:
                    count_O1_R_I_i = len(df[(df.O1 == j) & (df.O2 == i) & (df.R == r)])
                else:
                    count_O1_R_I_i = len(df[(df.O1 == j) & (df.R == r)])
                if count_O1_R_I_i > 0:
                    p = count_O1_R_I_i / count_I_i
                    print(-p * np.log(p))
                    summation1 += -p * np.log(p)
    summation1 /= num_nouns

    summation2 = 0
    for i in range(num_nouns):
        count_I_i = len(df[(df.O1 == i) | (df.O2 == i)])
        for j in range(num_nouns): 
            for r in range(num_relations):
                if not i==j:
                    count_O2_R_I_i = len(df[(df.O2 == j) & (df.O1 == i) & (df.R == r)])
                else:
                    count_O2_R_I_i = len(df[(df.O2 == j) & (df.R == r)])
                if count_O2_R_I_i > 0:
                    p = count_O2_R_I_i / count_I_i
                    -p * np.log(p)
                    summation2 += -p * np.log(p)
    summation2 /= num_nouns

    return np.mean([summation1, summation2])


def concept_centric_entropy2(split, num_nouns, num_relations):
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation = 0
    for i in range(num_nouns):
        count_I_i = len(df[(df.O1 == i) | (df.O2 == i)])
        for j1 in range(num_nouns): 
            for j2 in range(num_nouns):
                for r in range(num_relations):
                    if i==j1 or i==j2:
                        count_O1_O2_R_I_i = len(df[(df.O1 == j1) & (df.O2 == j2) & (df.R == r)])
                    else:
                        count_O1_O2_R_I_i = 0
                    if count_O1_O2_R_I_i > 0:
                        p = count_O1_O2_R_I_i / count_I_i
                        summation += -p * np.log(p)
                        #print(-p * np.log(p))
    summation /= num_nouns
    return summation

def concept_centric_entropy3(num_nouns, num_relations, df):
    # Exp[Entr[P(R|O)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")

    for j in range(num_nouns):
        count_c_j = len(df[(df.O1 == j) | (df.O2 == j)])
        for r in range(num_relations):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)]) + len(df[(df.R == r) & (df.O2 == j)]) - len(df[(df.R == r) & (df.O1 == j) & (df.O2 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_c_j)
                #print(summation)
    summation /= count_all
    return summation

def concept_centric_entropy4(num_nouns, num_relations, df):
    # Exp[Entr[P(R|O1)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")

    for j in range(num_nouns):
        count_c_j = len(df[(df.O1 == j)])
        for r in range(num_relations):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_c_j)
    summation /= count_all
    return summation

def concept_centric_entropy5(num_nouns, num_relations, df):
    # Exp[Entr[P(R|O2)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")

    for j in range(num_nouns):
        count_c_j = len(df[(df.O2 == j)])
        for r in range(num_relations):
            count_O_R = len(df[(df.R == r) & (df.O2 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_c_j)
    summation /= count_all
    return summation


def group_centric_entropy(split, num_nouns, num_relations):
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation = 0

    for i1 in range(num_nouns):
        for i2 in range(num_nouns):
            count_I_i1_I_i2 = len(df[(df.O1 == i1) | (df.O2 == i2)]) + len(df[(df.O1 == i2) | (df.O2 == i1)])
            for j1 in range(num_nouns):
                for j2 in range(num_nouns):
                    for r in range(num_relations):
                        if (i1==j1 and i2==j2) or (i1==j2 and i2==j1):
                            count_O1_O2_R_I_i1_I_i2 = len(df[(df.O1 == j1) & (df.O2 == j2) & (df.R == r)]) +\
                                                        len(df[(df.O1 == j2) & (df.O2 == j1) & (df.R == r)])
                        else:
                            count_O1_O2_R_I_i1_I_i2 = 0
                        if count_O1_O2_R_I_i1_I_i2>0:
                            p = count_O1_O2_R_I_i1_I_i2 / count_I_i1_I_i2
                            summation += -p * np.log(p)
    summation /= num_nouns**2
    return summation

def relation_centric_entropy2(split, num_nouns, num_relations):
    
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation = 0
    for r in range(num_relations):
        count_R = len(df[(df.R == r)])
        for j1 in range(num_nouns): 
            for j2 in range(num_nouns):
                count_O1_O2_R = len(df[(df.O1 == j1) & (df.O2 == j2) & (df.R == r)])
                if count_O1_O2_R > 0:
                    p = count_O1_O2_R / count_R
                    summation += -p * np.log(p)
    summation /= num_relations
    return summation

def relation_centric_entropy(split, num_nouns, num_relations):
    
    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation1 = 0
    for r in range(num_relations):
        count_R = len(df[(df.R == r)])
        for j in range(num_nouns): 
            count_O1_R = len(df[(df.O1 == j) & (df.R == r)])
            if count_O1_R > 0:
                p = count_O1_R / count_R
                summation1 += -p * np.log(p)
    summation1 /= num_relations

    summation2 = 0
    for r in range(num_relations):
        count_R = len(df[(df.R == r)])
        for j in range(num_nouns): 
            count_O2_R = len(df[(df.O2 == j) & (df.R == r)])
            if count_O2_R > 0:
                p = count_O2_R / count_R
                summation2 += -p * np.log(p)
    summation2 /= num_relations

    return np.mean([summation1, summation2])

def relation_centric_entropy3(num_nouns, num_relations, df):
    # Exp[Entr[P(O|R)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)]) + len(df[(df.R == r) & (df.O2 == j)]) - len(df[(df.R == r) & (df.O1 == j) & (df.O2 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_r)
    summation /= count_all
    return summation

def relation_centric_entropy4(num_nouns, num_relations, df):
    # Exp[Entr[P(O1|R)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_r)
    summation /= count_all
    return summation

def relation_centric_entropy5(num_nouns, num_relations, df):
    # Exp[Entr[P(O2|R)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O2 == j)])
            if count_O_R > 0:
                summation += - count_O_R * np.log(count_O_R/count_r)
    summation /= count_all
    return summation


def relation_scope_divergence(split, num_nouns, num_relations):

    nouns = list(range(num_nouns))
    train_triplets, test_triplets = eval(f"create_data_{split}")(nouns)
    df = pd.DataFrame(train_triplets, columns =['O1', 'O2', 'R'])

    summation1 = 0
    for r in range(num_relations):
        count_R = len(df[(df.R == r)])
        for i in range(num_nouns):
            count_O1_R = len(df[(df.O1 == i) & (df.R == r)])
            if count_O1_R > 0:
                p = count_O1_R / count_R
                summation1 += p * np.log(num_nouns * p)
    summation1 /= num_relations

    summation2 = 0
    for r in range(num_relations):
        count_R = len(df[(df.R == r)])
        for i in range(num_nouns):
            count_O2_R = len(df[(df.O2 == i) & (df.R == r)])
            if count_O2_R > 0:
                p = count_O2_R / count_R
                summation2 += p * np.log(num_nouns * p)
    summation2 /= num_relations

    return np.mean([summation1, summation2])

def divergence(num_nouns, num_relations, df):
    # KL[P(O|R) || P(O)]

    summation = 0
    count_all = len(df)
    print(f"count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)]) + len(df[(df.R == r) & (df.O2 == j)]) - len(df[(df.R == r) & (df.O1 == j) & (df.O2 == j)])
            count_c_j = len(df[(df.O1 == j) | (df.O2 == j)])
            if count_O_R > 0:
                summation += count_O_R * np.log(count_O_R * count_all * 2/ (count_c_j*count_r))
    
    summation /= count_all
    return -summation
    
def divergence2(num_nouns, num_relations, df):
    # KL[P(O1|R) || P(O1)]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O1 == j)])
            count_c_j = len(df[(df.O1 == j)])
            if count_O_R > 0:
                summation += count_O_R * np.log(count_O_R * count_all/ (count_c_j*count_r))
    
    summation /= count_all
    return -summation

def divergence3(num_nouns, num_relations, df):
    # KL[P(O2|R) || P(O2)]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for r in range(num_relations):
        count_r = len(df[df.R == r])
        for j in range(num_nouns):
            count_O_R = len(df[(df.R == r) & (df.O2 == j)])
            count_c_j = len(df[(df.O2 == j)])
            if count_O_R > 0:
                summation += count_O_R * np.log(count_O_R * count_all/ (count_c_j*count_r))
    
    summation /= count_all
    return -summation


def concept_role_entropy(num_nouns, df):
    # Exp[Entr[P(O1|O)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for i in range(num_nouns):
        count_O_c_i = len(df[(df.O1 == i) | (df.O2 == i)])
        for j in range(num_nouns):
            if i==j: nume = len(df[(df.O1 == i)])
            else: nume = len(df[(df.O1 == j) & (df.O2 == i)])
            if nume > 0:
                summation += - nume * np.log(nume/count_O_c_i)
    summation /= count_all
    return summation


def concept_role_entropy2(num_nouns, df):
    # Exp[Entr[P(O2|O)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for i in range(num_nouns):
        count_O_c_i = len(df[(df.O1 == i) | (df.O2 == i)])
        for j in range(num_nouns):
            if i==j: nume = len(df[(df.O2 == i)])
            else: nume = len(df[(df.O1 == i) & (df.O2 == j)])
            if nume > 0:
                summation += - nume * np.log(nume/count_O_c_i)
    summation /= count_all
    return summation

def role_association(num_nouns, df):
    # Exp[Entr[P(O1|O2)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for i in range(num_nouns):
        count_O2_c_i = len(df[(df.O2 == i)])
        for j in range(num_nouns):
            count_O1_c_j_O2_c_i = len(df[(df.O1 == j) & (df.O2 == i)])
            if count_O1_c_j_O2_c_i > 0:
                summation += - count_O1_c_j_O2_c_i * np.log(count_O1_c_j_O2_c_i/count_O2_c_i)
    summation /= count_all
    return summation

def role_association2(num_nouns, df):
    # Exp[Entr[P(O2|O1)]]

    summation = 0
    count_all = len(df)
    #print(f"{split} count_all = {count_all}")
    for i in range(num_nouns):
        count_O1_c_i = len(df[(df.O1 == i)])
        for j in range(num_nouns):
            count_O2_c_j_O1_c_i = len(df[(df.O2 == j) & (df.O1 == i)])
            if count_O2_c_j_O1_c_i > 0:
                summation += - count_O2_c_j_O1_c_i * np.log(count_O2_c_j_O1_c_i/count_O1_c_i)
    summation /= count_all
    return summation

def concept_entropy(num_nouns, df):
    # Entr[P(O1)]

    summation = 0
    count_all = len(df)

    for i in range(num_nouns):
        count_O1_c_i = len(df[(df.O1 == i)])
        if count_O1_c_i > 0:
            p = count_O1_c_i/count_all
            summation += - p * np.log(p)
    return summation

def concept_entropy2(num_nouns, df):
    # Entr[P(O2)]

    summation = 0
    count_all = len(df)

    for i in range(num_nouns):
        count_O2_c_i = len(df[(df.O2 == i)])
        if count_O2_c_i > 0:
            p = count_O2_c_i/count_all
            summation += - p * np.log(p)
    return summation


def concept_entropy0(num_nouns, df):
    # Entr[P(O)]

    summation = 0
    count_all = len(df)

    for i in range(num_nouns):
        count_O_c_i = len(df[(df.O1 == i) | (df.O2 == i)])
        if count_O_c_i > 0:
            p = count_O_c_i/(count_all*2)
            summation += - p * np.log(p)
    return summation


def concept_role_index_entropy(num_nouns, df):
    # E[Entr[P(I_{O1=c_i} | O=c_i)]]

    summation = 0
    count_all = len(df)
    for i in range(num_nouns):
        count_O_c_i = len(df[(df.O1 == i) | (df.O2 == i)])
        count_O1_c_i = len(df[(df.O1 == i)])
        count_O2_c_i = len(df[(df.O2 == i)])
        delta = 0
        if count_O1_c_i > 0:
            delta -= count_O1_c_i * np.log(count_O1_c_i / count_O_c_i)
        if count_O2_c_i > 0:
            delta -= count_O2_c_i * np.log(count_O2_c_i / count_O_c_i)
        #print(f"#objs={num_nouns}, noun={i}, delta={delta} \ncount_O1_c_i={count_O1_c_i}, count_O2_c_i={count_O2_c_i}, count_O_c_i={count_O_c_i}")
        summation +=  delta
    summation /= count_all*2
    return summation

    



