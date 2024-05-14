
import spacy  
nlp = spacy.load("en_core_web_lg") 
import numpy as np

def preprocess(text):
    text = text.replace(",", " ,")
    text = " ".join(text.strip().split())
    return text

def extract_tuple(text, r):
    if not f" {r} " in text: return None
    # Doesn't inform a top-bottom relationship
    if r == "on" and any(b in text for b in ["on the side of", "on side", "on top.", "on the top."]): return None
    doc = nlp(text)

    # figure out pos_id of r
    # locate substring
    rs_idx = text.find(f" {r} ")+1
    re_idx = rs_idx + len(f"{r} ")
    tokens = [token for token in doc]
    for i, token in enumerate(tokens):
        if token.idx == rs_idx: rs = i
        elif token.idx == re_idx: re = i
    
    #token_texts = [token.text for token in doc]
    #token_pos_ = [token.pos_ for token in doc]
    #token_dep_ = [token.dep_ for token in doc]
    subj, obj = "", ""

    for tok in tokens:
        if tok.head.i == re-1 and tok.pos_ == "NOUN": 
            obj = tok
    
    if obj == "": return None
    # handle "a set/pair of ___"
    # add "nummod" to handle "two sets of ____"
    if set(["det", "prep"]).issubset(set([a.dep_ for a in obj.children])) or set(["nummod", "prep"]).issubset(set([a.dep_ for a in obj.children])):
        of_tok = [a for a in obj.children if a.dep_ == "prep"][0]
        if of_tok.text == "of":
            of_children = [a for a in of_tok.children]
            if len(of_children): obj = of_children[0].text
    if not isinstance(obj, str): obj = obj.text  

    if tokens[rs].head.pos_ == "NOUN": subj = tokens[rs].head
    else:
        target_i = tokens[rs].head.i
        if tokens[rs].dep_ == "ROOT": # find the closest NOUN prior to the ROOT
            for i in range(rs-1, 0, -1):
                if tokens[i].pos_ == "NOUN":
                    subj = tokens[i]
        else:
            for tok in tokens:
                if tok.head.i == target_i and tok.pos_ == "NOUN":
                    subj = tok
            if len(subj) == 0:
                for i in range(rs-1, 0, -1):
                    if tokens[i].pos_ == "NOUN": subj = tokens[i]

    
    if subj == "": return None
    # handle "a set/pair of ___"
    # add "nummod" to handle "two sets of ____"
    if set(["det", "prep"]).issubset(set([a.dep_ for a in subj.children])) or set(["nummod", "prep"]).issubset(set([a.dep_ for a in subj.children])):
        of_tok = [a for a in subj.children if a.dep_ == "prep"][0]
        if of_tok.text == "of":
            of_children = [a for a in of_tok.children]
            if len(of_children): subj = of_children[0].text
    if not isinstance(subj, str): subj = subj.text     
    """
    look_for = {"ROOT": None, "dobj": None, "nsubj": None, "pobj": None}
    for i in range(rs-1, 0, -1):
        if token_pos_[i] == "NOUN" and token_dep_[i] in look_for and look_for[token_dep_[i]] is None:
            look_for[token_dep_[i]] = tokens[i]            
    for k in look_for:
        if look_for[k] is not None: 
            subj = look_for[k]
            break

    look_for = {"ROOT": None, "dobj": None, "nsubj": None, "pobj": None}
    for i in range(re, len(tokens)):
        if token_pos_[i] == "NOUN" and token_dep_[i] in look_for:
            obj = tokens[i]
            break
    if len(subj)*len(obj) == 0: 
        print(text)
        #return None
    """
    tuple = (subj, obj, r)

    # remove edge cases
    if obj == "top": return None # "A piece of pie with meringue on top has a flaky crust and pumpkin for filling"
    if subj == "side" or obj == "side": return None # "An Olympus Tough TG-310 camera sitting on it's side in front of the box.", "A white , gas stove with light tan counter tops on each side."
    if subj == "display" or obj == "display": return None # "statues of nude cupids on display in front of a window", "A group of people walk around a room full of new cars on display."
    return tuple 


def extract_complete_subset(heatmap, nouns, image_complete=True, linguistic_complete=False):
    dead = []
    if image_complete == False and linguistic_complete == False:
        print("Nothing happened since neither completion requirement was provided")
        return heatmap, dead
    
    
    iters = 0
    killed_this_iter = True
    
    while np.sum(heatmap['image']) * np.sum(heatmap['linguistic']) > 0 and killed_this_iter:
        killed_this_iter = False
        if image_complete:
            sum_axis0, sum_axis1 = np.sum(heatmap['image'], axis=0), np.sum(heatmap['image'], axis=1)
            for i, s in enumerate(sum_axis0):
                if i in dead: continue
                if s == 0:
                    for j in range(len(nouns)):
                        # drop row i
                        heatmap['image'][i][j] = 0
                        # drop row i and col i
                        heatmap['linguistic'][i][j] = 0
                        heatmap['linguistic'][j][i] = 0

                    dead.append(i)
                    killed_this_iter = True
            for i, s in enumerate(sum_axis1):
                if i in dead: continue
                if s == 0:
                    for j in range(len(nouns)):
                        # drop col i
                        heatmap['image'][j][i] = 0
                        # drop row i and col i
                        heatmap['linguistic'][i][j] = 0
                        heatmap['linguistic'][j][i] = 0
                    dead.append(i)
                    killed_this_iter = True
        
        if linguistic_complete:
            sum_axis0, sum_axis1 = np.sum(heatmap['linguistic'], axis=0), np.sum(heatmap['linguistic'], axis=1)
            for i, s in enumerate(sum_axis0):
                if i in dead: continue
                if s == 0:
                    for j in range(len(nouns)):
                        # drop row i
                        heatmap['linguistic'][i][j] = 0
                        # drop row i and col i
                        heatmap['image'][i][j] = 0
                        heatmap['image'][j][i] = 0

                    dead.append(i)
                    killed_this_iter = True
            for i, s in enumerate(sum_axis1):
                if i in dead: continue
                if s == 0:
                    for j in range(len(nouns)):
                        # drop col i
                        heatmap['linguistic'][j][i] = 0
                        # drop row i and col i
                        heatmap['image'][i][j] = 0
                        heatmap['image'][j][i] = 0
                    dead.append(i)
                    killed_this_iter = True

        iters += 1
        print(f"dead nouns = {len(dead)}")
        print(f"""finish iter {iters}, 
            remaining examples = {np.sum(heatmap['image'])} (image), 
            {np.sum(heatmap['linguistic'])} (linguistic)""")    
        
    return heatmap, dead
