import random, time, os, pytz, argparse, yaml, sys
sys.path.append("../")
sys.path.append("../probing")
import numpy as np
from probing.probe import *
from tqdm import tqdm, trange
from probing_utils import *
from controlled_splits_dataset import *
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs.yaml")
#parser.add_argument('--split_method', type=str)
#parser.add_argument('--nouns_file', type=str)
args = parser.parse_args()

# To ensure we get reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def train(model, dataloader, optim, criterion, device):
    
    model.train()
    epoch_loss, epoch_acc = 0, 0
    #print("model dtype = ", next(model.parameters()).dtype)
    for (texts, labels) in tqdm(dataloader, desc="Training", mininterval=10):
        
        optim.zero_grad()
        
        output = model(list(texts)) # batch_size*2, num_classes

        gth = torch.cat([labels, 1 - (labels % 2) + 2*(labels // 2)], dim=1).long().view((-1,)).to(device)
        #print(texts)
        #print(gth)
        #print(f"gth.size() = {gth.size()}")
        #exit()
        #print(f"output.size() = {output.size()}")

        loss = criterion(output, gth)
        
        acc = get_acc(output, gth)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

def val(model, dataloader, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0

    with torch.no_grad():
        for (texts, labels) in tqdm(dataloader, desc="Validating", mininterval=10):
            
            output = model(list(texts)) # batch_size*2, num_classes

            gth = torch.cat([labels, 1 - (labels % 2) + 2*(labels // 2)], dim=1).long().view((-1,)).to(device)
        
            loss = criterion(output, gth)
            acc = get_acc(output, gth)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

""" Load Configs """
with open(args.config_file, 'rb') as f: config = yaml.full_load(f.read()) # dict
#os.mkdir(os.path.join(config["save_dir"], date))
#yaml.dump(config, open(os.path.join(config["save_dir"], date, "configs.yaml"), "w"))

""" Prepare Data """
with open(config['nouns_file'], "r") as f: nouns = [l.strip() for l in f.readlines()]
if "max_num_nouns" in config: nouns = nouns[:config['max_num_nouns']]
train_texts, test_texts = eval(f"create_data_{config['split_method']}")(nouns)
train_data = Controlled_Splits(train_texts)
test_data = Controlled_Splits(test_texts)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_dl = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
test_dl = DataLoader(test_data, shuffle=False, batch_size=config['batch_size'])

print("Prepare Data: finish\n")


""" Prepare Model """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PROBE(
    hidden_dim_multipliers = config['hidden_dim_multipliers'], 
    num_classes = 2, 
    lm = config['lm'], 
    lm_kwargs = config['lm_kwargs'],
    lm_howto_select_encoding_positions = config['lm_howto_select_encoding_positions'],
).to(device)

optim = Adam(model.parameters(), lr = float(config['lr']))
criterion = nn.CrossEntropyLoss().to(device)

print("Trainable Params: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
print("Prepare Model: finish\n")


""" Training """
best_val_loss = float('inf')
best_val_acc = 0.0

for epc in trange(config['epochs']):
    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_dl, optim, criterion, device)
    test_loss, test_acc = val(model, test_dl, criterion, device)

    torch.save(model.state_dict(), os.path.join(config["save_dir"], config["date"], "model.pt"))

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epc+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print()
print(f"Training: finish {config['date']}\n")


# """ Testing """

# model.load_state_dict(torch.load(os.path.join(config['save_dir'], config["date"], "model.pt")))
# print("Load ckpt with the best val_loss")

# test_loss, test_acc = val(model, test_dl, criterion, device)
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# test_loss, test_acc = val(model, ood_test_dl, criterion, device)
# print(f'OOD Test Loss: {test_loss:.3f} | OOD Test Acc: {test_acc*100:.2f}%')

# print("Testing: finish")
