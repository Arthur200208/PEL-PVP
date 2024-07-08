import esm
import numpy as np
import torch
from MyPeft import getMyPeftModel
from Train import train
from Data import prepareData
from utils import xcl

# seed
SEED = 2024
np.random.seed(SEED)
torch.manual_seed(SEED)

epochs = 200
lr = 0.001
accumulation_steps = 1
bacth_size = int(32/accumulation_steps)
truncation_seq_length = int(400*accumulation_steps)
r = 4
a = 4
aa = 0.96
m = 5
esm2, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
peft_esm2 = getMyPeftModel(esm2, r=r, a=a, dropout=0.3, m=5)


optimizer = torch.optim.Adamax(peft_esm2.parameters(),  weight_decay=0.000001)
loss = xcl.Focal_Loss(aa, 2)
modelname = "trained-model"

# dataset
train_batch, test_batch = prepareData(PositiveCSV="dataset/Train-Positive.fasta",
                                           NegativeCSV="dataset/Train-Negative.fasta",
                                           batch_size=bacth_size,
                                           accumulation_steps=accumulation_steps)


print(f"modelname:{modelname} model:{peft_esm2}  lr:{lr} r:{r} a:{a} batch_size:{bacth_size} epochs:{epochs} "
      f"truncation_seq_length:{truncation_seq_length} optm:{optimizer} loss:{loss}")

train(peft_esm2, train_batch, test_batch, optimizer, loss, epochs, alphabet, modelname, truncation_seq_length, accumulation_steps)