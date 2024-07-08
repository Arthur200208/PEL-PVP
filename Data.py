import re
import torch
from torch.utils.data import random_split, DataLoader
from utils import xcl

def read_protein_sequences(file):
    with open(file) as f:
        data = f.read()
    # 检查文件是否为FASTA格式
    if re.search('>', data) == None:
        print("Please input correct FASTA format protein sequence！！！")
    else:
        # 将文件中的每一行按照>分割，并将每一行拆分为字符串
        records = data.split('>')[1:]
        sequences = []
        sequence_name = []
        # 遍历每一行
        for fasta in records:
            array = fasta.split('\n')
            # 将每一行以\n分割，并将每一行拆分为字符串
            header, sequence = array[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
            # 将每一行的头部信息存入字符串
            name = header
            # 将每一行的序列信息存入字符串
            sequences.append(sequence)
            # 将每一行的序列信息存入字符串
            sequence_name.append(name)
        # 返回字符串
        return sequences, sequence_name


def prepareData(PositiveCSV, NegativeCSV, batch_size, accumulation_steps=1):
    Positive, _ = read_protein_sequences(PositiveCSV)
    Negative, _ = read_protein_sequences(NegativeCSV)
    xcl.checkLength(Positive)
    xcl.checkLength(Negative)

    len_data1 = len(Positive)
    len_data2 = len(Negative)
    # print(len_data1, len_data2)
    Positive_y = torch.ones(len_data1, dtype=torch.float32)
    Negative_y = torch.zeros(len_data2, dtype=torch.float32)

    Positive = tuple(zip(Positive, Positive_y))
    Negative = tuple(zip(Negative, Negative_y))

    train_dataset, test_dataset = random_split(dataset=Positive + Negative, lengths=[0.8, 0.2],
                                               generator=torch.Generator())
    train_batch = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False, pin_memory=True)
    test_batch = DataLoader(test_dataset, batch_size*accumulation_steps, shuffle=True, drop_last=False, pin_memory=True)

    return train_batch, test_batch


def prepareInDeData(PositiveCSV, NegativeCSV, batch_size):
    Positive, _ = read_protein_sequences(PositiveCSV)
    Negative, _ = read_protein_sequences(NegativeCSV)
    xcl.checkLength(Positive)
    xcl.checkLength(Negative)

    len_data1 = len(Positive)
    len_data2 = len(Negative)
    # print(len_data1, len_data2)
    Positive_y = torch.ones(len_data1, dtype=torch.float32)
    Negative_y = torch.zeros(len_data2, dtype=torch.float32)

    Positive = tuple(zip(Positive, Positive_y))
    Negative = tuple(zip(Negative, Negative_y))

    test_batch = DataLoader(Positive+Negative, batch_size, shuffle=True, drop_last=False, pin_memory=True)
    return test_batch

def preparePredictData(PositiveCSV, batch_size):
    Positive, _ = read_protein_sequences(PositiveCSV)
    xcl.checkLength(Positive)
    len_data1 = len(Positive)
    Positive_y = torch.ones(len_data1, dtype=torch.float32)
    Positive = tuple(zip(Positive, Positive_y))
    test_batch = DataLoader(Positive, batch_size, shuffle=True, drop_last=False, pin_memory=True)
    return test_batch
