import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import esm
import torch
from MyPeft import getMyPeftModel
from index import sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, AUPRC
from utils import xcl

metrics_dict = {"sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "mcc": mcc,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "AUPRC": AUPRC}

def read_protein_sequences(file):
    '''
    读取FASTA格式的蛋白质序列文件，并将其转换为字符串
    :param file: 蛋白质序列文件
    :return: 蛋白质序列，字符串
    '''
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


def preparePredictData(path, batch_size):
    Positive_seq, sequence_name = read_protein_sequences(path)
    xcl.checkLength(Positive_seq)
    len_data1 = len(Positive_seq)
    Positive_y = torch.ones(len_data1, dtype=torch.float32)
    Positive = tuple(zip(Positive_seq, Positive_y))
    test_batch = DataLoader(Positive, batch_size, shuffle=True, drop_last=False, pin_memory=True)
    return test_batch, Positive_seq, sequence_name


def predict(net, test_data, alphabet, sq_list, sequence_name):
    batch_converter = alphabet.get_batch_converter(500)
    net.eval()
    net = net
    y_list=[]
    result = [0] * len(sq_list)

    loop = tqdm(enumerate(test_data), leave=True, position=0, total=len(test_data))
    for num, (x, y) in loop:
        with torch.no_grad():
            a, b, x = batch_converter(list(zip(y, x)))
            x = x
            h = net.init_hidden(x.size(0))
            y_pre = net(x, h)
            y_pre = y_pre.flatten()
            y_list.extend(y_pre.cpu().detach().numpy().flatten().tolist())

    for i in range(len(y_list)):
        if (y_list[i] >= 0.5):
            result[i] = (sq_list[i], "Vacuolar Protein")
        else:
            result[i] = (sq_list[i], "Non Vacuolar Protein")
    print("处理完成", flush=True)

    df = pd.DataFrame(result, columns=['Protein Sequence', 'Y/N'])
    df.to_csv('result/output.csv', index=True)




if __name__ == '__main__':
    # 设置seed
    SEED = 2024
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    esm2, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

    peft_esm2 = getMyPeftModel(esm2, r=4, a=4, dropout=0.3, m=5)

    peft_esm2.load_state_dict(torch.load("PEL-PVP.pt"))
    test_batch, sq_list, sequence_name = preparePredictData("testdata.fasta", batch_size=1)

    predict(peft_esm2, test_batch, alphabet, sq_list, sequence_name)