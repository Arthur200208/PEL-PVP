import gc
import os
import torch
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from index import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC

metrics_dict = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "mcc": mcc,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "AUPRC": AUPRC
                }


def train(net, train_data, test_data, opt, loss, epochs, alphabet, modelname, truncation_seq_length=500, accumulation_steps= 1):
    highestAcc = None
    writer = SummaryWriter(log_dir=f'logs/{modelname}')
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=truncation_seq_length)
    net = net.cuda()
    for epoch in range(epochs):
        train_pre, val_pre, train_labels, val_labels, features, features0 = [], [], [], [], [], []
        net.train()
        table = PrettyTable(['name', 'value'])
        table2 = PrettyTable(['T\F', 'P', 'N'])
        loop = tqdm(enumerate(train_data), leave=True, position=0, total=len(train_data), desc=f'Train Epoch {epoch}')
        for num, (x, y) in loop:
            _, _, x = batch_converter(list(zip(y, x)))
            x = x.cuda()
            y = y.cuda()
            opt.zero_grad(set_to_none=True)
            h = net.init_hidden(x.size(0))
            y_hat = net.forward(x, h)
            y_hat = y_hat.flatten()
            l = loss(y_hat, y)
            l.backward()
            opt.step()  # update parameters of net
            # # 3. update parameters of net
            # if ((num + 1) % accumulation_steps) == 0:
            #     # optimizer the net

            train_pre.extend(y_hat.cpu().clone().detach().numpy().flatten().tolist())
            train_labels.extend(y.cpu().clone().detach().numpy().astype('int32').flatten().tolist())

        for key in metrics_dict.keys():
            if (key != "auc" and key != "AUPRC"):
                metrics = metrics_dict[key](train_labels, train_pre, thresh=0.5)
            else:
                metrics = metrics_dict[key](train_labels, train_pre)
            table.add_row(["Train "+key, "{:.2f}%".format(metrics*100)])
            writer.add_scalar("Train_" + key, metrics, epoch + 1)
        print(table)

        tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_pre, thresh=0.5)
        table2.add_row(['P', tp_t, fn_t])
        table2.add_row(['N', fp_t, tn_t])
        print(table2)
        del x, y, h,table, y_hat, tn_t, fp_t, fn_t, tp_t, train_labels, train_pre, metrics

        gc.collect()
        torch.cuda.empty_cache()

        net.eval()
        net = net.cuda()
        table = PrettyTable(['name', 'value'])
        table2 = PrettyTable(['T\F', 'P', 'N'])
        loop = tqdm(enumerate(test_data), leave=True, position=0, total=len(test_data), desc=f'Validation Epoch {epoch}')
        for num, (x, y) in loop:
            with torch.no_grad():
                _, _, x = batch_converter(list(zip(y, x)))
                x = x.cuda()
                y = y.cuda()
                h = net.init_hidden(x.size(0))
                y_hat = net.forward(x, h)
                y_hat = y_hat.flatten()
                val_pre.extend(y_hat.cpu().detach().numpy().flatten().tolist())
                val_labels.extend(y.cpu().detach().numpy().astype('int32').flatten().tolist())
        loss_epoch = loss(torch.tensor(val_pre).float(), torch.tensor(val_labels).float())

        print(f"loss:{loss_epoch}")
        for key in metrics_dict.keys():
            if (key != "auc" and key != "AUPRC"):
                metrics = metrics_dict[key](val_labels, val_pre, thresh=0.5)
                if (key == "f1"):
                    if (highestAcc == None) or (highestAcc < metrics):
                        highestAcc = metrics
                        torch.save(net.state_dict(), os.path.join("peft-model/"+modelname+".pt"))
                        print("Weights Saved")
            else:
                metrics = metrics_dict[key](val_labels, val_pre)
            table.add_row(["Validation "+key, "{:.2f}%".format(metrics*100)])
            writer.add_scalar("Validation_" + key, metrics, epoch + 1)

        print(table)
        tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_pre, thresh=0.5)
        table2.add_row(['P', tp_t, fn_t])
        table2.add_row(['N', fp_t, tn_t])
        print(table2)



        del x, y, h, y_hat, tn_t, fp_t, fn_t, tp_t, val_labels, val_pre, metrics, features, features0
        gc.collect()
        torch.cuda.empty_cache()