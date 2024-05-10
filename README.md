# **PEL-PVP:** Application of Plant Vacuolar Protein Discriminator Based on PEFT ESM-2 and bilayer LSTM in an Unbalanced Dataset

Welcome to PEL-PVP model! 

## Model Introduction

This model leverages the Transformer architecture and self-attention mechanisms to calculate pairwise interactions between residues in sequences, capturing the interdependencies and interactions between amino acids at different positions to extract spatial information. Additionally, a dual-layer LSTM with unique memory units is utilized to handle long-range dependencies, capturing more complex latent features suitable for longer protein sequences. Building on the vast pretrained parameters of the ESM-2 model, it is adaptively fine-tuned for vacuolar proteins using LoRa low-rank adaptation technology, effectively reducing the number of parameters and computational complexity during fine-tuning.

## Usage

### Version Dependencies

    Python version: 3.9.17
    PyTorch version: 2.2.1+cu118
    pandas version: 1.2.4
    numpy version: 1.26.0
    scikit-learn version: 1.3.1
    peft version 0.9.0
    tqdm version 4.65.0
    fair-esm version 2.0.0

### Before running the model, ensure that your environment has the necessary dependencies installed. You can use the following commands for installation:

```
# Install PyTorch
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 

# Install numpy
pip install numpy==1.26.0

# Install pandas
pip install pandas==2.1.1

# Install scikit-learn
pip install scikit-learn==1.3.1

# Install peft 
pip install peft==0.9.0

# Install tqdm                       
pip install tqdm==4.65.0

# Install fair-esm                       
pip install fair-esm==2.0.0
```



## Contribution and Issue Feedback

If you encounter any issues or want to contribute to the project, feel free to open an issue or submit a pull request. We welcome any form of contribution!

## Citation

If you use our PEL-PVP model [here](http://121.36.197.223:8080/py/PEL-PVP.pt) in your research, please cite our paper:

[Paper Title] - [Link to the Paper]

Thank you for using the PEL-PVP model! If you have any questions or need further assistance, feel free to contact us.

Author: Cuilin Xiao
Contact: 20213002749@hainanu.edu.cn