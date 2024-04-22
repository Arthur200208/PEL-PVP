# PSAC-6mA: DNA 6mA Modification Recognition Model

Welcome to PSAC-6mA model! This project introduces an innovative self-attention capsule network based on sequence positioning for accurate identification of N6-methyladenine (6mA) modification sites in DNA. The 6mA modification plays a crucial role in the growth, development, and disease regulation of organisms, and our PSAC-6mA model employs a unique approach to address this challenge.
Model Introduction

## Model Introduction

PSAC-6mA (Position layer-Self-Attention Capsule-6mA) is an innovative DNA 6mA modification recognition model that adopts the design of a self-attention capsule network. The model combines a positioning layer and self-attention mechanism, enabling precise localization and identification of 6mA modification sites in DNA sequences. The positioning layer is responsible for extracting positional relationships, avoiding the parameter sharing issue in traditional convolutional networks. The self-attention mechanism increases dimensionality, capturing correlations between capsules, allowing the model to efficiently extract features in multiple spatial dimensions, achieving efficient 6mA modification recognition.

## Usage

### Version Dependencies

    Python version: 3.9.18
    PyTorch version: 2.2.0.dev20231011+cu118
    pandas version: 2.1.1
    numpy version: 1.26.0

### Before running the model, ensure that your environment has the necessary dependencies installed. You can use the following commands for installation:

```
# Install PyTorch
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy
pip install numpy==1.26.0

# Install pandas
pip install pandas==2.1.1

```

## Experimental Results

We conducted experiments on DNA datasets from multiple species, and the PSAC-6mA model achieved satisfactory results in identifying 6mA sites. Detailed experimental data and analysis results can be found in our paper (refer to the Citation section for the paper link).

## Contribution and Issue Feedback

If you encounter any issues or want to contribute to the project, feel free to open an issue or submit a pull request. We welcome any form of contribution!

## Citation

If you use our PSAC-6mA model in your research, please cite our paper:

[Paper Title] - [Link to the Paper]

Thank you for using the PSAC-6mA model! If you have any questions or need further assistance, feel free to contact us.

Author: Yu
Contact: 3033795307@qq.com

License Information: [Specify License Information if Applicable]