# Knowledge Boosting (Model Collaboration During Low-Latency Inference)

[![Gradio demo](https://img.shields.io/badge/arxiv-abs-green)](https://arxiv.org/abs/2407.11055) [![Gradio demo](https://img.shields.io/badge/Interspeech%202024-pdf-blue)](https://www.isca-archive.org/interspeech_2024/srinivas24_interspeech.html#)

This repository provides code for the Knowledge Boosting architecture proposed in the paper, __Knowledge boosting during low latency inference__, presented at Interspeech 2024. 
Knowledge Boosting is a technique to enhance small model performance on-device during inference with assistance from a large model on a remote device. 
Knowledge boosting allows the large model to provide time-delayed hints to the small model on-device during inference time.

https://github.com/vysri/knowledge-boosting/assets/77802326/6b6d5ede-7546-4c0a-9213-a308980d568f

## Architecture
Our system architecture. The green arrow is present only during large model pre-training.  The red arrows are present  during knowledge boosting.  The black arrows are present both during pre-training and knowledge boosting.
The TF-GridNet model is used to demonstrate results and is the model documented in this repository.

![kb-animation](https://github.com/vysri/knowledge-boosting/assets/77802326/37660ddb-fc5d-470c-924e-7467e8accd27)

## Setup
    # Commands in all sections are run from the repo's top level directory
    conda create --name kb python=3.9
    conda activate kb
    pip install -r requirements.txt

## Training and Evaluation

### Dataset

We use Zenodo to host our datasets. You can access the different datasets below (download both part 1 and 2 for a specific dataset). Each dataset contains a train, validation, and test partition.
* [Target Speech Extraction (TSE) Part 1](https://zenodo.org/records/12575452)
* [Target Speech Extraction (TSE) Part 2](https://zenodo.org/records/12629275)
* [Source Separation (SS) Part 1](https://zenodo.org/records/12629604)
* [Source Separation (SS) Part 2](https://zenodo.org/records/12629652)

Create the `data` directory and untar the data from Zenodo. This example is for target speaker extraction. Replace 'tse' with 'ss' below for source separation:

    # Create data directory
    mkdir data
        
Download all tarballs from the datasets specified above.
        
    # Assemble the train dataset tarball
    cat kb-tse-dataset-train-part*.tar > /scr/kb-tse-dataset-train.tar
        
Untar the datasets from Zenodo into `data` directory.
        
    cd data
    tar -xvf kb-tse-dataset-train.tar -C .
    tar -xvf kb-tse-dataset-val.tar -C .
    tar -xvf kb-tse-dataset-test.tar -C .
    cd ..

### Training
You can run either the baseline models (train large and small models separately before joint training) or run joing configurations. These configurations are under `configs/baselines` and `configs/TSE_joint` or `configs/SS_joint` depending on the task. 

Note that in the joint configurations specifically, you will need to specify the `big_model_init_ckpt` argument which is a PyTorch (.pt) model checkpoint. You may generate your own through training the baseline configurations provided or refer to our model checkpoints ([TSE](https://drive.google.com/file/d/11K71ElmRia_isGCFR8HLpHK9bqw-Q372/view?usp=drive_link), [SS](https://drive.google.com/file/d/1sJIH8MAvCjPKuQBcsjmenTPPGCgiSvS8/view?usp=drive_link)).

    # Usage: trainer.py [-h] --config CONFIG --run_dir RUN_DIR [--resume] [--ckpt CKPT] [--test]
    python -m src.trainer --run_dir <NAME OF DIR TO LOG RUNS> --config <configs/PATH TO CONFIG.json>

## Citation

@inproceedings{srinivas-knowledgeboosting,
  title     = {Knowledge boosting during low-latency inference},
  author    = {Vidya Srinivas and Malek Itani and Tuochao Chen and Sefik Emre Eskimez and Takuya Yoshioka and Shyamnath Gollakota},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4338--4342},
  doi       = {10.21437/Interspeech.2024-331},
  issn      = {2958-1796},
}
