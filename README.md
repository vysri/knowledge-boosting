# Knowledge Boosting (Model Collaboration During Low-Latency Inference)

[![Gradio demo](https://img.shields.io/badge/arxiv-abs-green)](https://arxiv.org/abs/2211.02250) [![Gradio demo](https://img.shields.io/badge/Interspeech%202024-pdf-blue)](https://arxiv.org/pdf/2211.02250)

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
* [Target Speech Extraction Part 1](https://zenodo.org/records/12575452)
* [Target Speech Extraction Part 2](https://zenodo.org/records/12629275)
* [Source Separation Part 1](https://zenodo.org/records/12629604)
* [Source Separation Part 2](https://zenodo.org/records/12629652)

Create the `data` directory and untar the data from Zenodo. This example is for target speaker extraction. Replace 'tse' with 'ss' below for source separation:

    # Create data directory
    mkdir data
        
Download all tarballs from the datasets specified above.
        
    # Assemble the train dataset tarball
    cat kb-tse-dataset-train-part*.tar > /scr/kb-tse-dataset-train.tar
        
Untar the datasets from Zenodo into data directory
        
    cd data
    tar -xvf kb-tse-dataset-train.tar -C .
    tar -xvf kb-tse-dataset-val.tar -C .
    tar -xvf kb-tse-dataset-test.tar -C .
    cd ..

### Training

    # Usage: trainer.py [-h] --config CONFIG --run_dir RUN_DIR [--resume] [--ckpt CKPT] [--test]
    python -m src.trainer --run_dir <NAME OF DIR TO LOG RUNS> --config <configs/PATH TO CONFIG.json>

## Citation

    @misc{srinivas2024knowledgeboosting,
      title={Knlowledge boosting during low-latency inference}, 
      author={Vidya Srinivas and Malek Itani and Tuochao Chen and Emre Sefik Eskimez and Takuya Yoshioka and Shyamnath Gollakota},
      year={2022},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
    }
