# Sign Language Recognition

## Members

- Đào Quý An - 21020602
- Dương Quang Minh - 21020219
- Lê Văn Bảo - 21020171
- Nguyễn Thạch Anh -

## Description

Final Project for Human - Machine Interaction Class INT3412E 20

This repo was made by UET-VNU students

Welcome to ASIGN, the American Sign Language Recognition and Inquiry Support Application.

ASIGN is an application designed to assist in recognizing and researching American Sign Language (ASL).

ASIGN provides a powerful and user-friendly interactive experience, Here are the main features of the application:

- Recognition of distinct sign languages from video
- Spelling recognition using American Sign Language
- Sign language word-by-word learning videos
- Question and answer section
- Analysis and practice from practice videos

## Installation

```bash
# Clone project
git clone https://github.com/minhminh2k/Sign_Language_Recognition.git

# Create conda environment
conda create -p ./env python=3.10

conda activate ./env

# Install dependencies
pip install -r requirements.txt
conda install ffmpeg

# Run
streamlit run Sign_Language_Recognition.py

# Close
Ctrl + C before closing the app
```

## How to run

```bash
streamlit run ASL.py
```

## Reference

- [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/overview)
- [Google - American Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling/overview)
- [Training the ISLR model](https://www.kaggle.com/competitions/asl-signs/discussion/406684)
- [Training the ASLFR model](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434588)
