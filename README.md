This file contains the instructions for applying the code I used for my thesis titled 'Automatic Generation of Personalized Counter Narratives Based on User Profile'

The github repository consists of both scripts and notebooks. The codes were run using Google Colab (https://colab.research.google.com/).

For GPT-2 only:

STEP 1:
The first step is the fine-tuning process for which the following scripts and notebook are necessary:
finetune_gpt2.py
finetune_gpt2.sh
requirements.txt
finetune_gpt2.ipynb

The script 'finetune_gpt2.py' includes special tokens '<hatespeech>' and '<counternarrative>' in the lines 286--288. For the replication only, these are all special tokens necessary. For the experiments injecting profiling information, the special token '<personalinformation>' can be added to the dictionary.
The script finetune_gpt2.sh requires the training and validation data in lines 16 and 17 respectively.

Once these modifications are made and the scripts are uploaded on the drive, the notebook 'finetune_gpt2.ipynb' can be run. The requirements are specified and accessed through 'requirements.txt'.

STEP 2:
The second step is the generation of the counter narratives using following notebook:
generation_gpt2.ipynb
The testset can be inserted in the second cell. The output of the first step, including the best model, is saved under a folder called checkpoints which is indicated in the first cell under 'Load model and prepare data'.
For generating, the user can choose the desired decoding mechanism(s) for the generation.

STEP 3:
The third step is the automatic evaluation. The generated counter narratives from step 2 have to be compared to the gold counter narratives. Accordingly, a data file containing both the generated CN and gold CN has to be created.
The notebook for running the automatic evaluation is the following:
metrics.ipynb


For GPT-3.5 only:

The script used for generating the detailed profiles for one of the experiments in the thesis is the following:
profile_generation.py
The code can be run by inserting the own API key.

The script used for generating the personalized CNs by taking the gold CNs as reference is the following:
cn_personalization_updated.py
The code can be run by inserting the own API key.
