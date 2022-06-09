# RST Discourse Parsing with Second-Stage EDU-Level Pre-training

Implementation of the paper [RST Discourse Parsing with Second-Stage EDU-Level Pre-training,  ACL'22](https://aclanthology.org/2022.acl-long.294/)



## Requirements

PyTorch 1.7.0

Python 3.6

Transformers 3.5.0



## Data

[RST-DT](https://catalog.ldc.upenn.edu/LDC2002T07)
[GUM](https://github.com/amir-zeldes/gum/tree/master/rst)



## PLM

[XLNET+NEP+DMP](https://drive.google.com/file/d/1mOsePb4Gz9UmLzjbki-RO7f3tukJ8bna/view?usp=sharing)



## Training 

python -u driver/TrainTest.py  --config_file config.rst.xlnet.nep.dmp

