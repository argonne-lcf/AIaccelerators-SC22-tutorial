# BERT (language model)

Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google.

# BERT on Sambanova


1. Login to SN:  
```
ssh ALCFUserID@sambanova.alcf.anl.gov 
ssh sm-01 (or sm-02)
```

2. SDK setup:  
```
source /software/sambanova/envs/sn_env.sh
```

3. Copy scripts:  
```
cp /var/tmp/Additional/slurm/Models/ANL_Acceptance_RC1_11_5/bert_train-inf.sh ~/
```

4. Run scripts:  
```
cd ~
./bert_train-inf.sh
```
