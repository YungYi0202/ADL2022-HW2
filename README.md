# ADL Hw2 Readme

## Environment

```
conda env create -f environment.yml 
```

## Reproduce the prediction

```
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```

## How to train my model

### Parameters

- MC_CKPT_DIR: path/to/dir/of/mc/ckpt
- MC_PRED_DIR: path/to/dir/of/mc/prediction.csv
- QA_CKPT_DIR: path/to/dir/of/qa/ckpt
- QA_PRED_DIR: path/to/dir/of/qa/prediction.csv
- MC_EXPRIMENT_NUM, QA_EXPRIMENT_NUM
    - To distinguish different attempt.
    - The checkpoint files would be stored in `MC_CKPT_DIR / MC_EXPRIMENT_NUM /` and `QA_CKPT_DIR / QA_EXPRIMENT_NUM /`.
    - The prediction files would be stored in `MC_PRED_DIR / MC_EXPRIMENT_NUM /` and `QA_PRED_DIR / QA_EXPRIMENT_NUM /`.

### Stage1: Multiple Choice

Train and Test togother.
```
python run.py \
    --mc_experiment_number $MC_EXPRIMENT_NUM \
    --mc_pretrained_model_name_or_path hfl/chinese-macbert-base \
    --do_mc_train \
    --do_mc_test \
    --mc_ckpt_dir $MC_CKPT_DIR \
    --mc_pred_dir $MC_PRED_DIR \
    --qa_ckpt_dir $QA_CKPT_DIR \
    --qa_pred_dir $QA_PRED_DIR \
```

Test only
```
python run.py \
    --mc_experiment_number $MC_EXPRIMENT_NUM \
    --mc_pretrained_model_name_or_path $MC_CKPT_DIR/$MC_EXPRIMENT_NUM /checkpoint-xxxx \
    --do_mc_test \
    --mc_ckpt_dir $MC_CKPT_DIR \
    --mc_pred_dir $MC_PRED_DIR \
    --qa_ckpt_dir $QA_CKPT_DIR \
    --qa_pred_dir $QA_PRED_DIR \
```

### Stage2: Question Answering

Train and Test togother.
```
python run.py \
    --mc_experiment_number $MC_EXPRIMENT_NUM \
    --qa_experiment_number $QA_EXPRIMENT_NUM \
    --do_qa_train \
    --do_qa_test \
    --qa_pretrained_model_name_or_path hfl/chinese-macbert-large \
    --qa_num_epoch 10 \
    --mc_ckpt_dir $MC_CKPT_DIR \
    --mc_pred_dir $MC_PRED_DIR \
    --qa_ckpt_dir $QA_CKPT_DIR \
    --qa_pred_dir $QA_PRED_DIR \
```

Test only
```
python run.py \
    --mc_experiment_number $MC_EXPRIMENT_NUM \
    --qa_experiment_number $QA_EXPRIMENT_NUM \
    --do_qa_test \
    --qa_pretrained_model_name_or_path hfl/chinese-macbert-large \
    --qa_resume \
    --mc_ckpt_dir $MC_CKPT_DIR \
    --mc_pred_dir $MC_PRED_DIR \
    --qa_ckpt_dir $QA_CKPT_DIR \
    --qa_pred_dir $QA_PRED_DIR \
```


