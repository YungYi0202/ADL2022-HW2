# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection

### Train
Reproduce the result. (The parameters are all set default.)
```shell
python train_intent.py --device cuda:0 --checkpoint_name {your own checkpoint name}
```
### Test 

With your own checkpoint:

```shell
python test_intent.py --test_file {/path/to/testfile} --ckpt_path {{/path/to/checkpoint}} --pred_file {/path/to/predfile}
(Add other args if you have changed other parameters when training.)
```

Or

With the checkpoint provided by `download.sh`:

```shell
sh intent_cls.sh
```

## Slot tagging

### Train
Reproduce the result. (The parameters are all set default.)
```shell
python train_slot.py --device cuda:0 --checkpoint_name {your own checkpoint name}
```

### Test 

With your own checkpoint:

```shell
python test_slot.py --test_file {/path/to/testfile} --ckpt_path {{/path/to/checkpoint}} --pred_file {/path/to/predfile}
(Add other args if you have changed other parameters when training.)
```

Or

With the checkpoint provided by `download.sh`:

```shell
sh slot_tag.sh
```