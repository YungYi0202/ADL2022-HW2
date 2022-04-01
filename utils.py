import random
import torch
import numpy as np
import json
from datasets import Dataset, DatasetDict
# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

ending_names = [f"ending{i}" for i in range(4)]
def get_ending_names():
    return ending_names

# swag["train"][0]
# {'ending0': 'passes by walking down the street playing their instruments.',
#  'ending1': 'has heard approaching them.',
#  'ending2': "arrives and they're outside dancing and asleep.",
#  'ending3': 'turns the lead singer watches the performance.',
#  'label': 0,
#  'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
#  'sent2': 'A drum line',
# }
def swag_like_dataset(splits, questions, paragraphs, candidate_paragraph_ids, answers):
    tmp_dict = {split: _swag_like_dataset(questions[split], paragraphs, candidate_paragraph_ids[split], (answers[split] if split in answers.keys() else None)) for split in splits}
    dataset = DatasetDict()
    # using your `Dict` object
    for k,v in tmp_dict.items():
        dataset[k] = Dataset.from_dict(v)
    return dataset

def get_match_index(elements, target):
    for i, element in enumerate(elements):
        if element == target:
            return i
    raise ValueError

def _swag_like_dataset(questions, paragraphs, candidate_paragraph_ids, answers):
    assert len(questions) == len(candidate_paragraph_ids) 
    if not answers == None:
        assert len(candidate_paragraph_ids) == len(answers)
    dataset_split = dict()
    size = len(questions)
    dataset_split['sent1'] = questions
    dataset_split['sent2'] = ["" for i in range(size)]
    if answers == None:
        dataset_split['label'] = [0 for i in range(size)]
    else:
        dataset_split['label'] = [get_match_index(candidates, answer) for candidates, answer in zip(candidate_paragraph_ids, answers)]
    for i, ending_name in enumerate(ending_names):
        dataset_split[ending_name] = [ paragraphs[candidates[i]] for candidates in candidate_paragraph_ids]
    return dataset_split
