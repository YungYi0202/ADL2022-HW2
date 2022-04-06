import random
import torch
import numpy as np
import json
from datasets import Dataset, DatasetDict

TRAIN = "train"
DEV = "valid"
TEST = "test"
SPLITS = [TRAIN, DEV]

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

# ***** For MC ******
ending_names = [f"ending{i}" for i in range(4)]
def get_ending_names():
    return ending_names

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
# ***** End - For MC ******

# ***** For QA ******
def index_before_tokenize(tokens, start, end, max_seq_length):
    char_count, new_start, new_end = 0, max_seq_length, max_seq_length
    start_flag = 0
    end_flag = 0
        
    for i, token in enumerate(tokens):
        if token == '[UNK]' or token == '[CLS]' or token == '[SEP]':
            if i == start:
                new_start = char_count
            if i == end:
                new_end = char_count
            char_count += 1
        else:
            for c in token:
                if i == start and start_flag == 0:
                    #print(token)
                    new_start = char_count
                    start_flag = 1
                if i == end:
                    #print(token)
                    new_end = char_count
                    end_flag = 1
                if c != '#':
                    char_count += 1
    return new_start, new_end

def evaluate(data, output, tokenizer, device, max_seq_length, doc_stride, paragraph=None, paragraph_tokenized=None):
    ##### Postprocessing #####
    
    # Load all data into GPU
    data = [i.to(device) for i in data]

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    # index in the whole tokens (not just relative to window)
    entire_start_index = 0
    entire_end_index = 0
    
    for k in range(num_of_windows):
        #print('window',k)
        # Obtain answer by choosing the most probable start position / end position
        mask = data[1][0][k].bool() &  data[2][0][k].bool() # token type & attention mask
        
        masked_output_start = torch.masked_select(output.start_logits[k], mask)[:-1] # -1 is [SEP]
        start_prob, start_index = torch.max(masked_output_start, dim=0)
        #masked_output_end = torch.masked_select(output.end_logits[k], mask)[start_index:-1] # -1 is [SEP]
        masked_output_end = torch.masked_select(output.end_logits[k], mask)[:-1] # -1 is [SEP]
        end_prob, end_index = torch.max(masked_output_end, dim=0)
        #end_index += start_index 
        

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        masked_data = torch.masked_select(data[0][0][k], mask)[:-1] # -1 is [SEP]

        # Replace answer if calculated probability is larger than previous windows
        if (prob > max_prob) and (end_index - start_index <= 40) and (end_index > start_index):
            max_prob = prob
            entire_start_index = start_index.item() + doc_stride * k
            entire_end_index = end_index.item() + doc_stride * k
            #print('entire_start_index',entire_start_index)
            #print('entire_end_index',entire_end_index)
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(masked_data[start_index : end_index + 1])
            # Remove spaces in answer (e.g. "大 金" --> "大金")
            answer = answer.replace('✔', ' ').replace('✦','\u200b').replace('☺','\u200e').replace('☆','\u3000').replace('●','#').replace(' ','')

    
    # if [UNK] in prediction, use orignal span of paragrah
    if '[UNK]' in answer:
        print('found [UNK] in prediction, using original text')
        print('original prediction', answer)
        # find the index of answer in the orinal paragrah

        new_start, new_end = index_before_tokenize(tokens=paragraph_tokenized, 
                                                start=entire_start_index, end=entire_end_index, max_seq_length=max_seq_length)
        #print('new_start',new_start)
        #print('new_end',new_end)
        answer = paragraph[new_start:new_end+1]
        answer = answer.replace('✔', ' ').replace('✦','\u200b').replace('☺','\u200e').replace('☆','\u3000').replace('●','#')
        print('final prediction:',answer)
    
    return answer
# ***** End - For QA ******