from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from datasets import Dataset
from random import randint
from utils import TRAIN, DEV, TEST, SPLITS

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

class QA_Dataset(Dataset):
    def __init__(self, split, size, max_seq_len, max_question_len, doc_stride, answers, tokenized_questions, tokenized_paragraphs):
        self.mysplit = split
        self.answers = answers
        self.size = size

        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        
        self.max_question_len = max_question_len
        self.max_seq_len = max_seq_len
        assert self.max_question_len <= self.max_seq_len - 3
        self.max_paragraph_len = self.max_seq_len - 3 - self.max_question_len
        
        # self.max_question_len = 40
        # self.max_paragraph_len = 150
        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        # self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

        ##### TODO: Change value of doc_stride #####
        self.doc_stride = doc_stride

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        answer = self.answers[idx] if self.answers is not None else None
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[idx]
        
        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.mysplit == TRAIN:
            try:
                # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
                answer_start_token = tokenized_paragraph.char_to_token(answer["start"])
                answer_end_token = tokenized_paragraph.char_to_token(answer["start"] + len(answer["text"]) - 1)
                # print(f"idx: {idx}, answer[\"start\"]: {answer["start"]}, answer_start_token: {answer_start_token}, answer_end_token: {answer_end_token}")
                
                # A single window is obtained by slicing the portion of paragraph containing the answer
                # mid = (answer_start_token + answer_end_token) // 2
                # paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
                
                # if answer_start_token >  self.max_paragraph_len:
                #     # paragraph_start 不能從0開始
                #     paragraph_start = min(answer_start_token - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len)
                # else:
                #     paragraph_start = 0
                
                paragraph_start = max(0, answer_start_token - randint(0, self.max_paragraph_len - (answer_end_token - answer_start_token + 1)))

                paragraph_end = paragraph_start + self.max_paragraph_len
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
                input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
                
                # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
                answer_start_token += len(input_ids_question) - paragraph_start
                answer_end_token += len(input_ids_question) - paragraph_start
                
                # Pad sequence and obtain inputs to model 
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            except:
                print(f"idx: {idx}")
                print(f"answer span: {answer['start']} , {answer['start'] + len(answer['text']) - 1}")
                print(f"len(tokenized_paragraph): {len(tokenized_paragraph)}")
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask
