from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len

import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.PAD = "PAD"
        self.label_mapping[self.PAD]= 9
        self._idx2label[9] = "O"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # for sample in samples
        #   sample = ['tokens': [...], 'tags': [...], 'id': ...]
        tokens = [sample['tokens'] for sample in samples]
        # encoded_tokens = torch.tensor(self.vocab.encode_batch(tokens))
        encoded_tokens = self.vocab.encode_batch(tokens)
        labels = []
        lengths = []
        for sample in samples:
            tmp = [ self.label_mapping[tag] for tag in sample['tags']]
            lengths.append(len(tmp))
            labels.append(tmp)
        # labels.shape = [samples.size[0], seq_len]
        # labels = pad_to_len(labels, encoded_tokens.shape[1], self.label_mapping[self.PAD])
        labels = pad_to_len(labels, len(encoded_tokens[0]), self.label_mapping[self.PAD])
        labels = torch.tensor(labels)
        
        return {
            'encoded_tokens': encoded_tokens,
            'labels': labels,
            'lengths': lengths
        }

    def collate_fn_test(self, samples: List[Dict]) -> Dict:
        tokens = [sample['tokens'] for sample in samples]
        lengths = [len(token) for token in tokens]
        encoded_tokens = torch.tensor(self.vocab.encode_batch(tokens))
        return {
            'encoded_tokens': encoded_tokens,
            'id': [sample['id'] for sample in samples],
            'lengths': lengths
        }
        raise NotImplementedError


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
