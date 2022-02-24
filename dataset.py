from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # Make use of vocab
        # for sample in samples
        #   sample = ['text': ..., 'intent': ..., 'id': ...]
        text = [sample['text'].split() for sample in samples]
        encoded_text = torch.tensor(self.vocab.encode_batch(text, self.max_len))
        label = torch.tensor([ self.label_mapping[ sample['intent'] ] for sample in samples])
        return {
            'encoded_text': encoded_text,
            'label': label,
            'id': [sample['id'] for sample in samples]
        }
        # raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]