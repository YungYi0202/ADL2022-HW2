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
        
        self.BOS = "Label_BOS"
        self.label_mapping[self.BOS]= 10
        self._idx2label[10] = "O"

        self.EOS = "Label_EOS"
        self.label_mapping[self.EOS]= 11
        self._idx2label[11] = "O"

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
        #   sample = ['tokens':List[Str] , 'tags': [...], 'id': ...]
        # tokens = [sample['tokens'] for sample in samples]
        tokens = []
        for sample in samples:
            tmp = sample['tokens']
            # Add BOS token
            tmp.insert(0, Vocab.BOS)
            # Add EOS token
            tmp.append(Vocab.EOS)
            tokens.append(tmp)
        
        encoded_tokens = torch.tensor(self.vocab.encode_batch(tokens))
        labels = []
        lengths = []
        for sample in samples:
            tmp = [ self.label_mapping[tag] for tag in sample['tags']]
            lengths.append(len(tmp))
            tmp.insert(0, self.label_mapping[self.BOS])
            tmp.append(self.label_mapping[self.EOS])
            labels.append(tmp)
        # labels.shape = [samples.size[0], seq_len]
        labels = pad_to_len(labels, encoded_tokens.shape[1], self.label_mapping[self.PAD])
        labels = torch.tensor(labels)
        
        return {
            'encoded_tokens': encoded_tokens,
            'labels': labels,
            'lengths': lengths
        }

    def collate_fn_test(self, samples: List[Dict]) -> Dict:
        tokens = []
        lengths = []
        for sample in samples:
            tmp = sample['tokens']
            lengths.append(len(tmp))
            # Add BOS token
            tmp.insert(0, Vocab.BOS)
            # Add EOS token
            tmp.append(Vocab.EOS)
            tokens.append(tmp)
        #tokens = [sample['tokens'] for sample in samples]
        #lengths = [len(token) for token in tokens]
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
