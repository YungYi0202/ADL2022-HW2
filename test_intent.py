import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from intent_dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn_test, num_workers=args.num_workers)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    model.eval()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    
    test_intents = []
    test_ids = []

    with torch.no_grad():
        for batch in tqdm_object:
            logits = model(batch['encoded_text'].to(device=args.device))
            intent_ids = logits.argmax(dim=-1)
            intents = [dataset.idx2label(intent_id.item()) for intent_id in intent_ids]
            test_intents.extend(intents)
            test_ids.extend(batch['id'])


    # TODO: write prediction to file (args.pred_file)
    with open(f"{str(args.pred_file)}", mode="w+") as f :
        f.write("id,intent\n")
        for i in range(len(test_intents)):
            f.write(f'{test_ids[i]},{test_intents[i]}\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
