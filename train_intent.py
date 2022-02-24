import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

from tqdm.auto import tqdm

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn, num_workers=args.num_workers)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, True, datasets[TRAIN].num_classes).to(device=args.device)
    # model = None

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = None

    # TRY: lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode="min",
    # )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_loss = float("inf")

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()

        train_total = 0
        train_acc = 0
        tqdm_object = tqdm(train_loader, total=len(train_loader))

        for batch in tqdm_object:
            # logits.shape = [batch_size, num_classes] 
            logits = model(batch['encoded_text'].to(device=args.device))    
            # target,shape = [batch_size]
            target = batch['label'].to(device=args.device)
            
            # TODO: Do something to logits
            optimizer.zero_grad()
            loss = criterion(logits, target)    
            loss.backward()
            optimizer.step()

            train_total += batch['encoded_text'].size(0)
            train_acc += (logits.argmax(dim=-1).to("cpu") == target.to("cpu")).sum().item()

            tqdm_object.set_postfix(train_loss=loss.item(), train_acc=train_acc/train_total)
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()

        val_total = 0
        val_acc = 0
        tqdm_object = tqdm(dev_loader, total=len(dev_loader))

        valid_losses = []
        with torch.no_grad():
            for batch in tqdm_object:
                logits = model(batch['encoded_text'].to(device=args.device))
                target = batch['label'].to(device=args.device)

                loss = criterion(logits, target)

                val_total += batch['encoded_text'].size(0)
                val_acc += (logits.argmax(dim=-1).to("cpu") == target.to("cpu")).sum().item()

                valid_losses.append(loss.item())
                tqdm_object.set_postfix(valid_loss=loss.item(), valid_acc=val_acc/val_total)

        valid_loss = sum(valid_losses) / len(valid_losses)
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(f'Save model. epoch: {epoch} valid_loss: {valid_loss} valid_acc: {val_acc/val_total}')
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.checkpoint_name))

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--checkpoint_name", type=int, default='model.pt')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
