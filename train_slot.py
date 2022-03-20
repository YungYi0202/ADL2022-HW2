import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from slot_dataset import SeqClsDataset
from utils import Vocab
from model import SeqSlotClassifier

from tqdm.auto import tqdm
import os

import csv

from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

import random

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn, num_workers=args.num_workers)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqSlotClassifier(args.rnn_module, embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(device=args.device)
    # model = None

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = None

    # TRY: lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min",
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_loss = float("inf")
    best_acc = 0.0
    print(f'checkpoint_name: {args.checkpoint_name}')
    if args.enable_mask:
        print("Enable mask.")
    

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    valid_loss_per_epoch = []
    valid_acc_per_epoch = []

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()

        train_total = 0
        train_acc = 0
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        train_losses = []
        for batch in tqdm_object:
            # TODO: randomly mask it. Only when training
            batch_input = batch['encoded_tokens']
            lengths = batch['lengths']
            batch_size = len(lengths)

            if args.enable_mask:
                for i in range(batch_size):
                    if random.random() > 0.5:
                        mask_amt = int (lengths[i] * 0.2)
                        mask_unit = random.sample(range(0, lengths[i]), mask_amt)
                        for unit in mask_unit:
                            batch_input[i][unit] = vocab.unk_id
            batch_input = torch.tensor(batch_input, dtype=torch.int)
            logits = model(batch_input.to(device=args.device)) 
            # logits.shape = [batch_size, seq_len, num_classes]   

            target = batch['labels'].to(device=args.device)
            # target,shape = [batch_size, seq_len]

            # TODO: Do something to logits
            optimizer.zero_grad()
            loss = criterion(logits.view(-1, datasets[TRAIN].num_classes), target.view(-1))      
            loss.backward()
            optimizer.step()

            
            train_total += batch_size

            for i in range(batch_size):
                pred = logits[i].argmax(dim=-1).to("cpu")[:lengths[i]]
                label = target[i].to("cpu")[:lengths[i]]
                if (pred == label).sum().item() == lengths[i]:
                    train_acc += 1 
            
            train_losses.append(loss.item())
            tqdm_object.set_postfix(train_loss=loss.item(), train_acc=train_acc/train_total)
        
        train_loss = sum(train_losses) / len(train_losses)
        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc / train_total)

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()

        val_total = 0
        val_acc = 0
        tqdm_object = tqdm(dev_loader, total=len(dev_loader))

        valid_losses = []
        valid_pred_tag = []
        valid_true_tag = []
        with torch.no_grad():
            for batch in tqdm_object:
                batch['encoded_tokens'] = torch.tensor(batch['encoded_tokens'])
                logits = model(batch['encoded_tokens'].to(device=args.device)) 
                target = batch['labels'].to(device=args.device)

                loss = criterion(logits.view(-1, datasets[TRAIN].num_classes), target.view(-1))    

                lengths = batch['lengths']
                batch_size = len(lengths)
                val_total += batch_size

                for i in range(batch_size):
                    pred = logits[i].argmax(dim=-1).to("cpu")[:lengths[i]]
                    label = target[i].to("cpu")[:lengths[i]]
                    if args.enable_analysis:
                        valid_pred_tag.append([datasets[DEV].idx2label(x) for x in pred.tolist()])
                        valid_true_tag.append([datasets[DEV].idx2label(x) for x in label.tolist()])
                    correct_cnt = (pred == label).sum().item()
                    if correct_cnt == lengths[i]:
                        val_acc += 1 

                valid_losses.append(loss.item())
                tqdm_object.set_postfix(valid_loss=loss.item(), valid_acc=val_acc/val_total)

        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_loss_per_epoch.append(valid_loss)
        valid_acc_per_epoch.append(val_acc / val_total)

        lr_scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            if val_acc/val_total > best_acc:
                best_acc = val_acc/val_total
            print(f'Save model. epoch: {epoch} valid_loss: {valid_loss} valid_acc: {val_acc/val_total}')
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.checkpoint_name))
            if args.enable_analysis:
                valid_best_pred_tag = valid_pred_tag
                valid_best_true_tag = valid_true_tag

        elif val_acc/val_total > best_acc:
            best_acc = val_acc/val_total
            print(f'Save model. epoch: {epoch} valid_loss: {valid_loss} valid_acc: {val_acc/val_total}')
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.checkpoint_name))
            if args.enable_analysis:
                valid_best_pred_tag = valid_pred_tag
                valid_best_true_tag = valid_true_tag

    # TODO: Inference on test set
    print(f'checkpoint_name: {args.checkpoint_name}')
    if args.enable_analysis:
        with open("true.txt","w") as f:
            f.write(str(valid_best_true_tag))
        with open("pred.txt","w") as f:
            f.write(str(valid_best_pred_tag))
        print("classification_report")
        print(classification_report(valid_best_true_tag, valid_best_pred_tag, mode='strict', scheme=IOB2))
        with open(args.compare_file,"a+") as f:
            writer = csv.writer(f)
            writer.writerow([args.checkpoint_name])
            writer.writerow(train_loss_per_epoch)
            writer.writerow(train_acc_per_epoch)
            writer.writerow(valid_loss_per_epoch)
            writer.writerow(valid_acc_per_epoch)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    # module
    parser.add_argument("--rnn_module", type=str, default="LSTM")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--checkpoint_name", type=str, default='model.pt')

    parser.add_argument("--enable_analysis", type=bool, default=False)
    parser.add_argument("--compare_file", type=str, default="compare.csv")
    parser.add_argument("--enable_mask", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
