import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
# from transformers import AdamW, BertTokenizerFast, BertForQuestionAnswering, BertForMultipleChoice
from transformers import (
    AdamW, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer,
    set_seed
)

from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm

from utils import same_seeds, read_data, get_ending_names, swag_like_dataset
from dataset import DataCollatorForMultipleChoice

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv

TRAIN = "train"
DEV = "valid"
TEST = "test"
SPLITS = [TRAIN, DEV]

def main(args):
    # same_seeds(args.seed)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"args.do_mc_train: {args.do_mc_train}")
    print(f"args.do_mc_test: {args.do_mc_test}")
    print(f"args.do_qa_train: {args.do_qa_train}")
    print(f"args.do_qa_test: {args.do_qa_test}")
    splits = SPLITS + [TEST] if args.do_mc_test or args.do_qa_test else SPLITS
    print(f"splits: {splits}")
        
    if args.fp16_training:
        # !pip install accelerate==0.2.0
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
    print(f"device: {device}")
    print(f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}")
    print(f"checkpoint: {args.resume_from_checkpoint}")

    # Models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    mc_model = AutoModelForMultipleChoice.from_pretrained(args.pretrained_model_name_or_path).to(device)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path).to(device)

    data_paths = {split: args.data_dir / f"{split}.json" for split in splits}
    data = {split: read_data(path) for split, path in data_paths.items()}

    # Classfiy data
    questions = {split: [sample["question"] for sample in data[split]] for split in splits}
    candidate_paragraph_ids = {split: [sample["paragraphs"] for sample in data[split]] for split in splits}
    relevant_paragraph_ids = {split: [sample["relevant"] for sample in data[split]] for split in SPLITS}
    paragraphs = read_data(args.data_dir / "context.json")    
    ids = {split: [sample["id"] for sample in data[split]] for split in splits}
    
    # Set max length of tokenized data
    if args.max_len is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_len > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_len}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_len, tokenizer.model_max_length)

    # Function needed to tokenize data
    ending_names = get_ending_names()
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_seq_length)
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    mc_data = swag_like_dataset(splits, questions, paragraphs, candidate_paragraph_ids, relevant_paragraph_ids)
    tokenized_mc_data = mc_data.map(preprocess_function, batched=True)

    # Delete the unnecessary columns.
    tokenized_mc_data = {split: tokenized_mc_data[split].remove_columns(ending_names + ['sent1', 'sent2']) for split in splits}

    # Metric
    def mc_compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    
    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epoch,
        weight_decay=0.01,
        gradient_accumulation_steps=args.accu_grad,
    )

    trainer = Trainer(
        model=mc_model,
        args=training_args,
        train_dataset=tokenized_mc_data[TRAIN],
        eval_dataset=tokenized_mc_data[DEV],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=mc_compute_metrics,
    )
    if args.do_mc_train:
        # =====Train=====
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # =====Valid=====
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_mc_test:
        for split in splits:
            with open(args.mc_pred_dir / f"{split}.csv","a+") as f:
                predictions, _, _ = trainer.predict(test_dataset=tokenized_mc_data[split])
                preds = np.argmax(predictions, axis=1)
                writer = csv.writer(f)
                for i, pred_label in enumerate(preds):
                    writer.writerow([ids[split][i], candidate_paragraph_ids[split][i][pred_label]])
        
    # TODO:

    
    


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--mc_pred_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./mc_pred/",
    )
    parser.add_argument("--max_len", type=int, default=384)
    
    parser.add_argument("--experiment_number", , type=int, default=0)
    # data
    parser.add_argument("--max_len", type=int, default=384)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accu_grad", type=int, default=4)
    
    parser.add_argument("--num_workers", type=int, default=2)

    # training
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--checkpoint_name", type=str, default='model.pt')

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    parser.add_argument("--fp16_training", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="bert-base-chinese")
    
    parser.add_argument("--do_mc_train", type=bool, default=False)
    parser.add_argument("--do_mc_test", type=bool, default=True)
    parser.add_argument("--do_qa_train", type=bool, default=True)
    parser.add_argument("--do_qa_test", type=bool, default=True)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir = args.ckpt_dir / args.experiment_number
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    args.mc_pred_dir.mkdir(parents=True, exist_ok=True)
    args.mc_pred_dir = args.mc_pred_dir / args.experiment_number
    args.mc_pred_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)