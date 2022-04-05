import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import (
    AdamW, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup
)

from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm

from utils import read_data, get_ending_names, swag_like_dataset
from dataset import DataCollatorForMultipleChoice, QA_Dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import csv

TRAIN = "train"
DEV = "valid"
TEST = "test"
SPLITS = [TRAIN, DEV]

def multiple_choice(tokenizer, args, device, max_seq_length, splits, questions, candidate_paragraph_ids, relevant_paragraph_ids, paragraphs, ids):
    mc_model = AutoModelForMultipleChoice.from_pretrained(args.pretrained_model_name_or_path).to(device)
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
            output_dir=args.mc_ckpt_dir,
            evaluation_strategy="steps",
            eval_steps=args.check_val_every_step,
            learning_rate=args.mc_lr,
            per_device_train_batch_size=args.mc_batch_size,
            per_device_eval_batch_size=args.mc_batch_size * args.mc_accu_grad,
            num_train_epochs=args.mc_num_epoch,
            weight_decay=0.01,
            gradient_accumulation_steps=args.mc_accu_grad,
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

        metrics["train_samples"] = len(tokenized_mc_data[TRAIN])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if args.do_mc_test:
        all_predictions = dict()
        for split in splits:
            with open(args.mc_pred_dir / f"{split}.csv","a+") as f:
                print(f"Predicting split {split}...")
                predictions, _, _ = trainer.predict(test_dataset=tokenized_mc_data[split])
                preds = np.argmax(predictions, axis=1)
                all_predictions[split] = [ candidate_paragraph_ids[split][i][pred_label] for i, pred_label in enumerate(preds)]
                writer = csv.writer(f)
                for i, pred_label in enumerate(preds):
                    writer.writerow([ids[split][i], candidate_paragraph_ids[split][i][pred_label]])
        return all_predictions
    else:
        return None


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda:0"
    if args.resume_from_checkpoint is not None:
        args.pretrained_model_name_or_path = args.resume_from_checkpoint
    
    do_mc = args.do_mc_train or args.do_mc_test
    do_qa = args.do_qa_train or args.do_qa_test
    print(f"args.do_mc_train: {args.do_mc_train}")
    print(f"args.do_mc_test: {args.do_mc_test}")
    print(f"args.do_qa_train: {args.do_qa_train}")
    print(f"args.do_qa_test: {args.do_qa_test}")
    
    splits = SPLITS + [TEST] if args.do_mc_test or args.do_qa_test else SPLITS
    print(f"splits: {splits}")
        
    if args.fp16_training:
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
    print(f"device: {device}")
    print(f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}")
    print(f"resume_from_checkpoint: {args.resume_from_checkpoint}")

    # Models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    data_paths = {split: args.data_dir / f"{split}.json" for split in splits}
    data = {split: read_data(path) for split, path in data_paths.items()}
    paragraphs = read_data(args.data_dir / "context.json")  
    
    # Classfiy data
    questions = {split: [sample["question"] for sample in data[split]] for split in splits}
    candidate_paragraph_ids = {split: [sample["paragraphs"] for sample in data[split]] for split in splits} if do_mc else None
    relevant_paragraph_ids = {split: [sample["relevant"] for sample in data[split]] for split in SPLITS}
    ids = {split: [sample["id"] for sample in data[split]] for split in splits}
    answers = {split: [sample["answer"] for sample in data[split]] for split in SPLITS} if do_qa else None

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

    mc_predictions = None
    if do_mc:
        mc_predictions = multiple_choice(tokenizer, args, device, max_seq_length, splits, questions, candidate_paragraph_ids, relevant_paragraph_ids, paragraphs, ids)
    if do_qa:
        qa_model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path).to(device)
        # TODO: concat mispredicted result.
        wrong_predict_cnt = {split: 0 for split in SPLITS}
        
        def add_wrong_sample(split, i, pred_id):
            wrong_predict_cnt[split] += 1
            questions[split].append(questions[split][i])
            relevant_paragraph_ids[split].append(pred_id)
            # Default value when there is no answer in the paragragh.
            answers[split].append({"text": "","start": 0})
        
        if args.qa_train_with_mispredict:
            print("Adding wrongly predicted samples...")
        if mc_predictions == None:
            # Read from mc_pred_dir
            for split in splits:
                with open(args.mc_pred_dir / f"{split}.csv", 'r') as f:
                    rows = csv.reader(f)
                    if split == TEST:
                        relevant_paragraph_ids[split] = [int(row[1]) for row in rows]
                    elif args.qa_train_with_mispredict:
                        for i, (row, answer_id) in enumerate(zip(rows, relevant_paragraph_ids[split])):
                            pred_id = int(row[1])
                            if not pred_id == answer_id:
                                add_wrong_sample(split, i, pred_id)
        else:
            for split in splits:
                if split == TEST:
                    relevant_paragraph_ids[split] = [pred_id for pred_id in mc_predictions[split]]
                elif args.qa_train_with_mispredict:
                    for i, (pred_id, answer_id) in enumerate(zip(mc_predictions[split], relevant_paragraph_ids[split])):
                        if not pred_id == answer_id:
                            add_wrong_sample(split, i, pred_id)
        # Note: answers.keys() = [TRAIN, VALID]
        if TEST in splits:
            assert len(ids[TEST]) == len(relevant_paragraph_ids[TEST])

        # TODO: Check correctness

        tokenized_questions = {split: tokenizer(questions[split] , add_special_tokens=False) for split in splits}
        tokenized_paragraphs = tokenizer(paragraphs, add_special_tokens=False)

        # train_qa_set = QA_Dataset(TRAIN, max_seq_length, args.qa_max_question_len, args.qa_doc_stride, answers[split], relevant_paragraph_ids[split], tokenized_questions[split], tokenized_paragraphs)
        qa_sets = {split: QA_Dataset(str(split), max_seq_length, args.qa_max_question_len, args.qa_doc_stride, answers[split] if split in SPLITS else None, relevant_paragraph_ids[split], tokenized_questions[split], tokenized_paragraphs) for split in splits}

        # Note: Do NOT change batch size of dev_loader / test_loader !
        # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
        train_loader = DataLoader(qa_sets[TRAIN], batch_size=args.qa_batch_size, shuffle=True, pin_memory=True)
        dev_loader = DataLoader(qa_sets[DEV], batch_size=1, shuffle=False, pin_memory=True)
        test_loader = DataLoader(qa_sets[TEST], batch_size=1, shuffle=False, pin_memory=True) if args.do_qa_test else None

        def evaluate(data, output):
            ##### TODO: Postprocessing #####
            # There is a bug and room for improvement in postprocessing 
            # Hint: Open your prediction file to see what is wrong 
            
            answer = ''
            max_prob = float('-inf')
            num_of_windows = data[0].shape[1]
            
            for k in range(num_of_windows):
                # Obtain answer by choosing the most probable start position / end position
                start_prob, start_index = torch.max(output.start_logits[k], dim=0)
                end_prob, end_index = torch.max(output.end_logits[k], dim=0)
                
                # Probability of answer is calculated as sum of start_prob and end_prob
                prob = start_prob + end_prob
                
                if end_index < start_index:
                    tmp = end_index
                    end_index = start_index
                    start_index = tmp
                    
                # Replace answer if calculated probability is larger than previous windows
                if prob > max_prob:
                    max_prob = prob
                    # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                    answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
            
            # Remove spaces in answer (e.g. "大 金" --> "大金")
            return answer.replace(' ','')

        if args.do_qa_train:
            optimizer = AdamW(qa_model.parameters(), lr=args.qa_lr)
            total_steps = len(train_loader) * num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 2000, # Default value
                                                num_training_steps=total_steps)
            logging_step = 500
            qa_model.train()
            print("Start Training ...")
            
            for epoch in range(args.qa_num_epoch):
                step = 1
                train_loss = train_acc = 0.0
                
                for data in tqdm(train_loader):	
                    # Load all data into GPU
                    data = [i.to(device) for i in data]
                    
                    # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
                    # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
                    
                    # data = [input_ids, token_type_ids, attention_mask, answer_start_token, answer_end_token]

                    output = qa_model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

                    # Choose the most probable start position / end position
                    start_index = torch.argmax(output.start_logits, dim=1)
                    end_index = torch.argmax(output.end_logits, dim=1)
                    
                    # Prediction is correct only if both start_index and end_index are correct
                    train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
                    train_loss += output.loss
                    
                    output.loss.backward()
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    ##### TODO: Apply linear learning rate decay #####
                    
                    # Print training loss and accuracy over past logging step
                    if step % logging_step == 0:
                        print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                        train_loss = train_acc = 0

                print("Evaluating Dev Set ...")
                qa_model.eval()
                with torch.no_grad():
                    dev_acc = 0.0
                    for i, data in enumerate(tqdm(dev_loader)):
                        output = qa_model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                            attention_mask=data[2].squeeze(dim=0).to(device))
                        # prediction is correct only if answer text exactly matches
                        result = evaluate(data, output)
                        dev_acc += result == answers[DEV][i]["text"]
                        print(f"result: {result} answer: {answers[DEV][i]['text']}")
                    print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
                qa_model.train()

            # Save a model and its configuration file to the directory 「saved_model」 
            # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
            # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
            print("Saving Model ...")
            model_save_dir = args.qa_ckpt_dir 
            qa_model.save_pretrained(model_save_dir)

        if args.do_qa_test:
            print("Evaluating Test Set ...")

            results = []

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_loader)):
                    output = qa_model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                                attention_mask=data[2].squeeze(dim=0).to(device))
                    results.append(evaluate(data, output))

            result_file = args.qa_pred_dir / "test.csv"
            with open(result_file, 'w') as f:	
                f.write("id,answer\n")
                for i, result in enumerate(results):
                    # Replace commas in answers with empty strings (since csv is separated by comma)
                    # Answers in kaggle are processed in the same way
                    f.write(f"{ids[TEST][i]},{result.replace(',','')}\n")

            print(f"Completed! Result is in {result_file}")

    


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--mc_ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="/tmp2/b08902029/ADL/hw2/ckpt/",
    )
    parser.add_argument(
        "--qa_ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="/tmp2/b08902029/ADL/hw2/qa_ckpt/",
    )
    parser.add_argument(
        "--mc_pred_dir",
        type=Path,
        help="Directory to save the model file.",
        default="/tmp2/b08902029/ADL/hw2/mc_pred/",
    )
    parser.add_argument(
        "--qa_pred_dir",
        type=Path,
        help="Directory to save the model file.",
        default="/tmp2/b08902029/ADL/hw2/qa_pred/",
    )
    parser.add_argument("--max_len", type=int, default=512)
    
    parser.add_argument("--mc_experiment_number", type=int, default=0)
    parser.add_argument("--qa_experiment_number", type=int, default=0)
    # optimizer
    parser.add_argument("--mc_lr", type=float, default=3e-5)
    parser.add_argument("--qa_lr", type=float, default=3e-5)

    # mc
    parser.add_argument("--mc_batch_size", type=int, default=1)
    parser.add_argument("--mc_accu_grad", type=int, default=8)
    parser.add_argument("--check_val_every_step", type=int, default=1000)
    
    parser.add_argument("--num_workers", type=int, default=2)

    # qa
    parser.add_argument("--qa_batch_size", type=int, default=1)
    parser.add_argument("--qa_accu_grad", type=int, default=8)
    parser.add_argument("--qa_max_question_len", type=int, default=40)
    parser.add_argument("--qa_doc_stride", type=int, default=150)

    # training
    parser.add_argument("--mc_num_epoch", type=int, default=5)
    parser.add_argument("--qa_num_epoch", type=int, default=5)

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    parser.add_argument("--fp16_training", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="bert-base-chinese")

    # parser.add_argument("--do_mc_train", type=bool, default=False)
    # parser.add_argument("--do_mc_test", type=bool, default=False)
    # parser.add_argument("--do_qa_train", type=bool, default=True)
    # parser.add_argument("--do_qa_test", type=bool, default=False)
    
    parser.add_argument("--do_mc_train", action="store_true", help="Run or not.")
    parser.add_argument("--do_mc_test", action="store_true", help="Run or not.")
    parser.add_argument("--do_qa_train", action="store_true", help="Run or not.")
    parser.add_argument("--do_qa_test", action="store_true", help="Run or not.")
    parser.add_argument("--qa_train_with_mispredict", action="store_true", help="Run or not.")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    args.mc_ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.mc_ckpt_dir = args.mc_ckpt_dir / str(args.mc_experiment_number)
    args.mc_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    args.mc_pred_dir.mkdir(parents=True, exist_ok=True)
    args.mc_pred_dir = args.mc_pred_dir / str(args.mc_experiment_number)
    args.mc_pred_dir.mkdir(parents=True, exist_ok=True)

    args.qa_ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.qa_ckpt_dir = args.qa_ckpt_dir / str(args.qa_experiment_number)
    args.qa_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    args.qa_pred_dir.mkdir(parents=True, exist_ok=True)
    args.qa_pred_dir = args.qa_pred_dir / str(args.qa_experiment_number)
    args.qa_pred_dir.mkdir(parents=True, exist_ok=True)

    main(args)
