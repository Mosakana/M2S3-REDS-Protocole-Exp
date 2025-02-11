from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd

import torch

def tokenize_fn(example):
    prompt_ids = tokenizer.encode(example["prompt"], padding='max_length', max_length=900, truncation=True)
    
    target_ids = tokenizer.encode(example["target"], padding='max_length', max_length=100, truncation=True)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "labels": labels
    }

if __name__ == '__main__':

    with open('./1_train_str_prompt.txt', 'r', encoding='utf-8') as file:
        train_prompt = file.readlines()

    with open('./1_train_str_target.txt', 'r', encoding='utf-8') as file:
        train_target = file.readlines()

    with open('./1_eval_str_prompt.txt', 'r', encoding='utf-8') as file:
        eval_prompt = file.readlines()

    with open('./1_eval_str_target.txt', 'r', encoding='utf-8') as file:
        eval_target = file.readlines()

    clean_train_data = {'prompt': [doc.replace('\t', '\n')[:-2] for doc in train_prompt],
                        'target': [doc.replace('\t', '\n')[:-1] for doc in train_target]}

    clean_eval_data = {'prompt': [doc.replace('\t', '\n')[:-2] for doc in eval_prompt],
                        'target': [doc.replace('\t', '\n')[:-1] for doc in eval_target]}

    train_df = pd.DataFrame(clean_train_data)
    eval_df = pd.DataFrame(clean_eval_data)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = train_dataset.map(tokenize_fn, batched=False)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        save_steps=500,
        logging_steps=100,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    trainer.save_model('./model')

