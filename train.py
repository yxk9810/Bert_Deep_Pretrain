"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/23 13:39
@Email : handong_xu@163.com
"""
import warnings
import os
from utils.utils import *
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

warnings.filterwarnings('ignore')

def read_file(file):
    df = pd.read_csv(file)
    text = df['content']
    with open(DATA_DIR+'/train.txt','w',encoding='utf-8') as ft:
        ft.write('\n'.join(text.tolist()))
    return text

model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR+'/bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR+'/bert-base-chinese')
tokenizer.save_pretrained(os.path.join(MODEL_DIR,'deep_train_bert_base_chinese'))

train_path = DATA_DIR+'/train.txt'
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size = 256
)
valid_dataset = LineByLineTextDataset(
    tokenizer,
    file_path=train_path,
    block_size=256
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,mlm=True,mlm_probability=0.15
)
training_args = TrainingArguments(
    output_dir="./ckpt/deep_train_bert_base_chinese",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,# 轮数不要设置的太大
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(f'./paper_roberta_base')




if __name__ == '__main__':
    filePath = os.path.join(DATA_DIR,'unlabeled_data.csv')
    read_file(filePath)
