from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import os
import torch
import pandas as pd
from transformers import TrainerCallback


# Load IMDb dataset for binary classification
def parse_mr(example):
    label = int(example['text'][0])  
    text = example['text'][2:].strip()  
    return {"text": text, "label": label}

def parse_yelp(example):
    label = int(example['label'])
    # Convert label: 1 -> 0, 2 -> 1
    label = 0 if label == 1 else 1
    text = example['text'].strip()
    return {"text": text, "label": label}

def parse_boolq(example):
    label = int(example['text'][0])   
    text = example['text'][2:].strip()  
    return {"text": text, "label": label}

def parse_mr_gsm8k_answer(example):
    label = int(example['text'][0])   
    text = example['text'][2:].strip()  
    return {"text": text, "label": label}

def parse_mr_gsm8k_solution(example):
    label = int(example['text'][0])   
    text = example['text'][2:].strip()  
    return {"text": text, "label": label}

# dataset = load_dataset("imdb")
def dataset_preprocess(data_name):

    if data_name == 'mr':
        dataset = load_dataset("text", data_files="./data/mr/train.txt")
        dataset = dataset.map(parse_mr)
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset, test_dataset
    
    elif data_name == 'boolq':
        dataset = load_dataset("text", data_files="./data/train_data/boolq/train.txt")
        dataset = dataset.map(parse_boolq)
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset, test_dataset
    
    elif data_name == 'mr-gsm8k_answer':
        dataset = load_dataset("text", data_files="./data/mr-gsm8k/answer_train.txt")
        dataset = dataset.map(parse_mr_gsm8k_answer)
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset, test_dataset

    elif data_name == 'mr-gsm8k_solution':
        dataset = load_dataset("text", data_files="./data/mr-gsm8k/solution_train.txt")
        dataset = dataset.map(parse_mr_gsm8k_solution)
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset, test_dataset
        
    elif data_name == 'yelp':
        dataset = pd.read_csv("./data/yelp/train.csv", header=None)
        dataset.columns = ["label", "text"]
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.map(parse_yelp)
        train_test_split = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        
        return train_dataset, test_dataset
    

# Tokenize the dataset
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

class SaveModelCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        output_dir = f"./models/{output_dir_name}/{data_name}_finetuned_model/epoch_{int(epoch)}"
        kwargs['model'].save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved at {output_dir}")

# Split into 80% train, 20% test
data_name = 'yelp'
# model_name = "yash3056/Llama-3.2-1B-imdb"
model_name = "yash3056/Llama-3.2-3B-imdb"
output_dir_name = "llama_3.2_3b"

# Separate into train and test
train_dataset, test_dataset = dataset_preprocess(data_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize training set
tokenized_train_dataset = train_dataset.map(preprocess, batched=True)
# Tokenize test set
tokenized_test_dataset = test_dataset.map(preprocess, batched=True)

# Load model for binary classification (num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./models/{output_dir_name}/{data_name}_finetuned_model",
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # learning_rate=3e-5,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    callbacks=[SaveModelCallback(tokenizer=tokenizer)]
)

# Fine-tune the model
trainer.train()

# Save model and tokenizer
model.save_pretrained(f"./models/{output_dir_name}/{data_name}_finetuned_model/final")
tokenizer.save_pretrained(f"./models/{output_dir_name}/{data_name}_finetuned_model/final")