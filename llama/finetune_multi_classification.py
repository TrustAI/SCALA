from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import os
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_per_process_memory_fraction(0.5)

def parse_ag(example):
    label = int(example['label'])
    label = label - 1
    text = example['text'].strip()
    return {"text": text, "label": label}


def dataset_preprocess(data_name):
    if data_name == 'ag':
        dataset = pd.read_csv("./data/ag/train.csv", header=None)
        dataset.columns = ["label", "source","text"]
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.map(parse_ag)

        # Split into train/test sets
        train_test_split = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        
        return train_dataset, test_dataset

# Tokenize the dataset
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)


# Split into 80% train, 20% test
data_name = 'ag'
# base_model = "yash3056/Llama-3.2-1B-imdb"
base_model = "yash3056/Llama-3.2-3B-imdb"
model_name = "llama-3.2-3b"

# Separate into train and test
train_dataset, test_dataset = dataset_preprocess(data_name)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize training set
tokenized_train_dataset = train_dataset.map(preprocess, batched=True)
# Tokenize test set
tokenized_test_dataset = test_dataset.map(preprocess, batched=True)

# Load model for multi-class classification (num_labels=4)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=4, ignore_mismatched_sizes=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./models/{model_name}/{data_name}_finetuned_model",
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # learning_rate=2e-5,
    # learning_rate=3e-5,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# Fine-tune the model
trainer.train()

# Save model and tokenizer
model.save_pretrained(f"./models/{model_name}/{data_name}_finetuned_model")
tokenizer.save_pretrained(f"./models/{model_name}/{data_name}_finetuned_model")