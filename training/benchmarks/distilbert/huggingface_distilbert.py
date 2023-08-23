from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# >>> dataset['train']
# Dataset({
#     features: ['idx', 'sentence', 'label'],
#     num_rows: 67349
# })
dataset = load_dataset("sst2")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  logging_strategy = "epoch",
                                  logging_first_step = True,
                                  save_steps = 10000,
                                  do_eval = True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
import pdb; pdb.set_trace()
# print(trainer.evaluate())
trainer.train()
