# Drug Review Prediction using Transformers

This repository contains a Jupyter Notebook that demonstrates how to use the Transformers library to fine tune a model and predict drug reviews.

## Requirements

The code in this notebook requires the following Python packages:

- transformers
- datasets
- scikit-learn

You can install these with pip:

```bash
!pip install transformers[torch]
!pip install datasets
!pip install scikit-learn
```

## Overview

The project is focused on fine tuning to predict top 10 most common drug conditions based on user reviews. We will use both HuggingFace's TrainerAPI and PyTorch's training and evaluation loop. Specifically, we:

- Download a dataset from UCI Repository containing drug reviews.
- Preprocess the data to remove null values and other unnecessary information (such as removing HTML tags).
- Fine-tune a DistilBERT model on the preprocessed data:
  - Using HuggingFace's TrainerAPI with half precision (fp16)
  - Using PyTorch's training and evaluation loop with half precision (fp16) 
- Evaluate the model on a test dataset.

## How to Run

1. Clone the repository
2. Open the Jupyter Notebook
3. Run all cells

### Data Download and Preprocessing

The notebook will automatically download and preprocess the data. It uses the `datasets` library for this. The preprocessing steps involve:

- Filtering out reviews with missing conditions
- Selecting only the top 10 most common conditions to predict
- Removing HTML tags from the reviews

### Model Training and Comparison

Two different approaches are used for training the model:

1. Using the `Trainer` API from the `transformers` library.
2. Manually defining the training loop using PyTorch.

Both methods are explored and compared in terms of their performance and ease of use.

#### Code Snippets

**Model Training using Trainer API**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./trainer',
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    evaluation_strategy='steps',
    eval_steps=1000,
    fp16=True,
    logging_steps=1000,
    lr_scheduler_type='cosine',
    warmup_steps=0,

)

# Custom metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction):
  predictions, labels = eval_pred.predictions, eval_pred.label_ids

  predictions = np.argmax(predictions, axis=1)
  accuracy = accuracy_score(labels, predictions)
  balanced_accuracy = balanced_accuracy_score(labels, predictions)

  return {
      'accuracy': accuracy,
      'balanced_accuracy': balanced_accuracy,
  }

# Data Collator
from transformers.data.data_collator import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['eval'],
    compute_metrics = compute_metrics,
    data_collator=data_collator,
)

trainer.train()
```

**Model Training using PyTorch's Custom Loop**

```python
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.cuda.amp import autocast, GradScaler

# Set up configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

model.to(device)
model.train()

num_epochs = 3
num_steps = len(train_loader) * num_epochs
progress_bar = tqdm(range(num_steps))

# Initialize train metrics
train_loss = 0
train_num_samples = 0
train_steps = 0
eval_steps = 1000

# Train loop
scaler = GradScaler()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # fp16
        with autocast():
            output = model(**batch)
            loss = output.loss

        scaler.scale(loss).backward()
        train_loss += loss.item() * len(batch)
        train_num_samples += len(batch)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        train_steps += 1
        progress_bar.update(1)

        # eval loop
        if train_steps % eval_steps == 0:

            # Reset train metrics
            print(f'steps = {train_steps} train loss = {train_loss / train_num_samples}')
            train_loss = 0
            train_num_samples = 0

            # Initialize eval metrics
            eval_loss = 0
            eval_preds = []
            eval_trues = []
            eval_num_samples = 0

            model.eval()
            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    output = model(**eval_batch)
                    loss = output.loss

                    y_pred = output.logits.argmax(dim=-1).cpu()
                    y_true = eval_batch['labels'].cpu()

                    eval_preds.extend(y_pred)
                    eval_trues.extend(y_true)

                    eval_loss += loss.item() * len(eval_batch)
                    eval_num_samples += len(eval_batch)

            # Calculate eval metrics
            eval_loss /= eval_num_samples
            eval_acc = accuracy_score(eval_trues, eval_preds)
            eval_balanced_acc = balanced_accuracy_score(eval_trues, eval_preds)

            print(f'eval_loss = {eval_loss} // eval_acc = {eval_acc} // eval_balanced_acc = {eval_balanced_acc}')

            model.train()

```

## Evaluation Metrics and Results

Here are the evaluation metrics for both training methods:

### Hugging Face's Trainer API

- Training Duration: 26min 01sec
- Test Accuracy: 89.24%
- Test Balanced Accuracy: 83.46%

### Custom PyTorch Training Loop

- Training Duration: 31min 36sec
- Test Accuracy: 89.34%
- Test Balanced Accuracy: 83.72%

## Future Work

- Implement data augmentation techniques.
- Use other transformer models for comparison.
