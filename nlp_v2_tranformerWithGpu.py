# %% [markdown]
# # Install
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers datasets scikit-learn matplotlib seaborn tqdm pandas
# 
# Install PyTorch with CUDA support
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 
# Install other libraries
# pip install transformers datasets scikit-learn matplotlib seaborn tqdm pandas
# 

# %% [markdown]
# # start

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# %%
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # Print if using CUDA (GPU) or CPU

# %%
# %% Load Data
df = pd.read_csv("data/text.csv")


# %% [markdown]
# # Visualize Data

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Emotion Distribution')
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# # Data Preparation

# %%
X = df['text'].tolist()
y = df['label'].tolist()

# %%
# Label encoding to numeric for BERT (if not already done)
unique_labels = sorted(set(y))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
y = [label2id[label] for label in y]

# %% [markdown]
# # Dataset Preparation for PyTorch

# %%
# Define a custom dataset class for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text using the tokenizer provided by the transformers library
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# %%
# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_labels)).to(device)

# %% [markdown]
# # K-Fold Cross-Validation and Training

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=0)
accuracy_scores = []
classification_reports = []
confusion_matrices = []

# %%
for train_index, test_index in kf.split(X):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # Create PyTorch Datasets
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_len=128)
    test_dataset = TextClassificationDataset(X_test, y_test, tokenizer, max_len=128)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3  # 3 epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(3):  # 3 epochs
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / len(train_loader)})

        print("epoch : ",epoch)
    print("end KFold")
    # Evaluation loop
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(batch['labels'].cpu().numpy())

    # Calculate accuracy, classification report, and confusion matrix
    accuracy_scores.append(accuracy_score(y_true, y_pred))
    classification_reports.append(classification_report(y_true, y_pred, target_names=unique_labels, output_dict=True))
    confusion_matrices.append(confusion_matrix(y_true, y_pred))

# %% [markdown]
# # Model Evaluation

# %%
# Print average accuracy
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"\nAverage Accuracy across folds: {avg_accuracy:.4f}")

# Print classification reports and confusion matrices for each fold
for i, report in enumerate(classification_reports):
    print(f"\nClassification Report for Fold {i+1}:\n")
    print(pd.DataFrame(report).transpose())

for i, matrix in enumerate(confusion_matrices):
    print(f"\nConfusion Matrix for Fold {i+1}:\n")
    print(matrix)


