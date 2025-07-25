import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from transformer_model import Transformer

# 1. Load processed data and vocabulary
train_df = pd.read_csv('train_processed.csv')
val_df = pd.read_csv('val_processed.csv')
test_df = pd.read_csv('test_processed.csv')

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 2. Custom Dataset
class SummarizationDataset(Dataset):
    def __init__(self, df):
        self.questions = df['question_ids_padded'].apply(eval).tolist()
        self.answers = df['answer_ids_padded'].apply(eval).tolist()
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        return {
            'encoder_input_ids': torch.tensor(self.questions[idx], dtype=torch.long),
            'decoder_input_ids': torch.tensor(self.answers[idx], dtype=torch.long)
        }

# 3. DataLoader
BATCH_SIZE = 32
train_dataset = SummarizationDataset(train_df)
val_dataset = SummarizationDataset(val_df)
test_dataset = SummarizationDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Model, optimizer, loss
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    vocab_size=len(vocab),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

# 5. Training and validation loop
EPOCHS = 10
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        encoder_input = batch['encoder_input_ids'].to(device)
        decoder_target = batch['decoder_input_ids'].to(device)
        optimizer.zero_grad()
        outputs = model(encoder_input, decoder_target[:, :-1])
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_target[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch['encoder_input_ids'].to(device)
            decoder_target = batch['decoder_input_ids'].to(device)
            outputs = model(encoder_input, decoder_target[:, :-1])
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_target[:, 1:].reshape(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'transformer_model.pt')
        print("Best model saved.")

# 6. Test Evaluation
model.load_state_dict(torch.load('transformer_model.pt'))
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        encoder_input = batch['encoder_input_ids'].to(device)
        decoder_target = batch['decoder_input_ids'].to(device)
        outputs = model(encoder_input, decoder_target[:, :-1])
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_target[:, 1:].reshape(-1))
        test_loss += loss.item()
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")