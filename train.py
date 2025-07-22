import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from transformer_model import Transformer


# Loading the processed data and vocabulary
train_df = pd.read_csv('train_processed.csv')
val_df = pd.read_csv('val_processed.csv')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Prepare the dataset and dataloader
class SummarizationDataset(Dataset):
    def __init__(self, df):
        self.encoder_input_ids = df['question_ids_padded'].apply(eval).tolist()
        self.decoder_input_ids = df['answer_ids_padded'].apply(eval).tolist()
    def __len__(self):
        return len(self.encoder_input_ids)
    def __getitem__(self, idx):
        return {
            'encoder_input_ids': torch.tensor(self.encoder_input_ids[idx], dtype=torch.long),
            'decoder_input_ids': torch.tensor(self.decoder_input_ids[idx], dtype=torch.long)
        }
    
train_dataset = SummarizationDataset(train_df)
val_dataset = SummarizationDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = TransformerModel(vocab_size=len(vocab), d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

# Training loop
for epoch in range(10):  # Adjust the number of epochs as needed
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        encoder_input = batch['encoder_input_ids']
        decoder_target = batch['decoder_input_ids']
        outputs = model(encoder_input, decoder_target[:, :-1])
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_target[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch['encoder_input_ids']
            decoder_target = batch['decoder_input_ids']
            outputs = model(encoder_input, decoder_target[:, :-1])
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_target[:, 1:].reshape(-1))
            val_loss += loss.item()
decoder_target[:, :-1].reshape(-1)
val_loss += loss.item()
print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}') 

# Saving the final model
torch.save(model.state_dict(), 'transformer_model.pt')