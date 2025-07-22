import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# Importing the dataset
df=pd.read_csv('medquad.csv')
print(df[['question', 'answer']].head())

# Cleaning the dataset
def clean_text(text):
    text = str(text)
    text = text.encode('utf-8', 'ignore').decode()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9.,;:!?\'"()\- ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# Tokenizing the dataset
def tokenize(text):
    return text.lower().split()

df['question_tokens'] = df['question'].apply(tokenize)
df['answer_tokens'] = df['answer'].apply(tokenize)

# Creating a voacabuloary from the dataset
all_tokens = [token for tokens in df['question_tokens'] for token in tokens] + \
                [token for tokens in df['answer_tokens'] for token in tokens]

vocab_counter = Counter(all_tokens)
vocab = {token: idx+2 for idx, (token, _) in enumerate(vocab_counter.most_common())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1  # Adding special tokens for padding and unknown words}

# Converting tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

df['question_ids'] = df['question_tokens'].apply(lambda tokens: tokens_to_indices(tokens, vocab))
df['answer_ids'] = df['answer_tokens'].apply(lambda tokens: tokens_to_indices(tokens, vocab))

# Truncating or padding sequences

max_question_length = 64
max_answer_length = 32

def pad_sequence(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value] * (max_len - len(seq)) 

df['question_ids_padded'] = df['question_ids'].apply(lambda x: pad_sequence(x, max_question_length, vocab['<PAD>']))
df['answer_ids_padded'] = df['answer_ids'].apply(lambda x: pad_sequence(x, max_answer_length, vocab['<PAD>']))

# Splitting the dataset into training and testing sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv('train_processed.csv', index=False)
val_df.to_csv('val_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)

# Save vocab as a pickle file
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)