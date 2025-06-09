import torch
import torch.nn as nn
from functools import partial

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_lstm_layer = nn.Linear(embedding_dim * 2, 1)
        
        # Alphabet helpers   
        alphabet = 'NACGT'
        dna2int = { a: i for a, i in zip(alphabet, range(5))}
        int2dna = { i: a for a, i in zip(alphabet, range(5))}

        self.intseq_to_dnaseq = partial(map, int2dna.get)
        self.dnaseq_to_intseq = partial(map, dna2int.get)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) # embedded shape: (batch_size, seq_len, embedding_dim)

        # Prepare input for the linear layer: concatenate embeddings of prev and curr tokens
        # We need to process pairs of tokens (x[i], x[i+1])
        # This means the output will have length seq_len - 1
        batch_size, seq_len, embedding_dim = embedded.shape
        logits = []
        
        for i in range(1, seq_len):
            # Get the embeddings of the current and previous tokens
            curr_emb = embedded[:, i, :] # curr_emb shape: (batch_size, embedding_dim)
            prev_emb = embedded[:, i-1, :] # prev_emb shape: (batch_size, embedding_dim)

            pair_input = torch.cat([prev_emb, curr_emb], dim=1) # pair_input shape: (batch_size, 2 * embedding_dim)
            lstm_out = self.linear_lstm_layer(pair_input) # lstm_out shape: (batch_size, 1)
            logits.append(lstm_out)

        logits = torch.stack(logits, dim=1) # logits shape: (batch_size, seq_len - 1, 1)
        return logits.squeeze(-1)
    
    def predict(self, seq):
        seq_tensor = torch.tensor([list(self.dnaseq_to_intseq(seq))], dtype=torch.long)

        with torch.no_grad():
            logits = self.forward(seq_tensor)
            # Apply sigmoid and sum to get the predicted count
            sigmoid_outputs = torch.sigmoid(logits)
            rounded_sigmoid_outputs = torch.round(sigmoid_outputs)
            predicted_count = torch.sum(sigmoid_outputs, dim=1).item()
            return predicted_count, rounded_sigmoid_outputs
