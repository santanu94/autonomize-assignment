import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from functools import partial

class CpGPredictor2(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size):
        super(CpGPredictor2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Linear layer to predict the count from the last hidden state
        self.linear = nn.Linear(hidden_dim, 1)
        
        # Alphabet helpers   
        alphabet = 'NACGT'
        dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
        int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
        dna2int.update({"pad": 0})
        int2dna.update({0: "<pad>"})

        self.intseq_to_dnaseq = partial(map, int2dna.get)
        self.dnaseq_to_intseq = partial(map, dna2int.get)


    def forward(self, x, lengths):
        embedded = self.embedding(x) # embedded shape: (batch_size, seq_len, embedding_dim)

        # Pack the padded sequences
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Use the hidden state from the last time step of the last layer
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We want the last layer's hidden state: hidden[-1, :, :]
        last_hidden_state = hidden[-1, :, :] # shape: (batch_size, hidden_dim)

        # Pass the last hidden state through the linear layer
        logits = self.linear(last_hidden_state) # shape: (batch_size, 1)

        # Squeeze the last dimension to get shape (batch_size,)
        logits = logits.squeeze(-1)

        return logits
    
    def predict(self, seq):
        seq_tensor = torch.tensor([list(self.dnaseq_to_intseq(seq))], dtype=torch.long)

        with torch.no_grad():
            lengths = torch.tensor([len(seq)])
            logits = self.forward(seq_tensor, lengths)
            # Squeeze the output and get the item
            predicted_count = logits.squeeze(-1).item()
            
            return predicted_count, round(predicted_count)
