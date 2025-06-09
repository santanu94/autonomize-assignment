import streamlit as st
import torch

from model1 import CpGPredictor
from model2 import CpGPredictor2

torch.classes.__path__ = [] # added to get around some incompatibility error

# Load the models
MODEL1_VOCAB_SIZE = 5
MODEL2_VOCAB_SIZE = 6
EMBEDDING_DIM = 16
LSTM_HIDDEN = 64
LSTM_LAYER = 1

model1 = CpGPredictor(embedding_dim=EMBEDDING_DIM, hidden_dim=LSTM_HIDDEN, vocab_size=MODEL1_VOCAB_SIZE)
model2 = CpGPredictor2(embedding_dim=EMBEDDING_DIM, hidden_dim=LSTM_HIDDEN, num_layers=LSTM_LAYER, vocab_size=MODEL2_VOCAB_SIZE)

try:
    model1.load_state_dict(torch.load("model1.pth"))
    model1.eval()
    model1_loaded = True
except FileNotFoundError:
    st.error("Model 1 not found. Please train Model 1 first.")
    model1_loaded = False

try:
    model2.load_state_dict(torch.load("model2.pth"))
    model2.eval()
    model2_loaded = True
except FileNotFoundError:
    st.error("Model 2 not found. Please train Model 2 first.")
    model2_loaded = False


# Define the predict function within the Streamlit app script
def predict(model_instance, seq, lstm_model: bool):
    model_instance.eval()
    # Convert DNA sequence to integer sequence using the correct dna2int mapping
    # Assuming dna2int includes 'N', 'A', 'C', 'G', 'T' and 'pad' (0)
    # Make sure to handle 'N' if it's not included in your training data
    seq_int = [dna2int.get(char, dna2int['N']) for char in seq] # Use 'N' for unknown characters or handle as needed
    seq_tensor = torch.tensor([seq_int], dtype=torch.long)

    with torch.no_grad():
        if not lstm_model and model1_loaded:
            return model1.predict(seq)
        elif lstm_model and model2_loaded:
            lengths = torch.tensor([len(seq_int)]) # Use length of integer sequence
            logits = model_instance(seq_tensor, lengths)
            predicted_count = logits.squeeze(-1).item()
            return predicted_count, round(predicted_count)
        else:
            return "Model not loaded."


st.title("CpG Detector")

st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ("Model 1 (Fixed Length)", "Model 2 (Variable Length)"))

# Input text area for DNA sequence
st.write("Enter a DNA sequence:")
sequence_input = st.text_area("Sequence", height=100)

if st.button("Predict"):
    if sequence_input:
        if model_choice == "Model 1 (Fixed Length)":
            if model1_loaded:
                if len(sequence_input) >= 2 and len(sequence_input) <= 128:
                    prediction, sigmoid_output = model1.predict(sequence_input.upper())
                    st.write(f"Model 1 Prediction (rounded): {sigmoid_output}")
                    st.write(f"Model 1 Logits (sigmoid activated): {round(prediction)}")
                    st.write(f"Model 1 is trained to output a sequence of 0,1 for each two consecutive characters. len(logits) == len(input_seq) - 1")
                else:
                    import random
                    print("".join([random.choice("NACGT") for _ in range(128)]))
                    st.warning("Model 1 requires sequences of atleast length 2.")
            else:
                st.warning("Model 1 is not loaded.")
        elif model_choice == "Model 2 (Variable Length)":
            if model2_loaded:
                if len(sequence_input) >= 1 and len(sequence_input) <= 128:
                    prediction, rounded_prediction = model2.predict(sequence_input.upper())
                    st.write(f"Model 2 Prediction (rounded): {rounded_prediction}")
                    st.write(f"Model 2 Logits: {prediction:.4f}")
                    st.write(f"Model 2 is trained to output a real number count representing the number of 'CG' pairs")
                else:
                     st.warning(f"Model 2 requires sequences with length between {min_len} and {max_len}.")
            else:
                st.warning("Model 2 is not loaded.")
    else:
        st.warning("Please enter a DNA sequence.")
