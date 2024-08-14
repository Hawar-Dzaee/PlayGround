import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Assume we have a model, data, loss function, and optimizer defined
# For this example, we'll create dummy versions
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize dummy data, model, loss function, and optimizer

X = torch.linspace(-3,3,(10))
y = X * 2
model = DummyModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)





# Initialize or load the model, loss function, and optimizer
if 'model' not in st.session_state:
    st.session_state.model = DummyModel()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = torch.optim.Adam(st.session_state.model.parameters(), lr=0.01)
if 'loss_fn' not in st.session_state:
    st.session_state.loss_fn = nn.MSELoss()

def run_epoch():
    model = st.session_state.model
    optimizer = st.session_state.optimizer
    loss_fn = st.session_state.loss_fn
    
    y_hat = model(X.view(-1, 1))
    loss = loss_fn(y_hat, y.view(-1, 1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

# Streamlit app
st.title("Epoch Runner")

if 'epoch_count' not in st.session_state:
    st.session_state['epoch_count'] = 0

if 'losses' not in st.session_state:
    st.session_state['losses'] = []

if st.button("Run Epoch"):
    loss = run_epoch()
    st.session_state['epoch_count'] += 1
    st.session_state['losses'].append(loss)
    
    st.write(f"Epoch {st.session_state['epoch_count']} completed.")
    st.write(f"Loss: {loss:.4f}")

if st.session_state.losses:
    st.line_chart(st.session_state['losses'])