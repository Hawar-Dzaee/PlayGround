import numpy as np 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 
import torch 
from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import streamlit as st 


#TODO
# 1- standardize the data.
# 2- update the line after running each epoch.
# 3- create (ball & Diamond).
# 4- after the ball after running each epoch.
# -------------------------------------------

def Data_Plot_Single_Feature(X1,y):
  plot = go.Scatter(
    x = X1,
    y = y ,
    mode = 'markers',
    name = 'data '
  )
  return plot


def Model_Plot_Single_Feature(X1,W1):
  plot_2 = go.Scatter(
    x = X1, 
    y = W1*X1,
    mode = 'lines',
    name = 'Model'
  )
  return plot_2


def landscape(X1,W1,y,cost_fn = nn.MSELoss(),n_samples =20):
      X1 = torch.tensor(X1)
      W1 = torch.tensor(W1)
      y = torch.tensor(y)
      
      w1_range = torch.linspace(W1-10,W1+10,n_samples)
      COST = []
      for w1 in w1_range:
         pred = X1*w1
         cost = cost_fn(pred,y)
         COST.append(cost.item())

      plot = go.Scatter(
        x = w1_range,
        y = COST,
        mode = 'lines'
      )

      return plot


#-----------------------------------------------------STREAMLIT

st.set_page_config(layout='wide')
st.title("Play Ground")
st.write('By : Hawar Dzaee')
#-----------------------------------------Widgets 
st.sidebar.header("Feature Selection")
n_samples = st.sidebar.slider("Number of samples:", min_value=10, max_value=100, value=20, step=5)

# Dropdown for loss function
loss_function = st.sidebar.selectbox(
    "Loss function:",
    ["L1Loss", "MSELoss", "SmoothL1Loss"]
)

# Map string to actual loss function
loss_fn_map = {
    "L1Loss": nn.L1Loss(),
    "MSELoss": nn.MSELoss(),
    "SmoothL1Loss": nn.SmoothL1Loss()
}
selected_loss_fn = loss_fn_map[loss_function]

#----------------------------------------- 


X, y,coef = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2 ,random_state=1, bias=0,coef=True)
X = X.reshape(-1)


fig_data = go.Figure(data = [Data_Plot_Single_Feature(X,y),Model_Plot_Single_Feature(X,coef)])
fig_loss = go.Figure(data= [landscape(X,coef,y,cost_fn=selected_loss_fn)])

col1, col2 = st.columns(2)

with col1:
      st.plotly_chart(fig_data, use_container_width=True)
with col2:
      st.plotly_chart(fig_loss, use_container_width=True)


#---------------------------------------------------------------SESSION
X_tensor = torch.tensor(X,dtype=torch.float32).view(-1,1)
y_tensor = torch.tensor(y,dtype=torch.float32).view(-1,1)


model = nn.Linear(1,1,bias=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)



if 'model' not in st.session_state:
    st.session_state["model"] = model 
if 'optimizer' not in st.session_state:
    st.session_state["optimizer"] = torch.optim.Adam(st.session_state["model"].parameters(), lr=0.01)


def run_epoch():
    model = st.session_state["model"]
    optimizer = st.session_state["optimizer"]
    
    y_hat = model(X_tensor)
    loss = selected_loss_fn(y_hat, y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),model.weight[0].item()


st.title("Epoch Runner")

if 'epoch_count' not in st.session_state:
    st.session_state['epoch_count'] = 0

# if 'losses' not in st.session_state:    # This might come in handy later on. 
#     st.session_state['losses'] = []


if st.button("Run Epoch"):
    loss,weight = run_epoch()
    st.session_state['epoch_count'] += 1
    # st.session_state['losses'].append(loss)
    
    st.write(f"Epoch {st.session_state['epoch_count']} completed.")
    st.write(f"Loss: {loss:.4f}")
    st.write(f'Weight : {weight}')


st.write(st.session_state)