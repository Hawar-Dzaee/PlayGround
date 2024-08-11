import numpy as np 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 
import torch 
from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import streamlit as st 




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
