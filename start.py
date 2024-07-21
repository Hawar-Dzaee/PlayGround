import streamlit as st 
import torch 
from torch import nn 
import plotly.graph_objects as go 
from typing import Optional,Callable
from sklearn import datasets



def landscape(X1:torch.Tensor,
              y:torch.Tensor,
              X2:Optional[torch.Tensor]= None,
              bias = False,
              cost_fn = nn.MSELoss(),
              span = 10,
              n_samples = 300):

  # Type & size
  if not isinstance(X1,torch.Tensor):
    X1 = torch.tensor(X1)
  X1 = X1.view(-1,1)
  if not isinstance(y,torch.Tensor):
    y = torch.tensor(y)
  y = y.view(-1,1).float()


  cost_fn = cost_fn

  #-------------------------------------ONE FEATURES-----------------------------------
  if (X2 is None) and (bias is False):
    landscape = go.Scatter(
      x = torch.linspace(-span,span,n_samples),
      y = [cost_fn((X1*w),y).item() for w in torch.linspace(-span,span,n_samples)],
      mode = 'lines'
    )
  
  #-------------------------------------TWO FEATURES-----------------------------------
  if (X2 is not None) and (bias is False):
    if not isinstance(X2,torch.Tensor):
      X2 = torch.tensor(X2)
    X2 = X2.view(-1,1)


    w1_m,w2_m = torch.meshgrid(torch.linspace(-span,span,n_samples),torch.linspace(-span,span,n_samples),indexing='ij')
    w1_f,w2_f = w1_m.flatten() , w2_m.flatten()

    COST = []
    for w1,w2 in zip(w1_f,w2_f):
      pred = (w1*X1) + (w2*X2)
      cost = cost_fn(pred,y)
      COST.append(cost.item())
    COST = torch.tensor(COST).view(n_samples,n_samples)

    landscape = go.Surface(
      x = w1_m,
      y = w2_m,
      z = COST,
    )

  # -------------------------------------ONE FEATURE AND BIAS-----------------------------------
  if (X2 is  None) and (bias is True):


    w1_m,b_m = torch.meshgrid(torch.linspace(-span,span,n_samples),torch.linspace(-span,span,n_samples),indexing='ij')
    w1_f,b_f = w1_m.flatten() , b_m.flatten()

    COST = []
    for w1,b in zip(w1_f,b_f):
      pred = (w1*X1) + b
      cost = cost_fn(pred,y)
      COST.append(cost.item())
    COST = torch.tensor(COST).view(n_samples,n_samples)

    landscape = go.Surface(
      x = w1_m,
      y = b_m,
      z = COST,
    )


  return landscape




# ----------------------------
# streamlit 

st.set_page_config(layout='wide')
st.title("Loss Landscape")
st.write('By : Hawar Dzaee')


# X,y = datasets.make_regression(n_features=2,n_samples=20,noise=2,random_state=1,bias=40)
# Landscape = landscape(X1 = X[:,0],
#                       X2 = X[:,1], 
#                       y=y,
#                       # bias = True,
#                       cost_fn=nn.L1Loss(),
#                       n_samples=50,
#                       span=200
#                       )


st.sidebar.header("Feature Selection")
n_features = st.sidebar.radio("Number of features:", ['One Feature And No Bias', 'Two Features','One Feature And A Bias'])
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




# Generate data based on user selection
if n_features == 'One Feature And No Bias':
    X, y = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2, random_state=1, bias=0)
elif n_features == 'Two Features':
    X, y = datasets.make_regression(n_features=2, n_samples=n_samples, noise=2, random_state=1, bias=0)
else: 
    X, y = datasets.make_regression(n_features=2, n_samples=n_samples, noise=2, random_state=1, bias=30)




# Create landscape
if n_features == 'One Feature And No Bias':
    Landscape = landscape(
        X1=X,
        y=y,
        bias=False,
        cost_fn=selected_loss_fn,
        n_samples=50,
        span=200
    )
elif n_features == 'Two Features':
    Landscape = landscape(
        X1=X[:, 0],
        X2=X[:, 1],
        y=y,
        bias=False,
        cost_fn=selected_loss_fn,
        n_samples=50,
        span=200
    )

else : 
    Landscape = landscape(
      X1=X[:, 0],
      y=y,
      bias=True,
      cost_fn=selected_loss_fn,
      n_samples=50,
      span=200
    )
   







st.plotly_chart(go.Figure(data=[Landscape]))