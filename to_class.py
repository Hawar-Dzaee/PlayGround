
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 

import torch 
from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from typing import Optional,Callable

import streamlit as st 



class Data:

  def __init__(self,X1:torch.Tensor,
               y :torch.Tensor,
               X2:Optional[torch.Tensor]= None,
               bias:Optional[torch.Tensor]= None,
               n_samples = 30):
    
    
    self.X1 = torch.tensor(X1).flatten()
    self.y  = torch.tensor(y).flatten()
    self.X2 = torch.tensor(X2) if X2 is not None else None
    self.bias = torch.tensor(bias) if bias is not None else None
    self.n_samples = n_samples



#----------------------------------------------------------------------------------------------------------------PLOT DATA
  def plot_data(self):
#---------------------------ONE FEATURES--------------------
    if (self.X2 is None ) and (self.bias is None):
      plot = go.Scatter(
        x = self.X1,
        y = self.y,
        mode = 'markers')

      layout = go.Layout(
              title='Single Feature Regression Plot',
              xaxis=dict(title='X1'),
              yaxis=dict(title='Y'),
              hovermode='closest')

  #-----------TWO FEATURES--------
    if (self.X2 is not None) and (self.bias is None):
      plot = go.Scatter3d(
        x = self.X1,
        y = self.X2,
        z = self.y,
        mode = 'markers')

      layout = go.Layout(
              title='Two Features Regression Plot',
              scene=dict(
                  xaxis_title='X1',
                  yaxis_title='X2',
                  zaxis_title='Y'
              ),hovermode='closest')

  
#-------------ONE FEATURE AND A BIAS------------------
    if (self.X2 is None) and (self.bias is not None):
      plot = go.Scatter(
          x = self.X1,
          y = self.X1 + self.bias,
          mode = 'markers')

      layout = go.Layout(
        title='Single Feature Regression Plot',
        xaxis=dict(title='X1'),
        yaxis=dict(title='Y'),
        hovermode='closest')



    figure = go.Figure(data=[plot],layout=layout)
    return figure
  

#----------------------------------------------------------------------------------------------------------------LOSS LANDSCAPE
  def landscape(self,
                bias = False,
                cost_fn = nn.MSELoss(),
                span = 10):


    #-------------------------------------ONE FEATURES-----------------------------------
    if (self.X2 is None) and (bias is False):
      w1_range = torch.linspace(span-10,span+10,self.n_samples)
      COST = []
      for w1 in w1_range:
         pred = self.X1*w1
         cost = cost_fn(pred,self.y)
         COST.append(cost.item())
      landscape = go.Scatter(
        x = w1_range,
        y = COST,
        mode = 'lines'
      )
    
    #-------------------------------------TWO FEATURES-----------------------------------
    if (self.X2 is not None) and (bias is False):
      w1_m,w2_m = torch.meshgrid(torch.linspace(span-10,span+10,self.n_samples ),torch.linspace(span-10,span+10,self.n_samples ),indexing='ij')
      w1_f,w2_f = w1_m.flatten() , w2_m.flatten()

      COST = []
      for w1,w2 in zip(w1_f,w2_f):
        pred = (w1*self.X1) + (w2*self.X2)
        cost = cost_fn(pred,self.y)
        COST.append(cost.item())
      COST = torch.tensor(COST).view(self.n_samples ,self.n_samples )

      landscape = go.Surface(
        x = w1_m,
        y = w2_m,
        z = COST,
      )

    # -------------------------------------ONE FEATURE AND BIAS-----------------------------------
    if (self.X2 is  None) and (bias is True):

      w1_m,b_m = torch.meshgrid(torch.linspace(span-10,span+10,self.n_samples ),torch.linspace(bias-10,bias+10,self.n_samples),indexing='ij')
      w1_f,b_f = w1_m.flatten() , b_m.flatten()

      COST = []
      for w1,b in zip(w1_f,b_f):
        pred = (w1*self.X1) + b
        cost = cost_fn(pred,self.y)
        COST.append(cost.item())
      COST = torch.tensor(COST).view(self.n_samples ,self.n_samples )

      landscape = go.Surface(
        x = w1_m,
        y = b_m,
        z = COST,
      )

    figure = go.Figure(data=landscape)
    return figure
  

#----------------------------------------------------------------------------------------
# streamlit
st.set_page_config(layout='wide')
st.title("Play Ground")
st.write('By : Hawar Dzaee')
#-----------------------------------------Widgets 
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
#--------------------------------------------------------


col1, col2 = st.columns(2)

if n_features == 'One Feature And No Bias':
    X, y = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2, random_state=1, bias=0)
    data = Data(X,y)
    fig_data = data.plot_data()
    fig_loss = data.landscape(cost_fn=selected_loss_fn)
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)

elif n_features == 'Two Features':
    X, y = datasets.make_regression(n_features=2, n_samples=n_samples, noise=2, random_state=1, bias=0)
    data = Data(X[:,0],y,X[:,1])
    fig_data = data.plot_data()
    fig_loss = data.landscape(cost_fn=selected_loss_fn)
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)

elif n_features == 'One Feature And A Bias': 
    X, y = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2, random_state=1, bias=5)
    data = Data(X,y,bias=5)
    fig_data = data.plot_data()
    fig_loss = data.landscape(bias=True,cost_fn=selected_loss_fn)
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)



