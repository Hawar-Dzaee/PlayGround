
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
               n_samples = 20,
               init_w_1 = torch.tensor(1.),
               init_w_2 = torch.tensor(1.)
               ):
    
    self.X1 = torch.tensor(X1).flatten()
    self.y  = torch.tensor(y).flatten()
    self.X2 = torch.tensor(X2) if X2 is not None else None
    self.bias = torch.tensor(bias) if bias is not None else None
    self.n_samples = n_samples
    self.init_w_1 = init_w_1
    self.init_w_2 = init_w_2

    self.ONE_FEATURE  = (self.X2 is None) and (self.bias is None)
    self.TWO_FEATURES = (self.X2 is not None) and (self.bias is None)
    self.BIAS = (self.X2 is None) and (self.bias is not None)



#----------------------------------------------------------------------------------------------------------------PLOT DATA
  def plot_data(self):
#---------------------------ONE FEATURES--------------------
    if self.ONE_FEATURE:
      plot = go.Scatter(
        x = self.X1,
        y = self.y,
        mode = 'markers',
        name = 'Data')

      layout = go.Layout(
              title='Single Feature Regression Plot',
              xaxis=dict(title='X1',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
              yaxis=dict(title='Y',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)') ,
              hovermode='closest')

  #-----------TWO FEATURES--------
    if self.TWO_FEATURES:
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
    if self.BIAS:
      plot = go.Scatter(
          x = self.X1,
          y = self.y,
          mode = 'markers')

      layout = go.Layout(
        title='Single Feature Regression Plot',
        xaxis=dict(title='X1',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
        yaxis=dict(title='Y',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
        hovermode='closest')



    # figure = go.Figure(data=[plot],layout=layout)
    # return figure
    return plot,layout
  

#----------------------------------------------------------------------------------------------------------------LOSS LANDSCAPE
  def landscape(self,cost_fn = nn.MSELoss(),coef_1 = 10,coef_2 =10):
    '''cost_fn : choosing the loss function.
       coef_1  : provide w1, so the landscape can capture the sweet spot, right of the sweet spot, and left of the sweet spot.
       coef_2  : provide w2, same logic as the above'''


    #-------------------------------------ONE FEATURES-----------------------------------
    if self.ONE_FEATURE:

      w1_range = torch.linspace(coef_1-10,coef_1+10,self.n_samples)
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

      layout = go.Layout(
              title='Loss Landscape',
              xaxis=dict(title='w',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
              yaxis=dict(title='Loss',zeroline = True,zerolinewidth = 2,zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
              hovermode='closest')

    
    #-------------------------------------TWO FEATURES-----------------------------------
    if self.TWO_FEATURES:
      w1_m,w2_m = torch.meshgrid(torch.linspace(coef_1-10,coef_1+10,self.n_samples ),torch.linspace(coef_2-10,coef_2+10,self.n_samples ),indexing='ij')
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
        name = 'Loss'
      )

      layout = go.Layout(scene = dict(
            dict( 
               xaxis = dict(
                  title = 'w1',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
                yaxis = dict(
                  title = 'w2',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)' ),
                zaxis = dict(
                  title = 'Loss',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)' )
                  )))

    # -------------------------------------ONE FEATURE AND BIAS-----------------------------------
    if self.BIAS:

      w1_m,b_m = torch.meshgrid(torch.linspace(coef_1-10,coef_1+10,self.n_samples ),torch.linspace(self.bias-10,self.bias+10,self.n_samples),indexing='ij')
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

      layout = go.Layout(
            dict( scene = dict(
               xaxis = dict(
                  title = 'w',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)'),
                yaxis = dict(
                  title = 'b',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)' ),
                zaxis = dict(
                  title = 'Loss',
                  zeroline = True,
                  zerolinewidth = 2,
                  zerolinecolor = 'rgba(205, 200, 193, 0.7)' )
                  )))

    figure = go.Figure(data=landscape,layout=layout)
    return figure
  
#----------------------------------------------------------------------------------------------------------------PLOT MODEL------------METHOD 3--------

  def fit_model(self):

      if self.ONE_FEATURE :
        plot_fit_model = go.Scatter(
          x = self.X1,
          y = self.X1 * self.init_w_1,
          mode = 'lines',
          line = dict(color='rgb(27,158,119)'),
          name = 'model'
        )

      elif self.TWO_FEATURES:
        X1_span = torch.linspace(self.X1.min().item(),self.X1.max().item(),100)
        X2_span = torch.linspace(self.X2.min().item(),self.X2.max().item(),100)

        X1_mesh,X2_mesh = torch.meshgrid(X1_span,X2_span,indexing='ij')
        y = (self.init_w_1 * X1_mesh) + (self.init_w_2 * X2_mesh)

        plot_fit_model = go.Surface(
          x = X1_mesh,
          y = X2_mesh,
          z = y ,
          name = 'model',
          colorscale = ['rgb(27,158,119)','rgb(27,158,119)'],
          showscale = False,

        )

      elif self.BIAS:
        plot_fit_model = go.Scatter(
          x = self.X1,
          y = (self.X1 * self.init_w_1) + self.bias,
          mode = 'lines',
          line = dict(color='rgb(27,158,119)'),
          name = 'model'
        )


      return plot_fit_model
  

#----------------------------------------------------------------------------------------------------------------CONTAINER------------METHOD 4--------

  def plot_container(self):
    data,data_layout = self.plot_data()
    fit_model = self.fit_model()
    
    figure = go.Figure(data=[data,fit_model],layout=data_layout)

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
    X, y,coef = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2, random_state=1, bias=0,coef=True)
    data = Data(X,y)
    fig_data = data.plot_container()
    fig_loss = data.landscape(cost_fn=selected_loss_fn,coef_1=coef)
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)

elif n_features == 'Two Features':
    X, y,coef = datasets.make_regression(n_features=2, n_samples=n_samples, noise=2, random_state=1, bias=0,coef=True)
    data = Data(X[:,0],y,X[:,1])
    fig_data = data.plot_container()
    fig_loss = data.landscape(cost_fn=selected_loss_fn,coef_1=coef[0],coef_2=coef[1])
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)

elif n_features == 'One Feature And A Bias': 
    X, y,coef = datasets.make_regression(n_features=1, n_samples=n_samples, noise=2, random_state=1, bias=5,coef=True)
    data = Data(X,y,bias=5)
    fig_data = data.plot_container()
    fig_loss = data.landscape(cost_fn=selected_loss_fn,coef_1=coef)
    with col1:
        st.plotly_chart(fig_data, use_container_width=True)
    with col2:
        st.plotly_chart(fig_loss, use_container_width=True)



