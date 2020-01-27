# Notes on AI

## Neural Networks

> "Inputs times weights plus bias. Activate" - Siraj Raval

Neural networks themselves have a pretty straightforward structure. In a nutshell, they work as **function approximator**, changing either their **weights** or their **bias** to get the desired result.

<p align="center">
	<img height="300" src="https://github.com/darroyolpz/AI-notes/blob/master/Images/NN.png?raw=true">
</p>

Given a huge set of data (aka *dataset*), neural networks are able to find the general patterns on the data, so that new predictions can be made. All this process is carried out using "simple" linear algebra operations, like matrix multiplication.

As matrices **inner indexes must match between each other** (i.e. [Ax**B**] and [**B**xC]), it's recommended to pass the inputs as a one-row vector (with as many columns as inputs) and weights as a matrix (matching the number of rows with the number of inputs, and number of columns with the amount of hidden layers).

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;h_1=x_1w_{11}+x_2w_{21}+x_3w_{31}"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;h_2=x_1w_{12}+x_2w_{22}+x_3w_{32}"></p>

In code one could use Numpy. Define a sigmoid function as activation and get the output:

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

hidden_inputs = np.dot(inputs, weights_to_hidden)
output = sigmoid(hidden_inputs)
```

## Gradient Descent

<p align="center">
	<img height="300" src="https://github.com/darroyolpz/AI-notes/blob/master/Images/NN_01.png?raw=true">
</p>

We need some measurements on how bad our predictions are, so easiest way to go is with the difference between the true target (y) and the network output (ŷ)

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;E=(y-\hat{y})"></p>

In order to not have negative errors, magnify the errors, count the entire dataset and do the math smooth, we'll define the error function as follows:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;E=\frac{1}{2}\sum_\mu(y^\mu-\hat{y}^\mu)^2"></p>

Only way to minimize the error function is through changing the weights & biases. Remember that the network output ŷ depends on the inputs and the weights, so error function is:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;E=\frac{1}{2}\sum_\mu(\hat{y}\mu-f(\sum_i w_i x_i^\mu))^2"></p>

If we update the weights in every pass of the predictions (epoch) we would finally get to the minimum weight. Best way to do so is by using the slope of the function, also call gradient. Step by step (thanks to the learning rate) weights will eventually converge to their minimum value.

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;\Delta{w_i}=-\eta\frac{\partial E}{\partial w_i}=-\eta\frac{\partial}{\partial w_i}\frac{1}{2}(y-\hat{y}(w_i))^2"></p>

Remember chain rule:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial}{\partial{z}}p(q(z))=\frac{\partial{p}}{\partial{q}}\frac{\partial{q}}{\partial{z}}"></p>

Taking into account:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;q=(y-\hat{y}(w_i))"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;p =\frac{1}{2}q(w_i)^2"></p>

We get:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial E}{\partial{w_i}}=(y-\hat{y})\frac{\partial}{\partial{w_i}}(y-\hat{y})=\frac{\partial}{\partial{w_i}}(y-\hat{y})^2=-(y-\hat{y})f'(h)x_i"></p>

We can then define an **output error term** (delta) and re-write the entire equation:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;\delta=(y-\hat{y})f'(h)"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;w_{i+1}=w_i+\eta\delta{x_i}"></p>

## Back-propagation

<p align="center">
	<img height="300" src="https://github.com/darroyolpz/AI-notes/blob/master/Images/Backprop_01.png?raw=true">
</p>

The same way we calculated the output error from the inputs to the outputs, one must also propagate that error across the different weights of the neural network.

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;\delta_j=(\sum w_{jk}\delta_k)f'(h_j)"></p>

One of the most convenient part of using a sigmoid function for neural networks is that its derivative it's pretty straightforward:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\Large&space;f'(h)= h(1-h)"></p>

```python
## Forward pass ##
hidden_input = np.dot(x, weights_input_hidden)
hidden_output = sigmoid(hidden_input)
output = sigmoid(np.dot(hidden_output, weights_hidden_output))

## Backward pass ##
error = y - output
output_error_term = error * output * (1 - output) # Error term

# Calculate the hidden layer's contribution to the error
hidden_error = np.dot(output_error_term, weights_hidden_output)

# Calculate the error term for the hidden layer
hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

#Update the change in weights
del_w_hidden_output += output_error_term * hidden_output
del_w_input_hidden += hidden_error_term * x[:, None]
```

## Pytorch

### Simple neural network

```python
import torch

# Sigmoid activation function
def activation(x):
	return 1/(1+torch.exp(-x))

# Generate some data
torch.manual_seed(7) # Random seeed
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))

# Calculate the output
# torch.mm is more strict than torch.matmul
h = torch.mm(features, weights.view(-1, 1)) + bias
y = activation(h)
```

### Network architectures

<p align="center">
	<img height="300" src="https://github.com/darroyolpz/AI-notes/blob/master/Images/mlp_mnist.png?raw=true">
</p>

```python
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.module):
	def __init__(self):
		super().__init__() # Mandatory as Pytorch needs to register all the layers we define
		
		img_height, img_width = 28, 28 # Sizes of the images
		hidden_units_1 = 128 # Number of hidden units at HL1
		hidden_units_2 = 64 # Number of hidden units at HL2
		output_units = 10 # Number of output units - one for each digit
		
    self.fc1 = nn.Linear(img_height*img_width, hidden_units_1)
		self.fc2 = nn.Linear(hidden_units_1, hidden_units_2)
    self.output = nn.Linear(hidden_units_2, output_units)
		self.dropout = nn.Dropout(p=0.2) # Dropouts

	def forward(self, x):
		x = x.view(x.shape[0], -1) # Make sure input tensor is flattened
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = F.softmax(self.output(x), dim=1) # No dropout here
		return x

model = Network()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
```