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
import Numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

hidden_inputs = np.dot(inputs, weights_to_hidden)
output = sigmoid(hidden_inputs)
```

