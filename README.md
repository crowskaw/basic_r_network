# basic_r_network
A reinterpretation of @iamtrask's 3-layer Basic Python Network, but written in R
The original code can be found here: http://iamtrask.github.io/2015/07/12/basic-python-network/

The neural network looks at a training input 4x3 matrix of 0s and 1s

  0 0 1
  
  1 1 1
  
  1 0 1
  
  0 1 1
  

And the training output 4x1 matrix of results (the first column of the matrix)

  0
  
  1
  
  1
  
  0
  

Feeding forward through a simple 3-layer artifical neural network, the synaptic weights will be adjusted using gradiant descent until they are able to predict the output data from the input data.

The final synaptic weights could then be used to test the expected result of a new vector, such as

  1 0 0

and should return 1, or

  0 1 0

and should return 0.
