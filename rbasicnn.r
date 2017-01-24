## Basic R Network
##
## This is based on the 'A Neural Network in 11 lines of Python (Part 1)'
## at http://iamtrask.github.io/2015/07/12/basic-python-network/
##
## This is a reinterpretation of the 3-Layer Neural Network example
## in the R language
##
## This code should not require any special packages
##========================================================

# Create sigmoid function and it's derivative for training
nonlin <- function(x, deriv = F){
  if(deriv == T){
    return(x * (1-x))
  }
  
  return(1 / (1 + exp(-x)))
}


# Create a 4x3 matrix of 0s and 1s to use as training data
X.string <- c(0,0,1, 1,1,1, 1,0,1, 0,1,1)
X <- matrix(X.string, nrow = 4, byrow = T)

# Create a 4x1 matrix of the values of the first column
# These will be the pattern the network is looking for
y <- matrix(c(0, 1, 1, 0), nrow = 4, byrow = T)

# Set RNG seed for consistency
set.seed(1)

# Create synaptic weights with means of 0
# The first layer of weights being 3x4, the second layer being 4x1
syn0 <- matrix(runif(12,-1,1), nrow = 3, byrow = T)
syn1 <- matrix(runif(4,-1,1), nrow = 4, byrow = T)

# Begin training data
for(i in 1:60000){
  
  # Feed forward through layers 0, 1, and 2
  l0 <- X
  l1 <- nonlin((l0 %*% syn0))
  l2 <- nonlin((l1 %*% syn1))
  
  # Error (how far off we missed the target)
  l2_error <- y - l2
  
  # Print error at landmarks
  if(i %% 10000 == 0){
    cat('Error:', mean(abs(l2_error)), '\n')
  }
  
  # What direction is the target value?
  # The derivative shows how much to change by
  l2_delta <- l2_error*nonlin(l2, deriv = T)
  
  # How much did the l1 weights contribute to the l2 error
  l1_error <- l2_delta %*% t(syn1)
  
  # What direction is the target l1 weights?
  l1_delta <- l1_error * nonlin(l1, deriv = T)
  
  # Adjust weights
  syn1 <- syn1 + (t(l1) %*% l2_delta)
  syn0 <- syn0 + (t(l0) %*% l1_delta)
}
