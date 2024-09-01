import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2

def solve(N, d):
  U, M1, M2 = initialise_input(N, d)

  '''
  Enter your code here for steps 1 to 6
  '''
  # Step 1
  X = np.zeros((N, d))
  Y = np.zeros((N, d))
  #   print(U)
  #   print(M1)
  #   print(M2)
  #   print(U[1].shape)
  #   print(U[1, :].reshape(1,-1))
  #   print("product")
  #   print((np.dot(U[1, :].reshape(1,-1),M1)).shape)


  for i in range(N):
      X[i] = np.dot(U[i, :].reshape(1,-1),M1)
      Y[i] = np.dot(U[i, :].reshape(1,-1),M2)

  # print("after dot product X: \n",X)
  # print("after dot product Y: \n",Y)

  # Step 2: Adding offset to X and Y
  offset = np.arange(1, N+1).reshape(-1, 1)
  # print("offset: \n",offset)
  X_offset = X + offset
  Y_offset = Y + offset
  # print("after adding offset x: \n",X_offset)
  # print("after adding offset y: \n",Y_offset)
  Z =np.dot(X_offset,Y_offset.T)
  # print(Z.shape)

  #Step 3: Sparisfying the matrix Z
  sparse_z = np.zeros(Z.shape)
  rows, cols = np.indices(Z.shape)

  sparse_z[rows==cols] = Z[rows==cols]

  sparse_z[(rows%2==cols%2)&(rows!=cols)] = Z[(rows%2==cols%2)&(rows!=cols)]
  #   print(sparse_z)
  #   return sparse_z

  # Step 4: Applying softmax function to each row of  matrix z

  softmax_z = np.zeros(sparse_z.shape)
  for i in range(N):
      softmax_z[i] = np.exp(sparse_z[i]) / np.sum(np.exp(sparse_z[i]))

  # print(softmax_z)
  # print([max(softmax_z[i]) for i in range(N)])

  # Step 5
  max_indices = np.argmax(softmax_z, axis=1)
  return max_indices

# N = 10
# d = 8
  
solve(N,d)
