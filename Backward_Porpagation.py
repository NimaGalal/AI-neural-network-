import numpy as np

def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

def sigmoid_derivative(Z):
    s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    return s * (1 - s)

def backward_propagation(AL, Y, caches, activation_type="relu"):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    Y = Y.reshape(AL.shape)
    
    current_cache = caches[L-1]
    A_prev, W, b, Z = current_cache
    
    dZ = AL - Y
    
    grads["dW" + str(L)] = (1/m) * np.dot(dZ, A_prev.T)
    grads["db" + str(L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    dA_prev = np.dot(W.T, dZ)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        A_prev, W, b, Z = current_cache
        
        if activation_type.lower() == "relu":
            dZ = dA_prev * relu_derivative(Z)
        elif activation_type.lower() == "sigmoid":
            dZ = dA_prev * sigmoid_derivative(Z)
            
        grads["dW" + str(l + 1)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l + 1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 0:
            dA_prev = np.dot(W.T, dZ)
            
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters
