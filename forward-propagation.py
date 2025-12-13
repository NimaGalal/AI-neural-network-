import numpy as np

def relu(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    return 1/(1 + np.exp(-np.clip(Z,-500, 500)))

def softmax(Z):
    exponential_of_Z = np.exp(Z-np.max(Z,axis=0, keepdims=True))
    return exponential_of_Z /np.sum(exponential_of_Z, axis=0,keepdims=True)

def forward_propagation(X,parameters,activation_type="relu"):
    caches =[]
    A =X
    L =len(parameters) //2
    
    for l in range(1,L):
        Input_From_Previous_layer = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z = np.dot(W,Input_From_Previous_layer) + b
        
        if activation_type.lower() == "relu":
            A =relu(Z)
        else:
            A =sigmoid(Z)
        
        cache = (Input_From_Previous_layer, W, b, Z)
        caches.append(cache)
    
    W = parameters['W'+ str(L)]
    b = parameters['b'+ str(L)]
    Z = np.dot(W, A) + b
    AL = softmax(Z)
    
    cache = (A, W, b, Z)
    caches.append(cache)
    
    return AL, caches
