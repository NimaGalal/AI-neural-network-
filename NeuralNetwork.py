import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

CIFAR_PATH = "./cifar-10-batches-py"
MNIST_PATH = "./Minst-DataSet"
TEST_FILE = 'test_batch'
NUM_CLASSES = 10

def unpickle(file): 
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_all_cifar10_training_data_prepared(cifar_path):
    X_list, Y_list = [], []
    for i in range(1, 2):  # 20k train images
        file_path = os.path.join(cifar_path, f'data_batch_{i}')
        data_dict = unpickle(file_path)
        X_list.append(data_dict[b'data']) 
        Y_list.append(np.array(data_dict[b'labels']))

    X_all = np.concatenate(X_list, axis=0) 
    X_train = X_all.T / 255.0

    Y_all = np.concatenate(Y_list, axis=0)  #10K test images
    Train_data_number = Y_all.shape[0]
    Y_train_one_hot = np.zeros((NUM_CLASSES, Train_data_number))
    Y_train_one_hot[Y_all, np.arange(Train_data_number)] = 1
    Y_train = Y_train_one_hot

    return X_train, Y_train

def Load_cifar10_test_data(cifar_path, num_classes=10):
    file_path = os.path.join(cifar_path, TEST_FILE)
    test_dict = unpickle(file_path)

    X_test_raw = test_dict[b'data'] 
    Y_test_raw = np.array(test_dict[b'labels']) 

    X_test = X_test_raw.T / 255.0
    
    m_test = Y_test_raw.shape[0] 
    Y_test_one_hot = np.zeros((num_classes, m_test))
    Y_test_one_hot[Y_test_raw, np.arange(m_test)] = 1
    Y_test = Y_test_one_hot
    
    return X_test, Y_test


def read_mnist_ubyte_files(images_path, labels_path):
    with open(labels_path, 'rb') as file:
        magic, num_labels = np.fromfile(file, dtype='>i4', count=2) 
        labels = np.fromfile(file, dtype=np.uint8) 
        
    with open(images_path, 'rb') as file:
        magic, num_images, rows, cols = np.fromfile(file, dtype='>i4', count=4) 
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows * cols)

    X = images.T / 255.0 
    m = len(labels)
    Y_one_hot = np.zeros((10, m))
    Y_one_hot[labels, np.arange(m)] = 1

    return X, Y_one_hot, labels


def load_and_split_mnist_data(mnist_path):
    train_images_path = os.path.join(mnist_path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(mnist_path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(mnist_path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(mnist_path, 't10k-labels.idx1-ubyte')
    
    X_train, Y_train, _ = read_mnist_ubyte_files(train_images_path, train_labels_path) #(784, 60000)
    X_test, Y_test, _ = read_mnist_ubyte_files(test_images_path, test_labels_path) #(784, 10000)

    return X_train, Y_train, X_test, Y_test


def initialize_parameters(layer_dims, activation_type="relu"):
    parameters = {}
    L = len(layer_dims)

    if activation_type.lower() == "relu":
        initializer_factor = 2.0 
    elif activation_type.lower() == "sigmoid":
        initializer_factor = 1.0 
    else:
        initializer_factor = 2.0 

    for l in range(1, L):
        n_in = layer_dims[l-1]
        scale = np.sqrt(initializer_factor / n_in)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def relu(Z):
    return np.maximum(0,Z)

def relu_backward(Z):
    return np.where(Z > 0, 1, 0)

def sigmoid(Z):
    return 1/(1 + np.exp(-np.clip(Z,-500, 500)))

def sigmoid_backward(Z):
    s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    return s * (1 - s)

def softmax(Z):
    exponential_of_Z = np.exp(Z-np.max(Z,axis=0, keepdims=True))
    return exponential_of_Z /np.sum(exponential_of_Z, axis=0,keepdims=True)

def forward_propagation(X,parameters,activation_type="relu"):
    caches =[]
    A =X
    L =len(parameters) 
    
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


def Cross_Entropy_Cost_Function(A_L, Y):
    m = Y.shape[1] 
    epsilon = 1e-10 
    A_L = np.clip(A_L, epsilon, 1.0 - epsilon)
    cost = (-1 / m) * np.sum(Y * np.log(A_L))
    cost = np.squeeze(cost)
    return cost

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
            dZ = dA_prev * relu_backward(Z)
        elif activation_type.lower() == "sigmoid":
            dZ = dA_prev * sigmoid_backward(Z)
            
        grads["dW" + str(l + 1)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l + 1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 0:
            dA_prev = np.dot(W.T, dZ)
            
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    return parameters


def train_model(X_train, Y_train, X_test, Y_test, layer_dims, hidden_activation, learning_rate=0.007, epochs=100):
    parameters = initialize_parameters(layer_dims, hidden_activation)
    costs = []
    
    for epoch in range(epochs):
        A_L_Train, caches = forward_propagation(X_train, parameters, hidden_activation)
        cost_train = Cross_Entropy_Cost_Function(A_L_Train, Y_train)
        grads = backward_propagation(A_L_Train, Y_train, caches, hidden_activation)
        parameters = update_parameters(parameters, grads, learning_rate)
        A_L_Test, caches = forward_propagation(X_test, parameters, hidden_activation)
        cost_test = Cross_Entropy_Cost_Function(A_L_Test, Y_test)
        print(f"Epoch {epoch}/{epochs} - Accuracy: {calculate_accuracy(A_L_Train,Y_train):.2f}% - Loss: {cost_train:.4f} - Val_accuracy: {calculate_accuracy(A_L_Test,Y_test):.2f}% - Val_loss: {cost_test:.4f}")
        
    return parameters, costs


def calculate_accuracy(A_L, Y_true_one_hot):
    predictions = np.argmax(A_L, axis=0)
    Y_true_raw = np.argmax(Y_true_one_hot, axis=0)  
    accuracy = np.mean(predictions == Y_true_raw) * 100
    return accuracy

def predict(X, parameters, hidden_activation):
    A_L, _ = forward_propagation(X, parameters, hidden_activation)
    predictions = np.argmax(A_L, axis=0)   
    return predictions

def Train_Model_With_Picked_Dataset():
    dataset, dims, activation, epochs = get_user_inputs_and_setup()
    LEARNING_RATE = 0.007
    print("\n" + "="*70)
    print("🚀 Starting Training Process")
    print("="*70)
    print(f"Model Architecture: {dims}")
    print(f"Hidden Activation: {activation.upper()}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Total Epochs: {epochs}")
    print("-" * 70)
    print("Epoch | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss")
    print("="*70)
    if dataset == 'C':
        X_train, Y_train = load_all_cifar10_training_data_prepared(CIFAR_PATH)
        X_test, Y_test = Load_cifar10_test_data(CIFAR_PATH)
    elif dataset == 'M':
        X_train, Y_train , X_test, Y_test = load_and_split_mnist_data(MNIST_PATH)

    Y_test_raw_labels = np.argmax(Y_test, axis=0)
    parameters,_ = train_model(X_train, Y_train, X_test, Y_test,dims,activation,LEARNING_RATE,epochs)
    predictions = predict(X_test,parameters,activation)
    final_accuracy = np.mean(predictions == Y_test_raw_labels) * 100
    print("\n" + "="*70)
    print("✅ Training Finished. Displaying Results on Test Set.")
    print("="*70)
    print("🎯 Accuracy on Test Set: {:.2f}%".format(final_accuracy))
    print("-" * 70)
    print("Distribution of Predicted vs. Actual Labels (First 20 Samples):")
    print("-" * 70)
    
    print("Actual Labels:    ", Y_test_raw_labels[:20].tolist())
    print("Predicted Labels: ", predictions[:20].tolist())
    print("-" * 70)



def get_user_inputs_and_setup():
    
    print("="*70)
    print("✨ WELCOME TO YOUR OWN NEURAL NETWORK BUILDER (NumPy DNN) ✨")
    print("="*70)
    
    # 1. Dataset Selection
    while True:
        print("\n▶ 1. Select the Dataset for Training:")
        dataset_choice = input("    - Enter 'C' for CIFAR-10 (3072 features) or 'M' for MNIST (784 features): ").upper().strip()
        
        if dataset_choice == 'C':
            input_dim = 3072
            print("\nDataset: CIFAR-10 (Image Classification)")
            break
        elif dataset_choice == 'M':
            input_dim = 784
            print("\nDataset: MNIST (Handwritten Digit Recognition)")
            break
        else:
            print("Error, Invalid choice. Please enter 'C' or 'M'.")
            
    # 2. Number of Hidden Layers
    print("\n--- Network Architecture Setup ---")
    while True:
        try:
            num_hidden_layers = int(input("▶ 2. Enter the number of hidden layers (e.g., 2): "))
            if num_hidden_layers < 1:
                print("Error, Input must be a positive integer for the number of layers.")
            else:
                break
        except ValueError:
             print("Error, Input must be a valid integer.")
    
    # 3. Neurons per Hidden Layer
    user_hidden_layers = []
    print(f"\n▶ 3. Enter the number of neurons for each hidden layer:")

    for i in range(1, num_hidden_layers + 1): 
        while True:
            try:
                neurons = int(input(f"    - Neurons in Layer {i}: "))
                if neurons < 1:
                    print("Error, Input must be a positive integer for the number of neurons.")
                else:
                    user_hidden_layers.append(neurons)
                    break
            except ValueError:
                 print("Error, Input must be a valid integer.")

    # 4. Activation Function
    while True:
        activation = input("\n▶ 4. Choose the Activation Function for hidden layers (Relu / Sigmoid): ").lower().strip()
        if activation in ['relu', 'sigmoid']:
            break
        print("Error, Please choose either 'Relu' or 'Sigmoid'.")

    # 5. Training Epochs
    while True:
        try:
            epochs = int(input("\n▶ 5. Enter the number of training epochs (e.g., 100): "))
            if epochs < 1:
                 print("Error, Input must be a positive integer for the number of epochs.")
            else:
                break
        except ValueError:
             print("Error, Input must be a valid integer.")

    
    output_dim = 10
    layer_dims = [input_dim] + user_hidden_layers + [output_dim]
    
    print("\n" + "="*70)
    print(f"Final Network Architecture: {layer_dims}")
    print(f"Hidden Activation: {activation.upper()}")
    print(f"Training Epochs: {epochs}")
    print("="*70)
    
    return dataset_choice, layer_dims, activation, epochs


Train_Model_With_Picked_Dataset()