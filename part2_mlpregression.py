import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

def loss_calculate(outputs, labels):
    sum = 0
    for i in range(len(outputs)):
        sum += torch.square(labels[i]-outputs[i])
    return sum/len(outputs)

def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 2 inputs, 32 hidden units, and 1 output unit
    The activation function of the hidden layer is sigmoid.
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    outputs = torch.from_numpy(np.zeros((len(input_data),1), dtype=np.float32))
    for n in range(len(input_data)):
        hidden_layer = torch.from_numpy(np.zeros((32,1), dtype=np.float32))
        hidden_layer=(torch.matmul(input_data[n],w1)+b1).sigmoid()
        outputs[n] = torch.matmul(hidden_layer,w2)+b2
    return outputs
    

# we load all training, validation, and test datasets for the regression task
train_dataset, train_label = pickle.load(open("data/part2_regression_train.data", "rb"))
validation_dataset, validation_label = pickle.load(open("data/part2_regression_validation.data", "rb"))
test_dataset, test_label = pickle.load(open("data/part2_regression_test.data", "rb"))


# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()

train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# w1 defines the parameters between the input layer and the hidden layer
w1 = torch.from_numpy(np.random.normal(0, 1, 64).astype(np.float32).reshape((2, 32))).requires_grad_(True)
# Here you are expected to initialize w1 via the Normal distribution (mean=0, std=1).
...
# b defines the bias parameters for the hidden layer
b1 = torch.from_numpy(np.random.normal(0, 1, 1 ).astype(np.float32).reshape((1,1))).requires_grad_(True)
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=1).
...
# w2 defines the parameters between the hidden layer and the output layer
w2 = torch.from_numpy(np.random.normal(0, 1, 32).astype(np.float32).reshape((32, 1))).requires_grad_(True)
# Here you are expected to initialize w2 via the Normal distribution (mean=0, std=1).
...
# and finally, b2 defines the bias parameters for the output layer
b2 = torch.from_numpy(np.random.normal(0, 1, 1 ).astype(np.float32).reshape((1,1))).requires_grad_(True)
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=1).
...

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# You are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.0001)


# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean squared error (MSE) loss over the validation dataset.
ITERATION = 1500
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration+1)
   
    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data   
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)
    # Here you are expected to calculate the MEAN squared error loss with respect to the network predictions and the training ground truth
    train_mse_loss = loss_calculate(train_predictions, train_label)
    
    train_loss_array.append(train_mse_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
    train_mse_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.  
    optimizer.step()
    with torch.no_grad():
        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)
        # Here you are expected to calculate the average/mean squared error loss for the validation datasets by using the validation dataset ground truth.
        validation_mse_loss = loss_calculate(validation_predictions, validation_label)
        validation_loss_array.append(validation_mse_loss.item())
    print("Iteration : %d - Train MSE Loss %.4f - Validation MSE Loss : %.2f" % (iteration+1, train_mse_loss.item(), validation_mse_loss.item()))

# after completing the training, we calculate our network's mean squared error score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # Here you are expected to calculate the network's MSE on the test dataset...
    test_loss = loss_calculate(test_predictions, test_label)
    print("Test MSE loss : %.4f" % test_loss.item())

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()





