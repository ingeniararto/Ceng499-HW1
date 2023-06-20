import torch
import torch.nn as nn
import numpy as np
import pickle


def prediction_result(test_prediction):
    max = 0
    index = 0
    for i in range(10):
        if test_prediction[i] > max:
            max=test_prediction[i]
            index=i
    return index


def calculate_accuracy(test_predictions, labels):
    sum=0
    for i in range(len(test_predictions)):
        sum += prediction_result(test_predictions[i])==labels[i]
    accuracy = sum/len(test_predictions)
    return  accuracy

class MLPModel(nn.Module):
    def __init__(self, layer_num, num_of_networks, learning_rate, num_of_epochs, activation_func):
        super(MLPModel, self).__init__()
        self.layer_num = layer_num
        if layer_num==2:
            self.layer1 = nn.Linear(num_of_networks[0], num_of_networks[1])
            self.layer2 = nn.Linear(num_of_networks[1], num_of_networks[2])
        elif layer_num==1:
            self.layer1 = nn.Linear(num_of_networks[0], num_of_networks[1])
        if activation_func==1:
            self.activation_function = nn.LeakyReLU()
        else:
            self.activation_function = nn.Tanh()
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs

    def forward(self, x):
        if self.layer_num==2:
            hidden_layer_output = self.activation_function(self.layer1(x))
            output_layer = self.layer2(hidden_layer_output)
        else:
            output_layer = self.activation_function(self.layer1(x))
        return output_layer


layer_nums_arr = [2,1]
num_of_networks_arr = [[784,32,10],[784,10]] # in the first one there is one hidden layer, in the second one there is no hidden layer
learning_rates_arr = [0.001,0.0001]
num_of_epochs_arr = [150,250]
activation_funcs_arr = [1,2] # 1 means LeakyRelu, 2 means Tanh for the activation function

# best parameters
# layer_nums_arr = [1]
# num_of_networks_arr = [[784,10]] 
# learning_rates_arr = [0.001]
# num_of_epochs_arr = [150]
# activation_funcs_arr = [1]

x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)


loss_function = nn.CrossEntropyLoss()


# STARTING ITERATIONS
n=0
for k in range(len(layer_nums_arr)):
    for learning_rate in learning_rates_arr:
        for num_of_epochs in num_of_epochs_arr:
            for activation_func in activation_funcs_arr:
                
                accuracies=torch.from_numpy(np.zeros((10),dtype=float))
                print("layer_num",layer_nums_arr[k],",",num_of_networks_arr[k],"-","learning rate:",learning_rate,"-","epoch number:",num_of_epochs,"-","activation function:",activation_func)
                for i in range(10):
                    print("trial ", i+1)
                    nn_model = MLPModel(layer_nums_arr[k],num_of_networks_arr[k],learning_rate,num_of_epochs,activation_func)
                    optimizer = torch.optim.Adam(nn_model.parameters(), lr=nn_model.learning_rate)

                    soft_max_function = torch.nn.Softmax(dim=1)

                    ITERATION = nn_model.num_of_epochs

                    for iteration in range(1, ITERATION +1):

                        optimizer.zero_grad()
                        predictions = nn_model(x_train)

                        loss_value = loss_function(predictions, y_train)

                        loss_value.backward()
                        optimizer.step()

                        with torch.no_grad():
                            train_prediction = nn_model(x_train)
                            train_loss = loss_function(train_prediction, y_train)
                            predictions = nn_model(x_validation)
                            probability_score_values = soft_max_function(predictions)
                            validation_loss = loss_function(predictions, y_validation)
                        print("Iteration : %d Training Loss : %f - Validation Loss %f" % (iteration, train_loss.item(), validation_loss.item()))
                    
                
                    # torch.save(nn_model.state_dict(), open("mlp_model"+str(n)+".mdl", "wb"))
                    n+=1

                    with torch.no_grad():
                        predictions = nn_model(x_test)
                        test_loss = loss_function(predictions, y_test)
                        print("Test - Loss %.2f" % (test_loss))
                        test_predictions = soft_max_function(predictions)
                        accuracies[i]= calculate_accuracy(test_predictions,y_test)

                # Calculate accuracy for every configuration   
                average_accuracy = torch.sum(accuracies)/10
                std_accuracy = torch.std(accuracies)
                confidence_l = average_accuracy - 1.96*(std_accuracy)/np.sqrt(10)
                confidence_h = average_accuracy + 1.96*(std_accuracy)/np.sqrt(10)
                print("Confidence Interval: (", confidence_l, ",", confidence_h, ")")

               

