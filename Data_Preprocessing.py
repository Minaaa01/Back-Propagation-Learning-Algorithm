import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import GUI

df = pd.read_csv('penguins.csv')
df.replace({'gender': {'male': 1, 'female': 0}}, inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['gender'] = df['gender'].astype(int)
# Spilt Data into Training and Testing Samples
train_c1 = df.iloc[:30]
test_c1 = df.iloc[30:50]
train_c2 = df.iloc[50:80]
test_c2 = df.iloc[80:100]
train_c3 = df.iloc[100:130]
test_c3 = df.iloc[130:]
# Concatenate The 3 Training Samples then Shuffle It
train_c1_c2_c3 = pd.concat([train_c1, train_c2, train_c3]).sample(frac=1)
# Concatenate The 3 Testing Samples
test_c1_c2_c3 = pd.concat([test_c1, test_c2, test_c3])
# Spilt Data to X_Train_Features
train_c1_c2_c3_X = train_c1_c2_c3.iloc[:, 1:]
# Spilt Data to X_Train_Labels
train_c1_c2_c3_Y = train_c1_c2_c3.iloc[:, 0]
# Spilt Data to X_Test_Features
test_c1_c2_c3_X = test_c1_c2_c3.iloc[:, 1:]
# Spilt Data to X_Test_Labels
test_c1_c2_c3_Y = test_c1_c2_c3.iloc[:, 0]
# Convert train_c1_c2_c3_X, train_c1_c2_c3_Y To arrays
X_train = np.array(train_c1_c2_c3_X)
Y_train = np.array(train_c1_c2_c3_Y)
# Convert test_c1_c2_c3_X, test_c1_c2_c3_Y To arrays
X_test = np.array(test_c1_c2_c3_X)
Y_test = np.array(test_c1_c2_c3_Y)
# Encode The Y_Train_Labels
label_encoder = LabelEncoder()
Y_train_integer_encoded = label_encoder.fit_transform(Y_train)
# Don't Shuffle The Data
Y_train_onehot_encoder = OneHotEncoder(sparse=False)
# Reshape the data so we can encode it
Y_train_integer_encoded = Y_train_integer_encoded.reshape(len(Y_train_integer_encoded), 1)
# Encode The Y_Train_Labels to binary
Y_train_onehot_encoder = Y_train_onehot_encoder.fit_transform(Y_train_integer_encoded)
# Encode The Y_Test_Labels
Y_test_integer_encoded = label_encoder.fit_transform(Y_test)
# Don't Shuffle The Data
Y_test_onehot_encoder = OneHotEncoder(sparse=False)
# Reshape the data so we can encode it
Y_test_integer_encoded = Y_test_integer_encoded.reshape(len(Y_test_integer_encoded), 1)
# Encode The Y_Test_Labels to binary
Y_test_onehot_encoder = Y_test_onehot_encoder.fit_transform(Y_test_integer_encoded)
# Normalization The X_Train_Features and X_Test_Features Using MinMaxScaler
scaler = MinMaxScaler()
X_train_ = scaler.fit_transform(X_train)
X_test_ = scaler.fit_transform(X_test)
# Global Dic For Calculated Error , weights and Net Value
calculated_error = {}
weights = {}
Net_Value_Initialization = {}


def Sigmoid_Derivative(x):
    return x * (1 - x)


def Hyperbolic_Tangent_Sigmoid_Derivative(x):
    return 1 - np.power(np.tanh(x), 2)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Hyperbolic_Tangent_Sigmoid(x):
    return np.tanh(x)


def Activation_Function(activation_function, x):
    if activation_function == 0:
        return Sigmoid(x)
    else:
        return Hyperbolic_Tangent_Sigmoid(x)


def Derivative_Activation_Function(activation_function, x):
    if activation_function == 0:
        return Sigmoid_Derivative(x)
    else:
        return Hyperbolic_Tangent_Sigmoid_Derivative(x)


def Weights_Initialization(num_hidden_layer, num_neurons_layer):
    # Add the three output class to the num_neurons_layer list
    num_neurons_layer.append(3)
    # loop through the hidden layer to Initialize the Weights between the input features and each neuron in each hidden layer
    for i in range(num_hidden_layer):
        # Initialize the Weights between the input features and the first hidden layer based on the number of neuron in this layer
        if i == 0:
            weights[i] = np.random.rand(5, num_neurons_layer[i])
        # Initialize the Weights between the next hidden layer and the previous one based on the number of neuron in these two hidden layer
        weights[i + 1] = np.random.rand(num_neurons_layer[i], num_neurons_layer[i + 1])
        # Initialize Net values to the number of neurons in this hidden layer
        Net_Value_Initialization[i] = np.zeros((num_neurons_layer[i], 1))
    # Initialize Net values to three output class
    Net_Value_Initialization[num_hidden_layer] = np.zeros((3, 1))
    # return the Initialization weights and Net Values
    return weights, Net_Value_Initialization


def Feed_Forward_Pass(X, weight, weights_inputs, num_hidden_layer, activation_function):
    # loop through the network
    for i in range(num_hidden_layer + 1):  # the last one will not be included
        # calculate the net value for the first neurons in the first hidden layer
        if i == 0:
            weights_inputs[0] = Activation_Function(activation_function,
                                                    np.dot(X, weight[0]))
        # calculate the net value for the other neurons in the other hidden layers and also to three output class
        else:
            # get the previous net values for the previous hidden layer
            Current_Net_Value = weights_inputs[i - 1]
            # calculate the net value for the current hidden layer
            weights_inputs[i] = Activation_Function(activation_function,
                                                    np.dot(Current_Net_Value, weight[i]))
    # return the calculated net values for the hidden layers and the 3 output classes
    return weights_inputs


def Back_Propagate_Pass(weights_inputs, weights_, num_hidden_layer, y, activation_function):
    # calculate the error for the 3 output classes
    calculated_error[num_hidden_layer] = (y - weights_inputs[
        num_hidden_layer]) * Derivative_Activation_Function(activation_function,
                                                            weights_inputs[num_hidden_layer])
    # go backward to calculate the error for all the hidden layers
    for i in reversed(range(num_hidden_layer)):
        # get the weights and errors for next one
        calculated_error[i] = np.dot(weights_[i + 1], calculated_error[i + 1])
        # compute the error for the neurons in this hidden
        calculated_error[i] = calculated_error[i] * np.transpose(
            Derivative_Activation_Function(activation_function,
                                           weights_inputs[i]))
    # return the calculated error
    return calculated_error


def Update_Weights(weight, num_hidden_layer, eta, error, X, Net_Value):
    # get the length of the error in the first hidden layer to use it later in the reshaped
    Error_L_0 = len(error[0])
    # update the weight between the input features and the first hidden layer
    weight[0] = weight[0] + eta * np.dot(X.reshape(5, 1), error[0].reshape(1, Error_L_0))
    for i in range(1, num_hidden_layer + 1):
        # update the Weights between the next hidden layer and the previous one based on the number of neuron in these two hidden layer
        Error_L = len(error[i])
        Net_Value_L = len(Net_Value[i - 1])
        # update the Weights between the next hidden layer and the previous one based on the number of neuron in these two hidden layer
        NetValue_Er_Mu = np.dot(error[i].reshape(Error_L, 1), Net_Value[i - 1].reshape(1, Net_Value_L))
        weight[i] = weight[i] + eta * np.transpose(NetValue_Er_Mu)


def Training_BackPropagation_Algorithm(X, Y, HL, eta, number_of_epochs, Activation_Function_):
    # use function Weights_Initialization to return the Initial_Weights, Initial_Net_Value
    Initial_Weights, Initial_Net_Value = Weights_Initialization(HL, list(map(int, GUI.Number_Of_Neurons_list[0])))
    # loop until reach the max number of epochs the user enter
    for i in range(number_of_epochs):
        # loop around all the train features
        for j in range(len(X)):
            # phase 1 : forward phase, get the predicted output -> net value
            Final_NValues = Feed_Forward_Pass(X[j], Initial_Weights, Initial_Net_Value, HL, Activation_Function_)
            # phase 2 : backwards phase, calculate error
            C_Errors = Back_Propagate_Pass(Final_NValues, Initial_Weights, HL, Y[j], Activation_Function_)
            # phase 3 : update weights
            Update_Weights(Initial_Weights, HL, eta, C_Errors, X[j], Final_NValues)
    # return the Initial_Weights, Initial_Net_Value after the training
    return Initial_Weights, Initial_Net_Value


def Get_TheDesired_Output(List):
    TheDesired_Output = 0
    # convert the 3 class label to list to go through it
    List = list(List)
    # loop through the list that contain the predicted values for the 3 classes
    for i in range(3):
        # Get the index for the max predicted value
        if List[i] > List[TheDesired_Output]:
            TheDesired_Output = i
    # return the max index for the max predicted value
    return TheDesired_Output


def Testing_BackPropagation_Algorithm(X, Y, F_Weights, F_Net_Values, HL, AF):
    # Number of Hits that the algorithms get the predicted label correct, Initial Value = 0
    Bingo = 0
    #  Create 3 * 3 Array Matrix with zero value Then will loop through this array to add the actual values
    confusion_matrix = np.zeros((3, 3))
    # Loop until the last one predicted
    for i in range(len(X)):
        # Get the Net values for the last 3 classes using the Final_weights and Final_Net_Value that calculated from Training_BackPropagation_Algorithm
        out = Feed_Forward_Pass(X[i], F_Weights, F_Net_Values, HL, AF)
        # Get the index for the max predicted value
        y = Get_TheDesired_Output(out[HL])
        # Get the index for the max Target value
        t = Get_TheDesired_Output(Y[i])
        # Compare Between The Predicted and The Actual Labels
        if y == t:
            # Calculate Number of True Positive
            confusion_matrix[y][y] += 1
            # add one if The _predict and the actual one is the same
            Bingo += 1
        else:
            confusion_matrix[y][t] += 1
    ACC = Bingo / float(len(Y)) * 100.0
    # return the confusion_matrix and the percentage accuracy
    return confusion_matrix, ACC


Final_weights, Final_Net_Value = Training_BackPropagation_Algorithm(X_train_, Y_train_onehot_encoder,
                                                                    int(GUI.Userinfo[0]), float(GUI.Userinfo[1]),
                                                                    int(GUI.Userinfo[2]), int(GUI.Userinfo[4]))
# print(f'Final_weights: {Final_weights}')
# print(f'Final_Net_Value: {Final_Net_Value}')
X_train_Confusion_Matrix, X_train_Accuracy = Testing_BackPropagation_Algorithm(X_train_, Y_train_onehot_encoder,
                                                                               Final_weights,
                                                                               Final_Net_Value,
                                                                               int(GUI.Userinfo[0]),
                                                                               int(GUI.Userinfo[4]))
print('Confusion Matrix For Training Samples:')
print(X_train_Confusion_Matrix)
print(f'Accuracy For Training Samples: {X_train_Accuracy}')
X_test_Confusion_Matrix, X_test_Accuracy = Testing_BackPropagation_Algorithm(X_test_, Y_test_onehot_encoder,
                                                                             Final_weights,
                                                                             Final_Net_Value, int(GUI.Userinfo[0]),
                                                                             int(GUI.Userinfo[4]))
print(50 * "*")
print('Confusion Matrix For Testing Samples:')
print(X_test_Confusion_Matrix)
print(f'Accuracy For Testing Samples: {X_test_Accuracy}')
