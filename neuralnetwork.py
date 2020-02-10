import numpy as np
class ThreeLayerNeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation_function):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = activation_function

        # initialize weights
        self.wih = np.random.normal(0.0, pow(self.hnodes, 0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, 0.5), (self.onodes, self.hnodes))



    def predict(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_output)
        final_outputs = self.activation_function(final_outputs)

        return final_outputs

    def train(self, input_list, targets):
        inputs = np.array(input_list, ndmin=2).T
        targets = np,array(targets, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_output)
        final_outputs = self.activation_function(final_outputs)

        # Calculate Error for hidedn and final
        final_error = targets - final_outputs
        hidden_error = np.dot(self.who.T, final_error)

        # Adjust the weights
        self.who += self.lr * np.dot(((final_error*final_outputs) * (1.0 - final_outputs)), np.transpose(hidden_inputs))
        self.wih += self.lr * np.dot(((hidden_error*hidden_output) * (1.0 - hidden_output)), np.transpose(inputs))






