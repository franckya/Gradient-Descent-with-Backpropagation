import numpy as np
import pandas as pd
from data_prep import features, targets, features_test, targets_test

def preprocess_data(features):
    """
    Convert categorical data to numeric and handle non-numeric data.
    """
    # Convert categorical columns using one-hot encoding
    features = pd.get_dummies(features)

    # Convert all columns to float, coerce errors to NaN and replace them with zero
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features.fillna(0, inplace=True)

    return features

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def forward_pass(x, weights_input_to_hidden, weights_hidden_to_output):
    """
    Make a forward pass through the network
    """
    x = np.array(x, dtype=np.float64)  # Ensure input is float64
    hidden_layer_input = np.dot(x, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

def backward_pass(x, target, learnrate, hidden_layer_output, output_layer_output, weights_hidden_to_output):
    """
    Make a backward pass through the network
    """
    error = target - output_layer_output
    output_error_term = error * output_layer_output * (1 - output_layer_output)
    hidden_error = np.dot(output_error_term, weights_hidden_to_output.T)
    hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)

    delta_w_h_o = learnrate * np.dot(hidden_layer_output.T, output_error_term)
    delta_w_i_h = learnrate * np.dot(x[:, None], hidden_error_term.T)
    return delta_w_h_o, delta_w_i_h

def update_weights(weights_input_to_hidden, weights_hidden_to_output, 
                   features, targets, learnrate):
    """
    Complete a single epoch of gradient descent and return updated weights
    """
    delta_w_i_h = np.zeros(weights_input_to_hidden.shape)
    delta_w_h_o = np.zeros(weights_hidden_to_output.shape)

    # Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        hidden_layer_out, output_layer_out = forward_pass(x,
            weights_input_to_hidden, weights_hidden_to_output)

        ## Backward pass ##
        delta_w_h_o_tmp, delta_w_i_h_tmp = backward_pass(x, y, learnrate,
            hidden_layer_out, output_layer_out, weights_hidden_to_output)
        delta_w_h_o += delta_w_h_o_tmp
        delta_w_i_h += delta_w_i_h_tmp

    n_records = features.shape[0]
    weights_input_to_hidden += delta_w_i_h / n_records
    weights_hidden_to_output += delta_w_h_o / n_records

    return weights_input_to_hidden, weights_hidden_to_output

def gradient_descent(features, targets, epochs=2000, learnrate=0.5):
    """
    Perform the complete gradient descent process on a given dataset
    """
    np.random.seed(11)
    features = preprocess_data(features)  # Preprocess features to ensure all numeric
    n_features = features.shape[1]
    n_hidden = 2
    weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden)).astype(np.float64)
    weights_hidden_output = np.random.normal(scale=1 / n_hidden ** .5, size=(n_hidden, 1)).astype(np.float64)

    for e in range(epochs):
        weights_input_hidden, weights_hidden_output = update_weights(weights_input_hidden, weights_hidden_output, features, targets, learnrate)
        if e % (epochs / 10) == 0:
            hidden_output = sigmoid(np.dot(features, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output, weights_hidden_output))
            loss = np.mean((out - targets) ** 2)
            print("Train loss: {:.3f}".format(loss))

    return weights_input_hidden, weights_hidden_output

def calculate_accuracy(features, targets, weights_input_hidden, weights_hidden_output):
    """
    Calculate the accuracy of predictions
    """
    features = preprocess_data(features)  # Ensure features is preprocessed
    hidden_output = sigmoid(np.dot(features, weights_input_hidden))
    output = sigmoid(np.dot(hidden_output, weights_hidden_output))
    predictions = output > 0.5
    accuracy = np.mean(predictions == targets)
    return accuracy

# Prepare data
features = preprocess_data(features)
features_test = preprocess_data(features_test)

# Calculate accuracy on test data
weights_input_hidden, weights_hidden_output = gradient_descent(features, targets, learnrate=0.5)
accuracy = calculate_accuracy(features_test, targets_test, weights_input_hidden, weights_hidden_output)
print("Prediction accuracy: {:.3f}".format(accuracy))
