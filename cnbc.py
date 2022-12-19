import numpy as np

# Assume we have two features, X1 and X2
# and two classes, Y1 and Y2

# Set the prior probabilities of each class
P_Y1 = 0.3
P_Y2 = 0.7

# Set the prior probabilities of each feature given each class
P_X1_Y1 = 0.4
P_X1_Y2 = 0.8
P_X2_Y1 = 0.7
P_X2_Y2 = 0.2

# Set the evidence (observed values of the features)
X1 = 1
X2 = 1

# Calculate the posterior probabilities of each class given the evidence
P_Y1_X1_X2 = (P_Y1 * P_X1_Y1 * P_X2_Y1) / ((P_Y1 * P_X1_Y1 * P_X2_Y1) + (P_Y2 * P_X1_Y2 * P_X2_Y2))
P_Y2_X1_X2 = (P_Y2 * P_X1_Y2 * P_X2_Y2) / ((P_Y1 * P_X1_Y1 * P_X2_Y1) + (P_Y2 * P_X1_Y2 * P_X2_Y2))

# Print the results
print("P(Y1|X1,X2) =", P_Y1_X1_X2)
print("P(Y2|X1,X2) =", P_Y2_X1_X2)

#===========================================================================

import numpy as np

def correlated_naive_bayes(x, y, x_test):
    classes = np.unique(y)
    num_features = x.shape[1]
    likelihoods = []
    for c in classes:
        likelihood = []
        for i in range(num_features):
            feature_values = x[y == c, i]
            likelihood.append((np.mean(feature_values), np.std(feature_values)))
        likelihoods.append(likelihood)

    priors = [len(y[y == c]) / len(y) for c in classes]

    predictions = []
    for x_t in x_test:
        posteriors = []
        for i, c in enumerate(classes):
            likelihood = 1
            for j, f in enumerate(x_t):
                mu, sigma = likelihoods[i][j]
                likelihood *= norm.pdf(f, mu, sigma)
            posteriors.append(likelihood * priors[i])
        predictions.append(classes[np.argmax(posteriors)])

    return predictions

#==========================================================================
import numpy as np

class CorrelatedNB:
  def fit(self, X, y):
    # Store the number of classes and the number of features
    self.num_classes = len(np.unique(y))
    self.num_features = X.shape[1]

    # Store the class prior probabilities (the frequency of each class in the training set)
    self.class_priors = [np.mean(y == c) for c in range(self.num_classes)]

    # Store the conditional probabilities for each class and feature
    self.conditional_probs = np.zeros((self.num_classes, self.num_features))
    for c in range(self.num_classes):
      for f in range(self.num_features):
        # Calculate the mean of the feature values for samples in class c
        feature_mean = np.mean(X[y == c, f])
        # Calculate the variance of the feature values for samples in class c
        feature_variance = np.var(X[y == c, f])
        self.conditional_probs[c, f] = (feature_mean, feature_variance)

  def predict(self, X):
    # Store the number of samples and the number of features
    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Store the log probabilities for each class
    log_probs = np.zeros((num_samples, self.num_classes))

    # Loop over the samples and classes and calculate the log probabilities
    for s in range(num_samples):
      for c in range(self.num_classes):
        # Calculate the log probability of class c given the features
        log_probs[s, c] = np.log(self.class_priors[c])
        for f in range(num_features):
          # Extract the mean and variance for the feature and class
          feature_mean, feature_variance = self.conditional_probs[c, f]
          # Calculate the log probability of the feature given the class
          log_prob = -0.5 * np.log(2 * np.pi * feature_variance) - 0.5 * (X[s, f] - feature_mean)**2 / feature_variance
          log_probs[s, c] += log_prob

    # Return the predicted class for each sample
    return np.argmax(log_probs, axis=1)
