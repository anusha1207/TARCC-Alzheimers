from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold

# Code source: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# Code source 2: https://www.analyticsvidhya.com/blog/2021/07/performing-multi-class-classification-on-fifa-dataset-using-keras/

def DNN_One(input, output, epochs=200, batch_size=5, n_split=5):
    # Data Processing on Input so that we can fit the input into our neural network training
    input_values = input.values
    input_float = input_values.astype(float)

    # Convert output to categorical variables
    cat_output = to_categorical(output)

    def DNN_model():
        model = Sequential()
        # First layer of nodes of number of columns/features in dataset and then hidden layer of 128 nodes
        model.add(Dense(128, input_shape=(len(input.columns), ), activation='relu'))
        # Another hidden layer of 64 nodes
        model.add(Dense(64, activation='relu'))
        # Output layer of number of output classes (number of unique values of 'P1_PT_TYPE' column/feature)
        model.add(Dense(len(output.unique()) + 1, activation='softmax'))
        # Use categorical_crossentropy for loss since this is a classification problem
        # Use ADAM optimizer since we are dealing with a large dataset (in terms of number of rows and features)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Training and Testing Phase; using Cross Validation to evaluate model
    estimator = KerasClassifier(build_fn=DNN_model, epochs=epochs, batch_size=batch_size, verbose=1)
    kfold = KFold(n_splits=n_split, shuffle=True)
    results = cross_val_score(estimator, input_float, cat_output, cv=kfold)
    return results.mean()

def DNN_Two(input, output, epochs=200, batch_size=5, n_split=5):
  # Data Processing on Input so that we can fit the input into our neural network training
  input_values = input.values
  input_float = input_values.astype(float)

  # Convert output to categorical variables
  cat_output = to_categorical(output)

  def DNN_model():
    model = Sequential()
    # First layer of nodes of number of columns/features in dataset and then hidden layer of 128 nodes
    model.add(Dense(64, input_shape=(len(input.columns), ), activation='relu'))
    # Another hidden layer of 64 nodes
    model.add(Dense(64, activation='relu'))
    # Another hidden layer of 64 nodes
    model.add(Dense(64, activation='relu'))
    # Output layer of number of output classes (number of unique values of 'P1_PT_TYPE' column/feature)
    model.add(Dense(len(output.unique()) + 1, activation='softmax'))
    # Use categorical_crossentropy for loss since this is a classification problem
    # Use ADAM optimizer since we are dealing with a large dataset (in terms of number of rows and features)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  # Training and Testing Phase; using Cross Validation to evaluate model
  estimator = KerasClassifier(build_fn=DNN_model, epochs=epochs, batch_size=batch_size, verbose=1)
  kfold = KFold(n_splits=n_split, shuffle=True)
  results = cross_val_score(estimator, input_float, cat_output, cv=kfold, error_score='raise')
  return results.mean()

def DNN_Three(input, output, epochs=200, batch_size=5, n_split=5):
  # Data Processing on Input so that we can fit the input into our neural network training
  input_values = input.values
  input_float = input_values.astype(float)

  # Convert output to categorical variables
  cat_output = to_categorical(output)

  def DNN_model():
    model = Sequential()
    # First layer of nodes of number of columns/features in dataset and then hidden layer of 128 nodes
    model.add(Dense(128, input_shape=(len(input.columns), ), activation='relu'))
    # Output layer of number of output classes (number of unique values of 'P1_PT_TYPE' column/feature)
    model.add(Dense(len(output.unique()) + 1, activation='softmax'))
    # Use categorical_crossentropy for loss since this is a classification problem
    # Use ADAM optimizer since we are dealing with a large dataset (in terms of number of rows and features)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  # Training and Testing Phase; using Cross Validation to evaluate model
  estimator = KerasClassifier(build_fn=DNN_model, epochs=epochs, batch_size=batch_size, verbose=1)
  kfold = KFold(n_splits=n_split, shuffle=True)
  results = cross_val_score(estimator, input_float, cat_output, cv=kfold, error_score='raise')
  return results.mean()

def DNN_Four(input, output, epochs=200, batch_size=5, n_split=5):
  # Data Processing on Input so that we can fit the input into our neural network training
  input_values = input.values
  input_float = input_values.astype(float)

  # Convert output to categorical variables
  cat_output = to_categorical(output)

  def DNN_model():
    model = Sequential()
    # First layer of nodes of number of columns/features in dataset and then hidden layer of 128 nodes
    model.add(Dense(128, input_shape=(len(input.columns), ), activation='relu'))
    # Another hidden layer of 64 nodes
    model.add(Dense(128, activation='relu'))
    # Another hidden layer of 64 nodes
    model.add(Dense(128, activation='relu'))
    # Output layer of number of output classes (number of unique values of 'P1_PT_TYPE' column/feature)
    model.add(Dense(len(output.unique()) + 1, activation='softmax'))
    # Use categorical_crossentropy for loss since this is a classification problem
    # Use ADAM optimizer since we are dealing with a large dataset (in terms of number of rows and features)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  # Training and Testing Phase; using Cross Validation to evaluate model
  estimator = KerasClassifier(build_fn=DNN_model, epochs=epochs, batch_size=batch_size, verbose=1)
  kfold = KFold(n_splits=n_split, shuffle=True)
  results = cross_val_score(estimator, input_float, cat_output, cv=kfold, error_score='raise')
  return results.mean()