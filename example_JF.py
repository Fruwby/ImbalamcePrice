import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot


# Scale data to be efficiently processed by the neural network
def data_scaling(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    transformed = scaler.transform(data)
    return transformed, scaler


# Descale data for post-processing analysis
def data_descaling(data, scaler):
    inversed = scaler.inverse_transform(data)  # inverse transform
    return inversed


def post_processing_metrics(y_test_true, y_test_hat, name='None'):
    # Return the performance of the prediction
    d = y_test_true - y_test_hat
    mse = np.mean(d ** 2)
    mae = np.mean(abs(d))
    rmse = np.sqrt(mse)
    r2 = 1 - (sum(d ** 2) / sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f'{name} :')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R-Squared: {r2}')
    print(f' ')

    pd_metrics = pd.DataFrame({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-Squared': r2}, index=[0])
    return pd_metrics


# Import training data
data = pd.read_hdf('full_data.h5')

X = data.iloc[:, 0:3].values
y_true = data.iloc[:, 3].values

# Divide into train and test sets
X_train, X_test, y_train_true, y_test_true = train_test_split(X, y_true, test_size=0.2)


# ########## Linear model ##########
reg = LinearRegression().fit(X_train, y_train_true)

# Get parameters
df0 = pd.DataFrame({f'intercept': reg.intercept_}, index=[0])
df1 = pd.DataFrame({f'coefficients': np.array(reg.coef_)})
df_LR_params = pd.concat([df0, df1], ignore_index=False, axis=1)

# Predict using the linear model
y_test_hat = reg.predict(X_test)
print(y_test_hat.shape)
y_test_hat2 = reg.intercept_ + np.dot(X_test, np.reshape(reg.coef_, (-1, 1)))
y_test_hat2 = np.reshape(y_test_hat2, -1)

# Compute performance on the test set
df_test_metrics = post_processing_metrics(y_test_true, y_test_hat, name='Linear Regression')
df_test_metrics2 = post_processing_metrics(y_test_true, y_test_hat2, name='Linear Regression2')

# Store information in Excel file
filename = f'final_LR_model.xlsx'
df_LR_params.to_excel(filename, sheet_name='Parameters')
df_test_metrics.to_excel(filename, sheet_name='Metrics')


# ########## Neural Network model ##########
# Hyper-parameters
n_layers = 2
n_neurons = 5
type_activation = 'LeakyReLU'  # 'LeakyReLU' -- 'ReLU'
batch_size = 32
n_epochs = 10

# Normalize the data for Neural Networks
X_train_norm, scaler_X = data_scaling(X_train)
y_train_true_norm, scaler_y = data_scaling(np.reshape(y_train_true, (-1, 1)))
y_train_true_norm = np.reshape(y_train_true_norm, -1)

X_test_norm = scaler_X.transform(X_test)
X_all = np.concatenate((X_train_norm, X_test_norm), axis=0)

# Store scaling values (for use in optimization model)
df0 = pd.DataFrame({'X_mean': scaler_X.mean_})
df1 = pd.DataFrame({'X_std': scaler_X.var_**0.5})
df2 = pd.DataFrame({'y_mean': scaler_y.mean_})
df3 = pd.DataFrame({'y_std': scaler_y.var_**0.5})
df_scaling = pd.concat([df0, df1, df2, df3], ignore_index=False, axis=1)
df_scaling.index = ['p_i', 'p_o', 'q']

# Divide into train and test sets
X_train_norm, X_valid_norm, y_train_true_norm, y_valid_true_norm = train_test_split(X_train_norm,
                                                                                    y_train_true_norm,
                                                                                    test_size=0.25)

# Fit the model
model = tf.keras.Sequential()
for ll in range(n_layers):
    model.add(tf.keras.layers.Dense(n_neurons, activation="linear",
                                    name=f'layer_{ll}', input_shape=(3,)))
    model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dense(1, activation="linear",
                                name=f'layer_{n_layers+1}'))
model.compile(optimizer='Adam', loss='mse')
# callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
# model.fit(X_train_norm, y_train_true_norm,
#           validation_data=(X_valid_norm, y_valid_true_norm),
#           shuffle=True,
#           callbacks=callbacks,
#           batch_size=batch_size,
#           epochs=n_epochs)

# Pruning part for sparsity
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.20,
                                                               final_sparsity=0.50,
                                                               begin_step=0,
                                                               end_step=np.ceil(X_train_norm.shape[0] / batch_size).astype(np.int32) * n_epochs)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='Adam', loss='mse')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    tfmot.sparsity.keras.UpdatePruningStep(),
]

model_for_pruning.fit(X_train_norm, y_train_true_norm,
                      validation_data=(X_valid_norm, y_valid_true_norm),
                      shuffle=True,
                      callbacks=callbacks,
                      batch_size=batch_size,
                      epochs=n_epochs)
model = model_for_pruning

# Get weights
df_weights = pd.DataFrame()
count = 0
for layer in model.layers:
    if len(layer.weights) > 1:
        count = count + 1
        df0 = pd.DataFrame({f'weight_{count}': np.reshape(layer.weights[0].numpy(), -1)})
        df1 = pd.DataFrame({f'bias_{count}': layer.weights[1].numpy()})
        df_weights = pd.concat([df_weights, df0, df1], ignore_index=False, axis=1)

# data = pd.read_excel('final_weights.xlsx')
# W = np.reshape(data.iloc[:150, 1].values, (3, 50))

# Predict values of the test set
y_test_hat_norm = model.predict(X_test_norm, batch_size=32, verbose=0)
print(y_test_hat_norm.shape)  # shape (20000, 1)
y_test_hat = data_descaling(y_test_hat_norm, scaler_y)
y_test_hat = np.reshape(y_test_hat, -1)

# Compute performance on the test set
df_test_metrics = post_processing_metrics(y_test_true, y_test_hat, name='Neural Network')

# Store information in Excel file
filename = f'final_NN_{type_activation}_{n_layers}layers_{n_neurons}neurons.xlsx'

writer = pd.ExcelWriter(filename, engine='xlsxwriter')
df_weights.to_excel(writer, sheet_name='Parameters')
df_scaling.to_excel(writer, sheet_name='Scaling')
df_test_metrics.to_excel(writer, sheet_name='Metrics')
writer.save()
