# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 04/10/25
### BY : JEEVANESH S
### REG: 212222243002


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Read the CSV file into a DataFrame
data = pd.read_csv('cardekho.csv')

# Display dataset info to select appropriate column
print("Dataset Info:")
print(data.head())
print("\nColumns:", data.columns)

# Select a numeric column for analysis (change this as needed)
numeric_columns = data.select_dtypes(include=[np.number]).columns
selected_column = numeric_columns[0]  # Using first numeric column
print(f"\nUsing column for analysis: {selected_column}")

# Perform Augmented Dickey-Fuller test
result = adfuller(data[selected_column])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

# Fit an AutoRegressive (AR) model with 13 lags
lag_order = 13
model = AutoReg(train_data[selected_column], lags=lag_order)
model_fit = model.fit()

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(data[selected_column], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data[selected_column], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Compare the predictions with the test data
mse = mean_squared_error(test_data[selected_column], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot the test data and predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data[selected_column].values, label=f'Test Data - {selected_column}')
plt.plot(predictions.values, label=f'Predictions - {selected_column}', linestyle='--')
plt.xlabel('Index')
plt.ylabel(selected_column)
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()q
```
### OUTPUT:

GIVEN DATA:
<img width="651" height="351" alt="image" src="https://github.com/user-attachments/assets/0c971e8c-b34f-4384-8798-c56c61eea1ab" />


PACF - ACF
<img width="531" height="411" alt="image" src="https://github.com/user-attachments/assets/04d67ac8-b1c7-43dc-bcba-e0c47ffb2262" />
<img width="530" height="406" alt="image" src="https://github.com/user-attachments/assets/9af05b1b-e1a0-41f8-99be-4560d205754c" />
MSE:
<img width="1340" height="32" alt="image" src="https://github.com/user-attachments/assets/3dceec3d-f6ff-4a06-b071-74f0d1bcb30b" />

FINIAL PREDICTION:
<img width="947" height="512" alt="image" src="https://github.com/user-attachments/assets/25b3941c-d376-4dfb-9a53-6eca0223f596" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
