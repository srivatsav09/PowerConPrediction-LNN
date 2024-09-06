import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ncps.tf import LTCCell

# Load dataset
# Assuming the dataset is in a CSV file
df = pd.read_csv('household_power_consumption.csv')

# Convert Date and Time into datetime and extract features
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Day'] = df['Datetime'].dt.day

# Drop original Date, Time, and Datetime columns
df.drop(['Date', 'Time', 'Datetime'], axis=1, inplace=True)

# Handle missing values if any
df.fillna(df.mean(), inplace=True)

# Select features and target
features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
            'Hour', 'DayOfWeek', 'Day']
target = 'Global_active_power'

X = df[features].values
y = df[target].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for the RNN input
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Reshape target for consistency
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Build the LNN model using ncps
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, X_train.shape[2])),
    tf.keras.layers.RNN(LTCCell(units=64), return_sequences=False),
    tf.keras.layers.Dense(1, activation='linear')  # Regression output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae:.4f}')
