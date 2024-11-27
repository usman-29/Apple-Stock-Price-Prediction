# Apple Stock Price Prediction

## Project Overview

Apple Inc. is a global leader in technology and innovation, making its stock a popular investment choice. Predicting Apple's stock price is a critical task for investors and analysts to make informed decisions. This project explores **Apple's stock price prediction** as a **multivariate time series problem** using advanced machine learning techniques, specifically an LSTM-based deep learning model.

---

## Dataset Details

The **Apple Stock Price Dataset** includes essential variables for stock market analysis:

| Variable Name | Description                                             |
|---------------|---------------------------------------------------------|
| **Date**      | Date of the stock price record                          |
| **Open**      | Stock's opening price on a given date                   |
| **High**      | Highest price the stock traded at during the day        |
| **Low**       | Lowest price the stock traded at during the day         |
| **Close**     | Stock's closing price on a given date                   |
| **Volume**    | Total number of shares traded on that date              |

---

## Methodology

1. **Data Preprocessing**:
   - Imported and cleaned the dataset using `numpy` and `pandas`.
   - Scaled the data using `MinMaxScaler` for normalization.
   - Split the dataset into training and testing sets using `train_test_split`.

2. **Model Architecture**:
   - A deep learning model with three LSTM layers and Dropout for regularization.
   - Fully connected Dense output layer for predictions.

3. **Model Compilation**:
   - Optimizer: Adam
   - Loss Function: Mean Squared Error
   - Metrics: Root Mean Squared Error

4. **Training**:
   - Trained the model for 100 epochs with a batch size of 3.
   - Used **EarlyStopping** to prevent overfitting and improve training efficiency.

5. **Libraries Used**:
   - `numpy`, `pandas` for data handling.
   - `tensorflow`, `keras` for model building.
   - `matplotlib` for visualization.
   - `sklearn` for scaling and splitting the data.

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=3,
                    callbacks=[early_stopping])
