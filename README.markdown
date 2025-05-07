*_This project was solely based for learning purposes and should not be used as a financing solution for any kind._*
## Stock Price Prediction with LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data from Yahoo Finance. The model is built using PyTorch and includes data preprocessing, model training, and visualization of predictions.

## Project Overview

The goal of this project is to forecast the closing price of a stock (Apple Inc., ticker: AAPL) using an LSTM model. The dataset is fetched via the `yfinance` library, preprocessed with standard scaling, and split into training and testing sets. The model predicts future stock prices and evaluates performance using the Root Mean Squared Error (RMSE).

Key features:
- Fetches historical stock data from Yahoo Finance.
- Preprocesses data with `StandardScaler` for normalization.
- Implements an LSTM model using PyTorch.
- Visualizes actual vs. predicted prices and prediction errors.
- Utilizes GPU acceleration (if available) for faster training.

## Installation

To run this project, ensure you have Python 3.11 installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include:
   ```
   numpy
   pandas
   matplotlib
   yfinance
   torch
   scikit-learn
   ```

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook Main.ipynb
   ```

## Results

The model generates predictions for the test dataset and evaluates performance using RMSE. The visualization includes:
- **Top Plot**: Actual stock prices (blue) vs. predicted prices (green).
- **Bottom Plot**: Prediction error (red) with the RMSE threshold (blue dashed line).

Example output:
![Actual vs Predicted Prices](https://i.imgur.com/EGFpE79.png)

The RMSE value indicates the model's accuracy, with lower values representing better performance.

## Future Improvements

- Experiment with different LSTM architectures (e.g., more layers or units).
- Incorporate additional features like trading volume or technical indicators.
- Implement hyperparameter tuning using grid search.
- Add support for multiple stocks.
- Deploy the model as a web application for real-time predictions.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows PEP 8 standards and includes relevant tests.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [scikit-learn](https://scikit-learn.org/) for preprocessing utilities.

For any questions or issues,please contact [alitaseeb2@gmail.com](mailto:alitaseeb2@gmail.com).
