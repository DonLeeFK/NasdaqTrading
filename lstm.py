import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
import csv


# Load data using yfinance and calculate indicators
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate SMA_20
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Calculate SMA_50
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    data = data.dropna()
    
    return data[['Close', 'SMA_20', 'SMA_50', 'RSI']]

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # Assuming 'Close' is the first column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        prob = self.sigmoid(out[:, 0])
        #change = torch.tanh(out[:, 1])
        change = out[:, 1]
        return prob, change

class KellyTrader:
    def __init__(self, stock_data, initial_balance):
        self.stock_data = stock_data
        self.balance = initial_balance
        self.portfolio = 0
        self.portfolio_values = [initial_balance]
        self.trades = []
        self.current_price = stock_data.iloc[0]['Close']

    def trade(self, probability, expected_change):
        self.current_price = self.stock_data.iloc[len(self.portfolio_values) - 1]['Close']
        
        kelly_fraction = probability - (1 - probability) / expected_change if expected_change != 0 else 0
        kelly_fraction = max(min(kelly_fraction, 1), 0)  # Limit the fraction to [0, 1]
        #kelly_fraction = probability - (1 - probability) / (1 + expected_change)
        #kelly_fraction = max(min(kelly_fraction, 1), 0)  # Limit the fraction to [0, 1]
        
        total_value = self.balance + self.portfolio * self.current_price
        current_stock_value = self.portfolio * self.current_price
        if isinstance(total_value, pd.Series):
            current_stock_percentage = np.where(
                total_value > 0,
                current_stock_value / total_value,
                0
            )
        else:
            current_stock_percentage = current_stock_value / total_value if total_value > 0 else 0
        
        target_stock_value = kelly_fraction * total_value
        trade_amount = target_stock_value - current_stock_value
        if isinstance(trade_amount, pd.Series):
            # Handle trade_amount as a series
            trade_amount = trade_amount.iloc[0]
        
        if trade_amount > 0:  # Buy
            if isinstance(self.current_price, pd.Series):
                self.current_price = self.current_price.iloc[0]

            # Determine the shares to buy
            shares_to_buy = min(trade_amount / self.current_price, self.balance / self.current_price)
            cost = shares_to_buy * self.current_price
            self.balance -= cost
            self.portfolio += shares_to_buy
            action = 'buy'
        elif trade_amount < 0:  # Sell
            if isinstance(self.current_price, pd.Series):
                self.current_price = self.current_price.iloc[0]

            # Determine the shares to sell
            shares_to_sell = min(-trade_amount / self.current_price, self.portfolio)
            proceeds = shares_to_sell * self.current_price
            self.balance += proceeds
            self.portfolio -= shares_to_sell
            action = 'sell'
        else:
            action = 'hold'
        
        portfolio_value = self.balance + self.portfolio * self.current_price
        self.portfolio_values.append(portfolio_value)
        
        if action != 'hold':
            self.trades.append({
                'action': action,
                'amount': abs(trade_amount),
                'price': self.current_price
            })
        
        # Print daily information
        current_price = self.current_price.iloc[0] if isinstance(self.current_price, pd.Series) else self.current_price
        balance = self.balance.iloc[0] if isinstance(self.balance, pd.Series) else self.balance
        total_portfolio_value = self.portfolio * current_price
        portfolio_value = self.portfolio_values[-1]  # Ensure we access the latest portfolio value correctly.
        portfolio_value = portfolio_value.iloc[0] if isinstance(portfolio_value, pd.Series) else portfolio_value
        current_stock_percentage = float(current_stock_percentage.iloc[0]) if isinstance(current_stock_percentage, pd.Series) else float(current_stock_percentage)
        trade_amount = abs(trade_amount.iloc[0] if isinstance(trade_amount, pd.Series) else trade_amount)

        # Handle potential NaN or negative stock percentage
        current_stock_percentage = 0 if pd.isna(current_stock_percentage) or current_stock_percentage < 0 else current_stock_percentage
        """
        print(f"  Stock Price: ${current_price:.2f}")
        print(f"  Balance: ${balance:.2f}")
        print(f"  Portfolio: {self.portfolio:.4f} shares (${total_portfolio_value:.2f})")
        print(f"  Total Value: ${portfolio_value:.2f}")
        print(f"  Probability: {probability}, Expected Change: {expected_change}")
        print(f"  Kelly Fraction: {kelly_fraction:.4f}")
        print(f"  Current Stock Percentage: {current_stock_percentage:.4f}")
        print(f"  Trade: {action.capitalize()} ${trade_amount:.2f}")
        print("--------------------")
        """
        record = {
            'Day': len(self.portfolio_values) - 1,
            'Stock Price': f"${current_price:.2f}",
            'Balance': f"${balance:.2f}",
            'Porfolio': f"${total_portfolio_value}",
            'Total Value': f"${portfolio_value:.2f}",
            'Current Stock Percentage': f"{100*current_stock_percentage:.2f}%",
            'Trade Action': action.capitalize(),
            'Trade Amount': f"${abs(trade_amount):.2f}",
            'Trade Ratio': f"{100*abs(trade_amount)/portfolio_value:.2f}%"
                  }
        return action.capitalize(), abs(trade_amount)/portfolio_value, self.balance/portfolio_value, record

    
        

    def close_position(self):
        if self.portfolio > 0:
            self.balance += self.portfolio * self.current_price
            self.trades.append({
                'action': 'sell',
                'amount': self.portfolio * self.current_price,
                'price': self.current_price
            })
            self.portfolio = 0
            self.portfolio_values.append(self.balance)
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate the Sharpe ratio given a series of returns.
        :param returns: List or array of returns
        :param risk_free_rate: Annual risk-free rate (default 2%)
        :return: Sharpe ratio
        """
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days in a year
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_daily_returns(self):
        strategy_returns = []
        for i in range(1, len(self.portfolio_values)):
            # Ensure previous value is treated as a scalar before comparison
            previous_value = self.portfolio_values[i-1] if not isinstance(self.portfolio_values[i-1], pd.Series) else self.portfolio_values[i-1].iloc[0]
            
            if previous_value != 0:
                daily_return = (self.portfolio_values[i] - previous_value) / previous_value
            else:
                daily_return = 0  # Assume no return if previous value was 0
                
            strategy_returns.append(daily_return)

        stock_returns = self.stock_data['Close'].pct_change().values[1:]  # Skip the first NaN

        return strategy_returns, stock_returns

    def plot_results(self):
        strategy_returns, stock_returns = self.calculate_daily_returns()

        # Calculate cumulative returns
        strategy_returns = [item if isinstance(item, (int, float)) else item[0] for item in strategy_returns]
        cumulative_strategy_returns = np.cumprod(1 + np.array(strategy_returns)) - 1
        cumulative_stock_returns = np.cumprod(1 + stock_returns) - 1

        # Create a date range for x-axis
        dates = self.stock_data.index[-len(strategy_returns):]

        # Ensure all arrays have the same length
        min_length = min(len(dates), len(cumulative_strategy_returns), len(cumulative_stock_returns))
        dates = dates[:min_length]
        cumulative_strategy_returns = cumulative_strategy_returns[:min_length]
        cumulative_stock_returns = cumulative_stock_returns[:min_length]

        # Calculate Sharpe ratios
        kelly_sharpe = self.calculate_sharpe_ratio(strategy_returns)
        stock_sharpe = self.calculate_sharpe_ratio(stock_returns)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(dates, cumulative_strategy_returns, label='Kelly Strategy', color='blue')
        plt.plot(dates, cumulative_stock_returns, label='Buy and Hold', color='red')
        plt.title('Cumulative Returns: Kelly Strategy vs Buy and Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('result/result.png')
        #plt.show()

        # Print summary statistics
        final_strategy_return = cumulative_strategy_returns[-1]
        final_stock_return = cumulative_stock_returns[-1]
        print(f"Initial Balance: ${self.portfolio_values[0]:.2f}")
        print(f"Final Balance: ${self.portfolio_values[-1]:.2f}")
        print(f"Kelly Strategy Total Return: {final_strategy_return:.2%}")
        print(f"Buy and Hold Total Return: {final_stock_return:.2%}")
        print(f"Kelly Strategy Sharpe Ratio: {kelly_sharpe:.4f}")
        print(f"Buy and Hold Sharpe Ratio: {stock_sharpe:.4f}")
        print(f"Number of trades: {len(self.trades)}")
        
        # Write CSV to file
        headers = ["Metric", "Value"]
        rows = [
            ["Initial Balance", f"${self.portfolio_values[0]:.2f}"],
            ["Final Balance", f"${self.portfolio_values[-1]:.2f}"],
            ["Kelly Strategy Total Return", f"{final_strategy_return:.2%}"],
            ["Buy and Hold Total Return", f"{final_stock_return:.2%}"],
            ["Kelly Strategy Sharpe Ratio", f"{kelly_sharpe:.4f}"],
            ["Buy and Hold Sharpe Ratio", f"{stock_sharpe:.4f}"],
            ["Number of trades", str(len(self.trades))]
        ]
        
        with open('result/result.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(rows)
        
        

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            prob, change = model(seq)
            loss = 3*criterion(prob, (labels > 0).float()) + criterion(change, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction function
def predict(model, data):
    model.eval()
    with torch.no_grad():
        prob, change = model(data)
    return prob.item(), change.item()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # Make predictions
        probs, changes = model(X_test)
        
        # Convert probabilities to binary predictions
        pred_direction = (probs > 0.5).float()
        true_direction = (y_test > 0).float()
        
        # Classification metrics
        accuracy = accuracy_score(true_direction.cpu(), pred_direction.cpu())
        precision = precision_score(true_direction.cpu(), pred_direction.cpu())
        recall = recall_score(true_direction.cpu(), pred_direction.cpu())
        f1 = f1_score(true_direction.cpu(), pred_direction.cpu())
        
        # Regression metric
        mae = torch.mean(torch.abs(changes - y_test)).item()
        
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return accuracy, precision, recall, f1, mae
def save_model(model, filename):
    """
    Save only the model state.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename):
    """
    Load only the model state.
    """
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"No model found at {filename}")
        return None


# Main execution
# Main execution
if __name__ == "__main__":
    # Set the training flag
    train = False  # Set to True for training, False for trading

    if train:
        # Load data for training
        ticker = "^IXIC"  # Nasdaq Composite index
        start_date = "2010-01-01"
        end_date = "2023-06-01"
        data = load_data(ticker, start_date, end_date)
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        seq_length = 10
        
        # Calculate daily returns first
        returns = np.diff(scaled_data[:, 0]) / scaled_data[:-1, 0]
        
        # Create sequences after calculating returns
        X, y = [], []
        for i in range(len(returns) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(returns[i+seq_length-1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Initialize model
        input_size = 4  # Close, SMA_20, SMA_50, RSI
        hidden_size = 64
        num_layers = 2
        output_size = 2
        model = StockLSTM(input_size, hidden_size, num_layers, output_size)
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training process
        criterion = nn.MSELoss()
        num_epochs = 300
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
        
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        
        # Save the model
        model_filename = "nasdaq_lstm_model.pth"
        save_model(model, model_filename)
        
        # Evaluate the model
        print("\nModel Evaluation:")
        accuracy, precision, recall, f1, mae = evaluate_model(model, X_test, y_test)
    else:
        # Load new data for trading
        ticker = "^IXIC"  # Nasdaq Composite index
        start_date = "2023-06-02"  # Start from where training data ended
        #end_date = "2024-08-14"  # Or use pd.Timestamp.today() for current date
        end_date = pd.Timestamp.today()
        print(end_date)
        trading_data = load_data(ticker, start_date, end_date)
        
        # Initialize model
        input_size = 4  # Close, SMA_20, SMA_50, RSI
        hidden_size = 64
        num_layers = 2
        output_size = 2
        model = StockLSTM(input_size, hidden_size, num_layers, output_size)
        
        model_filename = "nasdaq_lstm_model.pth"
        model = load_model(model, model_filename)
        
        if model is None:
            print("Error: Could not load the model. Exiting.")
            exit()
        
        # Prepare data for trading
        scaler = MinMaxScaler()
        scaled_trading_data = scaler.fit_transform(trading_data)
        
        seq_length = 10
        X_trading = []
        for i in range(len(scaled_trading_data) - seq_length):
            X_trading.append(scaled_trading_data[i:i+seq_length])
        X_trading = torch.FloatTensor(np.array(X_trading))
        
        # Initialize KellyTrader
        initial_balance = 1000000  # $10,000 initial balance
        trader = KellyTrader(trading_data, initial_balance)

        # Perform trading
        model.eval()  # Set the model to evaluation mode
        records = []
        with torch.no_grad():
            for i in range(len(X_trading)):
                sequence = X_trading[i].unsqueeze(0)
                prob, change = model(sequence)
                #print(prob, change)
                action, trade_ratio, cash_ratio, record= trader.trade(prob.item(), change.item())
                records.append(record)
                if i == len(X_trading) - 1:
                    message = f"Action: {action}, Trade Ratio: {100*trade_ratio}%, Cash Ratio: {100*cash_ratio}%"
                    with open("result/prediction.txt", "w") as file:
                        file.write(message)
                    print(f"Action: {action}, Trade Ratio: {100*trade_ratio}%, Cash Ratio: {100*cash_ratio}%")
        

        fieldnames = records[0].keys()
        with open("result/record.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        
                    
                

        # Close the final position
        trader.close_position()

        # Plot and print the results
        trader.plot_results()