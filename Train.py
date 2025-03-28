import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
torch.manual_seed(42)
class DataLoader:
    def __init__(self, tweet_file, exchange_file):
        self.tweet_file = tweet_file
        self.exchange_file = exchange_file
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test, self.start_prices_test, self.if_left = self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.tweet_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df.drop(columns=['tags', 'text', 'username'], inplace=True)
        df.sort_index()
        
        df_exchange = pd.read_csv(self.exchange_file)
        df_exchange['Datetime'] = pd.to_datetime(df_exchange['Datetime'])
        df_exchange.set_index('Datetime', inplace=True)
        
        df['Price_24h_Future'] = df.index.map(df_exchange['Close'].shift(-24))
        df.dropna(inplace=True)
        
        X = df.drop(columns=['Price_24h_Future', 'close'])
        start_prices = list(df['close'])
        y = df['Price_24h_Future']
        
        X_n = int(len(df) * 0.8)
        start_date_test = df.index[X_n]

        end_date_test = df.index.max()


        
        start_test_price = df_exchange['Close'][start_date_test]
        end_test_price = df_exchange['Close'][end_date_test]
        if_left = end_test_price / start_test_price
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        X_train = torch.tensor(X_scaled[:X_n], dtype=torch.float32)
        X_test = torch.tensor(X_scaled[X_n:], dtype=torch.float32)
        y_train = torch.tensor(y_scaled[:X_n], dtype=torch.float32)
        y_test = torch.tensor(y_scaled[X_n:], dtype=torch.float32)
        
        return X_train, X_test, y_train, y_test, start_prices[X_n:], if_left




class PricePredictor(nn.Module):
    def __init__(self,input_size,layer1,layer2,layer3,activation_function):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, layer1)
        self.activation1 = activation_function()
        self.fc2 = nn.Linear(layer1, layer2)
        self.activation2 = activation_function()
        self.fc3 = nn.Linear(layer2, layer3)
        self.activation3 = activation_function()
        self.fc4 = nn.Linear(layer3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        x = self.activation3(x)
        x = self.fc4(x)
        return x

class Trainer:
    def __init__(self, model, data_loader,writer_path, lr=0.001, epochs=10000):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.data_loader = data_loader
        self.writer = SummaryWriter(log_dir=writer_path)
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            outputs = self.model(self.data_loader.X_train)
            loss = self.criterion(outputs, self.data_loader.y_train)
            pred_test = self.model.forward(self.data_loader.X_test).detach().numpy()[:,0]
            test_mse = mean_squared_error(self.data_loader.y_test,pred_test)
            
            self.writer.add_scalar(
                tag='train loss',
                scalar_value=loss,
                global_step=epoch
            )
            self.writer.add_scalar(
                tag='test loss',
                scalar_value=test_mse,
                global_step=epoch
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        return test_mse

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(self.data_loader.X_test)
            mse = self.criterion(predicted, self.data_loader.y_test)
            predicted_prices = self.data_loader.scaler_y.inverse_transform(predicted.numpy())
            actual_prices = self.data_loader.scaler_y.inverse_transform(self.data_loader.y_test.numpy())
            r2 = r2_score( actual_prices, predicted_prices)
            total, max_total, correct_per = self.sim_year(predicted_prices, self.data_loader.start_prices_test, actual_prices)

            
            print(f'Ratio made: {total:.3f} | If always correct: {max_total:.3f} | If left in during: {self.data_loader.if_left:.3f} | Percentage Movement correct {correct_per:.3f}% ')
            print(f'MSE: {mse:.4f} | R² Score: {r2:.4f}')
            return correct_per,total
    
    @staticmethod
    def sim_year(predicted, starts, actual):
        total, max_total, left = 1, 1, 1
        correct_per = float(sum( (p > s and a >s )or( p< s and a < s )for p, s,a in zip(predicted, starts,actual)) / len(actual))*100

        for i in range(len(predicted)):
            if predicted[i] > starts[i]:
                total *= actual[i] / starts[i]
            if actual[i] > starts[i]:
                max_total *= actual[i] / starts[i]
            left *= actual[i] / starts[i]
        return total[0], max_total[0], correct_per

# Initialize objects


hparams_config = {
    'layer1':[128,64,32],
    'layer2':[64,32,16],
    'layer3':[16,8,4],
    'lr':[0.03,0.003,0.001],
    'epochs':[3000,5000,10000],
    'activation':[nn.Tanhshrink, nn.ReLU],
}


data_loader = DataLoader('tweets_BTC.csv', 'BTC-USD.csv')
WRITERPATH = 'runs/{}/{}/{}/{}/{}/{}/{}'

experiment_num = 0
for l1 in hparams_config['layer1']:
    for l2 in hparams_config['layer2']:
        for l3 in hparams_config['layer3']:
            for lr in hparams_config['lr']:
                for epochs in hparams_config['epochs']:
                    for i, function in enumerate(hparams_config['activation']):
                        experiment_num += 1
                        function_dict = {0:'TanhShrink',1:'Relu'}
                        print(f'experiment_num: {experiment_num} | layer1:{l1} | layer2:{l2} | layer3:{l3} | epochs:{epochs} | lr:{lr} | activation:{function_dict[i]}')
                        model = PricePredictor(data_loader.X_train.shape[1],l1,l2,l3,function)
                        trainer = Trainer(model, data_loader, writer_path=WRITERPATH.format(experiment_num,l1,l2,l3,epochs,lr,function),epochs=epochs,lr=lr)
    # Train and evaluate
                        test_mse  = trainer.train()
                        correct_per,ratio_made = trainer.evaluate()
                        trainer.writer.add_hparams(
                            {
                                'layer 1':l1,
                                'layer 2': l2,
                                'layer 3': l3,
                                'learning rate': lr,
                                'epochs':epochs,
                                'function':function_dict[i]
                            },
                            {
                                "test MSE":test_mse,
                                'Percentage movement correct':correct_per,
                                'Ratio Made': ratio_made


                            }
                        )

                        
