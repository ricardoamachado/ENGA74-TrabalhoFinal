import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple
import re
from pathlib import Path

# Função para ler arquivos CSV de uma pasta e concatenar em um DataFrame
def read_csv_folder(input_path: Path) -> pd.DataFrame:
    """
    Lê todos os arquivos CSV na pasta especificada e concatena em um único DataFrame.
    """
    df_full = pd.DataFrame()
    
    if not input_path.exists():
        raise FileNotFoundError(f"O diretório {input_path} não existe.")
    
    if not any(input_path.glob('*.csv')):
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado no diretório {input_path}.")
    for file in input_path.glob('*.csv'):
        print(f"Processando arquivo: {file.name}")
        city_regex = r'(\w+)_\d{4}.csv'
        match = re.match(city_regex, file.name)
        city_name = match.group(1) if match else 'Desconhecido'
        print(f"Nome da cidade: {city_name}")
        df = pd.read_csv(file, skiprows=2)
        df['City'] = city_name
        df_full = pd.concat([df_full, df], ignore_index=True)
    return df_full
#Funções de pré-processamento dos dados
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['Minute','Cloud Type', 'Ozone','Solar Zenith Angle', 'Surface Albedo',
                    'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI', 'DNI', 'Fill Flag', 'Cloud Fill Flag', 'Aerosol Optical Depth','Alpha','SSA','Asymmetry'], axis = 1)
    df = df.dropna()
    # Combine year, month, day, hour into a single datetime column
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    # Extract useful time-based features
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['HourOfDay'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    # Optionally, drop original columns if not needed
    df = df.drop(['Month', 'Day', 'Hour', 'Datetime'], axis=1)
    return df

def data_scaling_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    # Codificar a coluna 'City' usando One-Hot Encoding
    city_encoded = one_hot_encoder.fit_transform(df[['City']])
    city_encoded_df = pd.DataFrame(city_encoded, columns=one_hot_encoder.get_feature_names_out(['City']), index=df.index)
    # Concatenar as colunas codificadas com o restante do DataFrame (exceto 'City')
    df_no_city = df.drop(['City'], axis=1)
    df_all = pd.concat([df_no_city, city_encoded_df], axis=1)
    scaler = MinMaxScaler()
    df_all['DayOfYear'] = np.sin(df_all['DayOfYear'] * (2 * np.pi / 365))  # Normalização do dia do ano
    df_all['HourOfDay'] = np.sin(df_all['HourOfDay'] * (2 * np.pi / 24))  # Normalização da hora do dia
    columns_to_scale = [col for col in df_all.columns if col not in ['Year', 'DayOfYear', 'HourOfDay']]
    df_all[columns_to_scale] = scaler.fit_transform(df_all[columns_to_scale])
    # Uso dos dados até 2023 para treino e 2024 para teste.
    df_train = df_all[df_all['Year'] <= 2022]
    df_test = df_all[df_all['Year'] >= 2023]
    #Remove a coluna 'Year' dos dataframes.
    df_train = df_train.drop(['Year'], axis=1)
    df_test = df_test.drop(['Year'], axis=1)
    # Reset index for all dataframes
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_test
# Implementação dos modelos de redes neurais.
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # Se hidden não for fornecido, inicializa com zeros
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        # Passa pela GRU
        out, hidden = self.gru(x, hidden)
        
        # Aplica a camada linear apenas na última saída temporal
        out = self.fc(out[:, -1, :])  # Pega apenas o último timestep
        
        return out
    
    def init_hidden(self, batch_size):
        # Inicializa o estado oculto com zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        else:
            h_0, c_0 = hidden

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
# Define a função de perda RMSE.
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# Função para treinamento do modelo.
def train_model(model, train_loader, test_loader, epochs, lr, patience=15, weight_decay=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = RMSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

    best_test_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    train_mae = []
    test_mae = []
    train_r2 = []
    test_r2 = []
    best_model_state = None

    for epoch in range(epochs):
        # Treino
        model.train()
        train_loss = 0
        train_mae_epoch = 0
        y_true_train = []
        y_pred_train = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs.squeeze(), y_batch)
            mae = mae_criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_mae_epoch += mae.item()
            y_true_train.append(y_batch.detach().cpu())
            y_pred_train.append(outputs.detach().cpu())

        y_true_train = torch.cat(y_true_train)
        y_pred_train = torch.cat(y_pred_train)
        ss_res_train = torch.sum((y_true_train - y_pred_train) ** 2)
        ss_tot_train = torch.sum((y_true_train - torch.mean(y_true_train)) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train != 0 else torch.tensor(0.0)
        train_r2.append(r2_train.item())

        # Teste
        model.eval()
        test_loss = 0
        test_mae_epoch = 0
        y_true_test = []
        y_pred_test = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs.squeeze(), y_batch).item()
                test_mae_epoch += mae_criterion(outputs.squeeze(), y_batch).item()
                y_true_test.append(y_batch.cpu())
                y_pred_test.append(outputs.cpu())

        y_true_test = torch.cat(y_true_test)
        y_pred_test = torch.cat(y_pred_test)
        ss_res_test = torch.sum((y_true_test - y_pred_test) ** 2)
        ss_tot_test = torch.sum((y_true_test - torch.mean(y_true_test)) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test if ss_tot_test != 0 else torch.tensor(0.0)
        test_r2.append(r2_test.item())

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_mae_epoch /= len(train_loader)
        test_mae_epoch /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_mae.append(train_mae_epoch)
        test_mae.append(test_mae_epoch)

        scheduler.step(test_loss)

        # Early stopping baseado no teste
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na época {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f'Época {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Train MAE: {train_mae_epoch:.6f}, Test MAE: {test_mae_epoch:.6f}, Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses, train_mae, test_mae, train_r2, test_r2, best_test_loss

# Função para plotar as perdas.
def plot_losses(train_losses, test_losses):
    """Plota as perdas de treino e teste """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='green')
    plt.title('Losses durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    sns.set_theme(style='whitegrid')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df  = read_csv_folder(Path('data/'))
    df = data_preprocessing(df)
    df_train, df_test = data_scaling_split(df)
    ghi_values_train = torch.tensor(df_train['GHI'].values, dtype=torch.float32).unsqueeze(-1)
    ghi_values_test = torch.tensor(df_test['GHI'].values, dtype=torch.float32).unsqueeze(-1)

    X_train = ghi_values_train[:-1].unsqueeze(1)  # shape: (N-1, 1, 1)
    y_train = ghi_values_train[1:].squeeze(-1)          # shape: (N-1, 1)
    X_test = ghi_values_test[:-1].unsqueeze(1)  # shape: (N-1, 1, 1)
    y_test = ghi_values_test[1:].squeeze(-1)          # shape: (N-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[-1]
    mlp_model = SimpleMLP(input_size=input_size, hidden_size=16, output_size=1, num_layers = 2).to(device)
    gru_model = SimpleGRU(input_size=input_size, hidden_size=24, output_size=1, num_layers=1).to(device)
    lstm_model = SimpleLSTM(input_size=input_size, hidden_size=21, output_size=1, num_layers=2).to(device)
    train_losses_mlp, test_losses_mlp, train_mae_mlp, test_mae_mlp, train_r2_mlp, test_r2_mlp, best_test_loss_mlp = train_model(
        model=mlp_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        lr=9.53732587571571e-05,
        patience=15,
        weight_decay=4.6738485421988906e-05
    )
    train_losses_gru, test_losses_gru, train_mae_gru, test_mae_gru, train_r2_gru, test_r2_gru, best_test_loss_gru = train_model(
        model=gru_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        lr=3.977744609342973e-05,
        patience=15,
        weight_decay=0.0032407585068355014
    )
    train_losses_lstm, test_losses_lstm, train_mae_lstm, test_mae_lstm, train_r2_lstm, test_r2_lstm, best_test_loss_lstm = train_model(
        model=lstm_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        lr=0.001795777366159885,
        patience=15,
        weight_decay=1.9248472309738666e-06
    )
    print(f"Melhor perda de teste MLP: {best_test_loss_mlp:.6f}")
    print(f"Melhor perda de teste GRU: {best_test_loss_gru:.6f}")
    print(f"Melhor perda de teste LSTM: {best_test_loss_lstm:.6f}")
    plot_losses(train_losses_mlp, test_losses_mlp)
    plot_losses(train_losses_gru, test_losses_gru)
    plot_losses(train_losses_lstm, test_losses_lstm)


if __name__ == "__main__":
    main()
