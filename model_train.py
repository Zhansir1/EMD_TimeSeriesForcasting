import torch
import torch.nn as nn
from data_preprocessing import *
from model_evaluation import *
from plot_function import *


class RNNModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int
                 ):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            batch_first=True,
            nonlinearity='relu'
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # out 的形状将是 (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])  # 获取RNN的最后一个时间步的输出
        return out


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int
                 ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出进行预测
        return out


class GRUModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int
                 ):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出进行预测
        return out


def train_rnn_model(
        dataloader,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_epochs: int,
        learning_rate=0.001
                ):
    # 创建模型
    model = RNNModel(input_size, hidden_size, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # 将数据传递给模型
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    print('RNN Model Training completed.')
    return model


def train_lstm_model(
        dataloader,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_epochs: int,
        learning_rate=0.001
                ):
    # 创建模型
    model = LSTMModel(input_size, hidden_size, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # 将数据传递给模型
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    print('LSTM Model Training completed.')
    return model


def train_gru_model(
        dataloader,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_epochs: int,
        learning_rate=0.001
                ):
    # 创建模型
    model = GRUModel(input_size, hidden_size, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # 将数据传递给模型
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    print('GRU Model Training completed.')
    return model


def imf_train(imf_dataframe,
              window_size: int,
              model_name: str,
              graph_index: int,
              y_lim: tuple,
              batch_size: int = 16
              ):
    sequence_length = window_size

    dataset = DauPredictionDataset(dataframe=imf_dataframe, sequence_length=sequence_length)
    dataloader = DauPredictionDataLoader(dataset=dataset, batch_size=batch_size)

    if model_name == 'RNN':
        model = train_rnn_model(dataloader=dataloader,
                                input_size=1,
                                hidden_size=4,
                                output_size=1,
                                num_epochs=200)
    elif model_name == 'LSTM':
        model = train_lstm_model(dataloader=dataloader,
                                 input_size=1,
                                 hidden_size=64,
                                 output_size=1,
                                 num_epochs=200)
    else:
        model = train_gru_model(dataloader=dataloader,
                                input_size=1,
                                hidden_size=64,
                                output_size=1,
                                num_epochs=200)

    # 根据训练的模型预测出训练集的预测值，然后滚动预测测试集
    train_pred, train_true = train_model_res([0, 273 - sequence_length], dataset=dataset, model=model,
                                             window_size=sequence_length)
    test_pred = test_model(model=model, dataset=dataset, window_size=sequence_length,
                           start=273 - sequence_length, end=301 - sequence_length)

    line_args = [
        {'x': list(range(len(imf_dataframe.dau))), 'y': imf_dataframe.dau, 'label': 'real', 'color': 'c'},
        {'x': list(range(sequence_length, len(train_pred) + sequence_length)), 'y': train_pred, 'label': 'train_pred',
         'color': 'blue'},
        {'x': list(range(273, 302)), 'y': test_pred, 'label': 'test_pred', 'color': 'red'}
    ]
    line_plot((10, 8), line_args, y_lim=y_lim, y_label='DAU', title=f'{model_name}模型 IMF{str(graph_index)} 预测效果')
    return train_pred, test_pred
