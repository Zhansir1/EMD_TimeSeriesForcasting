import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class DauPredictionDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 sequence_length: int
                 ) -> None:
        self.dataframe = dataframe
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.dataframe) - self.sequence_length

    def __getitem__(self, index: int):
        end_index = index + self.sequence_length
        input_sequence = [[i] for i in self.dataframe.iloc[index: end_index]['dau'].values]
        target_value = self.dataframe.iloc[end_index]['dau']

        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        target_value = torch.tensor([target_value], dtype=torch.float32)

        return input_sequence, target_value


class DauPredictionDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size: int,
                 shuffle=False  # 对于时间序列数据不可以打乱数据
                 ):
        super(DauPredictionDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )


class Scaler:
    """
    data = pd.DataFrame({'value': [10, 20, 30, 40, 50, 60]})

    # 创建一个Scaler对象并拟合数据
    scaler = Scaler(feature_range=[0, 1])
    scaler.fit(data, 'value')

    # 对数据进行归一化
    scaled_data = scaler.transform(data['value'])

    # 还原归一化的数据
    original_data = scaler.inverse_transform(scaled_data)
    """
    def __init__(self, feature_range: list):
        self.feature_range = feature_range
        self.min_value = None
        self.max_value = None

    def fit(self, dataframe, column_name):
        # 获取指定列的最小值和最大值
        self.min_value = dataframe[column_name].min()
        self.max_value = dataframe[column_name].max()

    def transform(self, dataframe, column_name):
        if self.min_value is None or self.max_value is None:
            raise ValueError("Scaler must be fitted before transforming data.")

        # 对数据进行归一化
        scaled_data = (dataframe[column_name] - self.min_value) / (self.max_value - self.min_value)
        scaled_data = scaled_data * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data

    def inverse_transform(self, dataframe, column_name):
        if self.min_value is None or self.max_value is None:
            raise ValueError("Scaler must be fitted before inverting transformation.")

        # 还原归一化的数据
        original_data = (dataframe[column_name] - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        original_data = original_data * (self.max_value - self.min_value) + self.min_value
        return original_data

