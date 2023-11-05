import numpy as np
import torch


def mse(pred_list, true_list) -> np.float64:
    pred_list = np.reshape(np.array(pred_list), (-1, ))
    true_list = np.reshape(np.array(true_list), (-1, ))
    return np.mean((pred_list - true_list) ** 2)


def train_model_res(data_range: list,
                    window_size: int,
                    dataset,
                    model
                    ):  # 不包括右区间
    train_predicted_results = []
    real_results = []

    with torch.no_grad():
        for index in range(data_range[0], data_range[1]):
            inputs, real_label = dataset[index][0].view(1, window_size, 1), dataset[index][1]
            outputs = model(inputs)
            train_predicted_results.append(outputs.detach().numpy()[0][0])
            real_results.append(real_label.detach().numpy()[0])

    return train_predicted_results, real_results


def test_model(model,
               dataset,
               window_size: int,
               start: int,
               end: int
               ) -> list:
    pred_res = []
    inputs = dataset[start][0].view(1, window_size, 1)
    for index in range(start, end + 1):
        pred = model(inputs)
        pred_res.append(pred.detach().numpy()[0][0])
        prev, next = inputs[:, 1: window_size, :], torch.tensor([[[pred]]])
        inputs = torch.cat((prev, next), dim=1)

    return pred_res

