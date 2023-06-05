import torch
from torch import nn
from torch.testing import assert_allclose

def test_moving_avg_forward():
    kernel_size = 3
    stride = 1
    moving_avg_block = moving_avg(kernel_size, stride)
    
    x = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
    expected_output = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float32)

    output = moving_avg_block(x)
    
    assert_allclose(output, expected_output)

# Run the test
test_moving_avg_forward()
####################################################################################################
import torch
from torch import nn
from torch.testing import assert_allclose

def test_series_decomp_forward():
    kernel_size = 3
    series_decomp_block = series_decomp(kernel_size)
    
    x = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
    expected_residuals = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    expected_moving_mean = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float32)

    residuals, moving_mean = series_decomp_block(x)
    
    assert_allclose(residuals, expected_residuals)
    assert_allclose(moving_mean, expected_moving_mean)

# Run the test
test_series_decomp_forward()
#############################################################################################
import torch
from torch import nn
from torch.testing import assert_allclose

def test_model_forward():
    configs = MockConfigs()
    model = Model(configs)

    x = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
    expected_output_shape = (1, configs.pred_len, configs.enc_in)

    output = model(x)
    
    assert output.shape == expected_output_shape

# Define a mock class to mimic the Configs object
class MockConfigs:
    def __init__(self):
        self.seq_len = 5
        self.pred_len = 3
        self.individual = True
        self.enc_in = 1

# Run the test
test_model_forward()
############################################################################################
import argparse
from torch.optim import SGD

def test_adjust_learning_rate():
    optimizer = SGD([])
    epoch = 5
    args = argparse.Namespace(learning_rate=0.1, lradj='6')

    adjust_learning_rate(optimizer, epoch, args)
    expected_lr = 0.1 if epoch < 5 else 0.01

    for param_group in optimizer.param_groups:
        assert param_group['lr'] == expected_lr

# Run the test
test_adjust_learning_rate()
##################################################################################
import numpy as np
import torch

def test_early_stopping():
    early_stopping = EarlyStopping(patience=2, verbose=True, delta=0.1)
    val_loss = 0.2
    model = torch.nn.Linear(10, 1)
    path = './'

    early_stopping(val_loss, model, path)
    assert early_stopping.counter == 0
    assert early_stopping.best_score == -val_loss
    assert early_stopping.val_loss_min == val_loss

    early_stopping(val_loss + 0.2, model, path)
    assert early_stopping.counter == 1
    assert not early_stopping.early_stop

    early_stopping(val_loss + 0.3, model, path)
    assert early_stopping.counter == 2
    assert not early_stopping.early_stop

    early_stopping(val_loss + 0.4, model, path)
    assert early_stopping.counter == 0
    assert not early_stopping.early_stop
    assert early_stopping.best_score == -val_loss - 0.4
    assert early_stopping.val_loss_min == val_loss + 0.4

    early_stopping(val_loss + 0.3, model, path)
    assert early_stopping.counter == 1
    assert not early_stopping.early_stop

    early_stopping(val_loss + 0.2, model, path)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop

# Run the test
test_early_stopping()
############################################################################
import numpy as np

def test_RSE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_rse = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

    assert RSE(pred, true) == expected_rse

def test_CORR():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_corr = 0.01 * (((true - true.mean(0)) * (pred - pred.mean(0))).sum(0) /
                           np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0) + 1e-12))

    assert CORR(pred, true) == expected_corr

def test_MAE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_mae = np.mean(np.abs(pred - true))

    assert MAE(pred, true) == expected_mae

def test_MSE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_mse = np.mean((pred - true) ** 2)

    assert MSE(pred, true) == expected_mse

def test_RMSE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_rmse = np.sqrt(np.mean((pred - true) ** 2))

    assert RMSE(pred, true) == expected_rmse

def test_MAPE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_mape = np.mean(np.abs((pred - true) / true))

    assert MAPE(pred, true) == expected_mape

def test_MSPE():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_mspe = np.mean(np.square((pred - true) / true))

    assert MSPE(pred, true) == expected_mspe

def test_metric():
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([2, 3, 4, 5, 6])
    expected_mae = np.mean(np.abs(pred - true))
    expected_mse = np.mean((pred - true) ** 2)
    expected_rmse = np.sqrt(expected_mse)
    expected_mape = np.mean(np.abs((pred - true) / true))
    expected_mspe = np.mean(np.square((pred - true) / true))
    expected_rse = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
    expected_corr = 0.01 * (((true - true.mean(0)) * (pred - pred.mean(0))).sum(0) /
                            np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0) + 1e-12))

    assert metric(pred, true) == (expected_mae, expected_mse, expected_rmse, expected_mape, expected_mspe,
                                  expected_rse, expected_corr)

# Run the tests
test_RSE()
test_CORR()
test_MAE()
test_MSE()
test_RMSE()
test_MAPE()
test_MSPE()
test_metric()
########################################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_Dataset_Custom():
    root_path = '/path/to/dataset'
    flag = 'train'
    size = [96, 16, 16]
    features = 'S'
    data_path = 'ETTh1.csv'
    target = 'OT'
    scale = True
    timeenc = 0
    freq = 'h'
    train_only = False

    dataset = Dataset_Custom(root_path, flag, size, features, data_path, target, scale, timeenc, freq, train_only)

    # Test __init__()
    assert dataset.seq_len == size[0]
    assert dataset.label_len == size[1]
    assert dataset.pred_len == size[2]
    assert dataset.set_type == 0
    assert dataset.features == features
    assert dataset.target == target
    assert dataset.scale == scale
    assert dataset.timeenc == timeenc
    assert dataset.freq == freq
    assert dataset.train_only == train_only
    assert dataset.root_path == root_path
    assert dataset.data_path == data_path

    # Test __len__()
    assert len(dataset) == len(dataset.data_x) - dataset.seq_len - dataset.pred_len + 1

    # Test __getitem__()
    index = 0
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset.__getitem__(index)
    assert np.array_equal(seq_x, dataset.data_x[index:index+dataset.seq_len])
    assert np.array_equal(seq_y, dataset.data_y[index+dataset.label_len:index+dataset.seq_len+dataset.label_len+dataset.pred_len])
    assert np.array_equal(seq_x_mark, dataset.data_stamp[index:index+dataset.seq_len])
    assert np.array_equal(seq_y_mark, dataset.data_stamp[index+dataset.label_len:index+dataset.seq_len+dataset.label_len+dataset.pred_len])

    # Test inverse_transform()
    data = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    expected_inverse = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(dataset.inverse_transform(data), expected_inverse)

# Run the test
test_Dataset_Custom()
############################################################################################################
from argparse import Namespace
from torch.utils.data import DataLoader

def test_data_provider():
    args = Namespace(
        data='ETTh1',
        embed='timeF',
        train_only=False,
        batch_size=32,
        freq='h',
        root_path='/path/to/root',
        data_path='ETTh1.csv',
        seq_len=96,
        label_len=16,
        pred_len=16,
        features='S',
        target='OT',
        num_workers=4
    )
    flag = 'train'

    data_set, data_loader = data_provider(args, flag)

    # Test data_set initialization
    assert data_set.seq_len == args.seq_len
    assert data_set.label_len == args.label_len
    assert data_set.pred_len == args.pred_len
    assert data_set.set_type == 0
    assert data_set.features == args.features
    assert data_set.target == args.target
    assert data_set.timeenc == 1
    assert data_set.freq == args.freq
    assert data_set.train_only == args.train_only
    assert data_set.root_path == args.root_path
    assert data_set.data_path == args.data_path

    # Test data_loader initialization
    assert data_loader.batch_size == args.batch_size
    assert data_loader.shuffle == True
    assert data_loader.num_workers == args.num_workers
    assert data_loader.drop_last == True

    flag = 'test'
    data_set, data_loader = data_provider(args, flag)

    # Test data_loader for 'test' flag
    assert data_loader.batch_size == args.batch_size
    assert data_loader.shuffle == False
    assert data_loader.num_workers == args.num_workers
    assert data_loader.drop_last == False

    flag = 'pred'
    data_set, data_loader = data_provider(args, flag)

    # Test data_loader for 'pred' flag
    assert data_loader.batch_size == 1
    assert data_loader.shuffle == False
    assert data_loader.num_workers == args.num_workers
    assert data_loader.drop_last == False

# Run the test
test_data_provider()
########################################################################################
import unittest
from unittest.mock import Mock
from your_module import Exp_Main

class TestExpMain(unittest.TestCase):
    def setUp(self):
        self.args = Mock()
        self.exp_main = Exp_Main(self.args)

    def test__build_model(self):
        # Implement test case for _build_model method
        pass

    def test__get_data(self):
        # Implement test case for _get_data method
        pass

    def test__select_optimizer(self):
        # Implement test case for _select_optimizer method
        pass

    def test__select_criterion(self):
        # Implement test case for _select_criterion method
        pass

    def test_vali(self):
        # Implement test case for vali method
        pass

    def test_train(self):
        # Implement test case for train method
        pass

    def test_test(self):
        # Implement test case for test method
        pass

    def test_predict(self):
        # Implement test case for predict method
        pass

if __name__ == '__main__':
    unittest.main()
