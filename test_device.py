import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

# Hoặc với PyTorch
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# import xgboost as xgb
#
# print(xgb.__version__)
#
# # Kiểm tra GPU có hoạt động không
# params = {'tree_method': 'hist', 'device': 'cuda'}
# dtrain = xgb.DMatrix([[1, 2, 3], [4, 5, 6]], label=[0, 1])
# bst = xgb.train(params, dtrain, num_boost_round=1)
# print("XGBoost is using GPU")
#
# import numpy as np
# import lightgbm as lgb
#
# print(lgb.__version__)
#
# # Create sample data
# data = np.array([[1, 2, 3], [4, 5, 6]])
# label = np.array([0, 1])
#
# # Create LightGBM dataset
# dataset = lgb.Dataset(data, label=label)
#
# # Configure to use GPU
# params = {
#     'objective': 'binary',
#     'device': 'gpu',
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0
# }
#
# # Train the model
# gbm = lgb.train(params, dataset, num_boost_round=1)
# print("LightGBM is using GPU")
#
# import catboost as cb
#
# print(cb.__version__)
#
# # Kiểm tra hỗ trợ GPU
# try:
#     model = cb.CatBoostClassifier(task_type='GPU', devices='0:0')
#
#     # Tạo dữ liệu mẫu
#     train_data = [[1, 2, 3], [4, 5, 6]]
#     train_labels = [0, 1]
#
#     # Huấn luyện model
#     model.fit(train_data, train_labels, verbose=False)
#     print("CatBoost đang sử dụng GPU")
# except Exception as e:
#     print(f"GPU không khả dụng: {e}")
