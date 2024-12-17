File Keras và label các bạn có thể lên Teachable Machine để train lại và điều chỉnh theo bài của mình nhé. Model mình train nó chưa được tốt lắm nhưng có thể lấy file để chạy Demo nha

Để chạy được các code này các bạn hãy import các thư viện liên quan nhé

pip install opencv-python

pip install numpy

pip install time

pip install random

pip install pygame

pip install tensorflow

❗Nếu chạy code mà bị lỗi: "TypeError: Error when deserializing class 'DepthwiseConv2D' using config={'name': 'expanded_conv_depthwise', 'trainable': True, 'dtype': 'float32', 'kernel_size': [3, 3], 'strides': [1, 1], 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': [1, 1], 'groups': 1, 'activation': 'linear', 'use_bias': False, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'bias_regularizer': None, 'activity_regularizer': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}}, 'depthwise_regularizer': None, 'depthwise_constraint': None}.
Exception encountered: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}"

thì mình đã khắc phục bằng cách sử dụng python 3.10.0 và tải tensorflow 2.12.0 

Mong là nó sẽ hữu ích😊
