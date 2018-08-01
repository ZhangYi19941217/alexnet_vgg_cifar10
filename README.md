**NOTE: For users interested in multi-GPU, we recommend looking at the newer [cifar10_estimator](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) example instead.**

---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

数据集的下载，可直接运行cifar_train.py完成数据集下载，训练的要求。
注意修改：cifar10中关于数据集的存放位置
cifar10_eval_alexnet.py  # Alexnet的验证
cifar10_alexnet.py  # Alexnet的网络结构
cifar10_train_alexnet.py  # 针对cifar10的Alexnet的训练
cifar10_input.py  # 输入数据
cifar10_eval_vgg.py  # vgg的验证
cifar10_vgg.py  # vgg的网络结构
cifar10_train_vgg.py  # 针对cifar10的vgg的训练
