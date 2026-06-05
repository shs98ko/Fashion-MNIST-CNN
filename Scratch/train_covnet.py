# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist
from Scratch.common.simple_convnet import SimpleConvNet
from common.trainer import Trainer

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 25
loss_per_epoch = []

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 32, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=150, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch = 1000)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# Accuracy 그래프
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

iter_per_epoch = int(trainer.iter_per_epoch)
for i in range(0, len(trainer.train_loss_list), iter_per_epoch):
    loss_per_epoch.append(
        np.mean(trainer.train_loss_list[i:i+iter_per_epoch])
    )

# Loss 그래프
x = np.arange(len(loss_per_epoch))
plt.plot(x, loss_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1.0)
plt.title("Training Loss per Epoch")
plt.grid()
plt.show()

