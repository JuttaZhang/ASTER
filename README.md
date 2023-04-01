# [TNNLS2023] This code is the implementation of ASTER
[Adaptive Filter Pruning via Sensitivity Feedback](https://ieeexplore.ieee.org/document/10064249)

Author: Yuyao Zhang and Nikolaos M. Freris

The paper is accepted at the IEEE Transactions of Neural Networks and Learning Systems (TNNLS), 2023


## Usage
First, clone the repository locally:
```
git clone 
```
Then, install PyTorch == 1.9.1+cu111 and torchvision == 0.10.1+cu111 and ax-platform == 0.2.4:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install ax-platform == 0.2.4
```

For VGG-16 and CIFAR-10, run:
```
python BayesianMain.py \\
--model vgg16 --depth 16 --s 1e-4 --exp_flops 0.4 --batch_size 64 --test-batch-size 128 --epochs 320 --pec 0.5 --lb 0.7 --ub 1 
```




## Publications
If you find this repository useful for your research, please cite the following paper:
```
@article{zhang2023adaptive,
  title={Adaptive Filter Pruning via Sensitivity Feedback},
  author={Zhang, Yuyao and Freris, Nikolaos M},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
