# ISSUES LOG ***(Something maybe helpful)***

### Running Command
```
python /home/a516/score_sde_jittor/main.py \
    --config /home/a516/score_sde_jittor/configs/ve/cifar10_ddpm.py \
    --workdir /home/a516/score_sde_jittor/train_dir \
    --mode train
```
### 1
I suggest to download Jittor in Ubuntu. Please refer to https://www.cnblogs.com/Yi-blogs/p/18881246 for detailed advice.

### 2
Jittor needs `cupy` to do GPU computing. I used command `conda install cupy`, which is more convenient than `pip`. But you may downgrade your `numpy` to `<1.24`.

### 3
I don't know how to use Jittor-style 'tensorboard', so I use torch's tensorflow. If you do so, remember to convert `loss` from `jt.Var` to `numpy` or `torch.tensor`.

### 4
For convenience, I use `tensorflow.datasets` to import the `CIFAR-10`.
When using datasets from tensorflow, the data type of images are `tf.tensor`, which need to turned to numpy array by `.numpy()`.

