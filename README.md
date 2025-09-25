# README

Score SDE models implemented by Jittor framework.

*This verison seemed able to run, but there may be many problems now.*

## UPDATE LOG

- **2025 Sep 24** Now it can carry out a whole process of training DDPM model.

## SOLUTIONS OF POSSIBLE ISSUES ***(Something maybe helpful)***

### Running Command

```
python your_menu/main.py \
    --config your_menu/configs/ve/cifar10_ddpm.py \
    --workdir your_menu/train_dir \
    --mode train
```

Replace `your_menu` by your root directory of the model.

### 1

I suggest to download Jittor in Ubuntu. Please refer to https://www.cnblogs.com/Yi-blogs/p/18881246 for detailed advice.

### 2

Jittor needs `cupy` to do GPU computing. I used command `conda install cupy`, which is more convenient than `pip`. But you may downgrade your `numpy` to `<1.24`.

### 3

I don't know how to use Jittor-style 'tensorboard', so I use torch's tensorboard. If you do so, remember to convert `loss` from `jt.Var` to `numpy` or `torch.tensor`.

### 4

For convenience, I use `tensorflow.datasets` to import the `CIFAR-10`.
When using datasets from tensorflow, the data type of images are `tf.tensor`, which need to turned to numpy array by `.numpy()`.

### 5

When you try to convert `jt.Var` as the following way:

```
a = jt.Var([1, 2, 3]).to("cpu", jt.uint8).numpy()
```

You'd be careful because above `a` has type `int32`! Which is different from similar syntax in torch.

I think it is a bug of Jittor, detailed description is in the [issue](https://github.com/Jittor/jittor/issues/661).

