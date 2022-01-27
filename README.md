# Wafer-Defect-Classification
Implementation of the 2017 Deformable Convolutional Network algorithm (Jifeng Dai et al. at https://arxiv.org/abs/1703.06211) on MixedWM38 wafer map defect dataset (https://www.kaggle.com/co1d7era/mixedtype-wafer-defect-datasets).

Defining and implementing a new tensorflow layer "DefConvLayer"

Notes:
  1.   the offset metrices are learned through a standard convolutional layer, with 2xN number of filters (=channels) corresponding to N 2D offsets *for each input pixel*.
  2.   There is a set of 2D offsets for each spatial location of the input feature map. from the 2017 paper: "The output offset fields have the same spatial resolution with the    input feature map." Meaning, it won't be enough to compute a static "offset" input "x_offset" and then applyting regular convolution, but each output pixel need to compute the kernel weightings on a different locations of input pixels/values. for that, I'll use a straight forward implementation of the deformable convolutional block.

  Note This is different than Juliang Wang's implementation on GitHub (https://github.com/Junliangwangdhu/WaferMap, https://ieeexplore.ieee.org/document/9184890/).
![image](https://user-images.githubusercontent.com/96395197/151399177-b4e8dd43-1113-4be1-bfb8-d086435090a3.png)
![image](https://user-images.githubusercontent.com/96395197/151399311-73cb3b33-5a2b-47cd-9bd7-2dacffa8ac68.png)

  3.   Since the offsets are fractional values, the values of the input at the offset location will be "estimated" using [Bi-linear Interpolation].

Model Structure
I used 4 blocks of Deformable convolutional Layers with increasing channels (32, 64, 128, 128), with each followed by Batch Normalization and a ReLU activation. Classification block consisted of Global Average Pooling layer and a Fully connected layer with 8 output neurons each activated using a sigmoid.

Training
Using 80% of the (shuffled) dataset-- total of 38,015 samples, as the train set, SGD optimizer (as in Juliang Wang's paper) with 0.01 learning rate and 0.9 for momentum. I ran only 20 Epochs due to resource constraints.
I used binary crossentropy as the loss function and implemented an accuracy metric to work well on multilabel (more-than-)one-hot label vectors.

Evaluation
Accuracy on the test set came at 94.16%, and the average f-score accross classes came at 0.981, with 0.9375 as the lowest class value.
Accuracy across defect types compared to the 2020 paper of Juliang Wang (https://ieeexplore.ieee.org/document/9184890/):
![image](https://user-images.githubusercontent.com/96395197/151401819-21b1ae5b-e510-40d8-bdcc-6f21cdc3bb1d.png)
![image](https://user-images.githubusercontent.com/96395197/151401843-a3e91721-c98f-416a-b3d1-36a2e84b886a.png)
![image](https://user-images.githubusercontent.com/96395197/151401870-c3e655c6-b378-440e-93d5-08dc12543d8f.png)
![image](https://user-images.githubusercontent.com/96395197/151401882-6bddebf5-0e43-4626-84d1-4df9baecf8fd.png)
