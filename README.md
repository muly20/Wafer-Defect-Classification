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
Using 80% of the (shuffled) dataset-- total of 38,015 samples, as the train set, SGD optimizer with learning rate of 0.01 and momentum of 0.9. I ran only 20 Epochs due to resource constraints.
I used binary crossentropy as the loss function and implemented an accuracy metric to work well on multilabel (more-than-)one-hot label vectors.

Evaluation
Accuracy on the test set came at 94.16%, and the average f-score accross classes came at 0.981, with 0.9375 as the lowest class value.
Accuracy across defect types compared to the 2020 paper of Juliang Wang (https://ieeexplore.ieee.org/document/9184890/):
![image](https://user-images.githubusercontent.com/96395197/151402550-8a9f91c3-d858-44d9-bcf6-ab0a24dd6a56.png)
![image](https://user-images.githubusercontent.com/96395197/151402566-dd064507-6697-493a-93c7-4c0c67fa3bd3.png)
![image](https://user-images.githubusercontent.com/96395197/151402589-8da9e955-f088-4fed-8d19-94d1afbb67e4.png)
![image](https://user-images.githubusercontent.com/96395197/151402603-6b5123a3-2786-4234-a4b2-80bb926df895.png)
