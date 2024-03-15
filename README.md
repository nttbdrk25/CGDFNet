# CGDFNet: A light-weight backbone to adapt with extracting grouped dilation features

**Abstract:**

* Addressing grouped dilation features (GDFs) improved the learning ability of
MobileNetV1 in image representation. However, the computational complexity is
still at a high level, while the performance is a modest degree. This expensive cost
is principally caused by the backbone of MobileNetV1 taking deep feature maps in
several latest layers. To mitigate these issues, we propose a light-weight network
(called CGDF-Net) with an adaptative architecture to extract grouped dilation
features in more effect. CGDF-Net is structured by two main contributions: i)
Its backbone is formed by simply replacing several latest layers of MobileNetV1
with a pointwise convolutional layer for reducing the computational complexity;
ii) Embedding an attention mechanism into the GDF block to form a completed
GDF perceptron (CGDF) that directs the learning process into the significant
properties of objects in images instead of the trivial ones. Experimental results
on benchmark datasets for image recognition have validated that the proposed
CGDF-Net network obtained good performance with small computational cost in
comparison with MobileNets and other light-weight models.

<u>**Training TickNets on Datasets:**</u>

For Stanford Dogs. Note that it will automatically run for all TickNets, i.e., TickNet-basic, TickNet-small and TickNet-large
```
$ python TickNet_Dogs.py
```
For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small
```
$ python TickNet_ImageNet.py -a small
$ python TickNet_Places365.py -a small 
```
Note: Subject to your system, modify these training files (*.py) to have the right path to datasets

**Validating the trained models of TickNets:**
* For Stanford Dogs (TickNet-small)
```
$ python TickNet_Dogs.py --evaluate
```
* For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small.
```
$ python TickNet_ImageNet.py -a small --evaluate
$ python TickNet_Places365.py -a small --evaluate
```

Note: For instances of validation of TickNet-small, download the trained model of TickNet-small on Datasets: [Click here for Places365](https://drive.google.com/drive/folders/1EdlA3tuOutBJMR23B-fcSOKKB69hAQ5R?usp=sharing); [Click here for ImageNet-1k](https://drive.google.com/drive/folders/1t1M_QJwCmcaTgKBsJBmzrU-kabQeOPDT?usp=sharing); [Click here for Stanford Dogs](https://drive.google.com/drive/folders/1RGglukdrd5xDrGSo6ONmHTCZNZ-YwpZb?usp=sharing). And then locate the downloaded file at ./checkpoints/[name_dataset]/small

**Related citations:**

If you use any materials, please cite the following relevant works.

```
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  note         = {(submitted in 2023)}
}
```
