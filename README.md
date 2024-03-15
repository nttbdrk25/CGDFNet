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

<u>**An example for training CGDFNet on Places365:**</u>

```
$ python CGDFNet_train_places365.py
```
Note: Subject to your system, modify these training files (*.py) to have the right path to dataset

**Validating the trained model of CGDFNet on Places365:**
```
$ python CGDFNet_train_places365.py --evaluate
```

Note: For instances of validation of CGDFNet, download the trained model of CGDFNet on Places365: [Click here](https://drive.google.com/drive/folders/1EdlA3tuOutBJMR23B-fcSOKKB69hAQ5R?usp=sharing). And then locate the downloaded file at ./checkpoints/[name_dataset]/small

**Related citations:**

If you use any materials, please cite the following relevant works.

```
@article{CGDFNetNguyen24,
  author       = {Thanh Tuan Nguyen, Hoang Anh Pham, and Thanh Phuong Nguyen},
  title        = {A light-weight backbone to adapt with extracting grouped dilation features},
  journal      = {Pattern Analysis and Applications},
  note         = {(submitted in 2024)}
}
```
