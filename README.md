# keras-conv2d
Construyendo una CNN con keras, fue hecha siguiendo el tutorial de [Keras Conv2D and Convolutional Layers](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/), se hicieron modificaciones para que use Keras directamente atravez de tensorflow.

## 🐍Requerimientos
* Python 3.x
* Tensorflow 1.x
* Numpy
* Matplotlib
* OpenCV 2
* imutils
* scikit-learn

## ¿Como usar?
Navega hasta el directorio del repositorio
```console
$ cd /path/to/keras-conv2d
```
Despues descarga y descomprime el dataset CALTECH-101
```console
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar -zxvf 101_ObjectCategories.tar.gz
```
Para entrenar y evaluar el modelo
```console
$ python train.py --dataset 101_ObjectCategories
```
