#!/bin/bash

mkdir data

cd data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip -d train-images-idx3-ubyte.gz
gunzip -d train-labels-idx1-ubyte.gz
gunzip -d t10k-images-idx3-ubyte.gz
gunzip -d t10k-labels-idx1-ubyte.gz

echo "Done."

cd ..
