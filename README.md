# cOCR

## What is this project?

This project implements a multi-layer perceptron neural network architecture in the C programming language.

The network has been tested as an OCR, and was able to accurately recognize hand-written digits from the MNIST dataset.

The network is matrix based, calculations are performed via matrix operations, allowing for easy implementation of feed forward, backpropagation and gradient descent algorithms.

## Network accuracy
The network was tested with the hand-written digits from the MNIST dataset, and has achieved an accuarcy of 82% on a test set of 10000 images, while only being trained on 4096 images out of the 60000 total images of the MNIST dataset, due to CPU limitations. Plans to implement CPU/GPU acceleration are currently a work in progress.

![](https://i.imgur.com/xa4Z45A.png)

## MNIST dataset

To test the network with the dataset, simply run the `download_data.sh` script to download the MNIST dataset, and execute the ocr binary.

## Running the project

No additional dependencies are required to compile the project.

* Compile the source code
```bash
make mkdir # Generates the bin/ and obj/ folders
           # Only needs to be run once.
make
```

* Execute the binary
```bash
./bin/ocr [network.save]
```

## Roadmap

* [x] Design efficient memory based neural network structure
* [x] Implement matrix library
* [x] Implement feed forward algorithm
* [x] Implement backpropagation algorithm
* [x] Test neural network with basic XOR network
* [x] Add saving and loading capabilities for the network
* [x] Implement mini batch training in the network for prediction versatility
* [X] Implement dataset API to easily pass data into the network
* [x] Test neural network with OCR network trained to recognize handwritten digits
* [ ] Parallelize gradient descent with threads
* [ ] Implement basic cli interface to interact with the network
