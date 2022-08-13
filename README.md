# cl-OCR (Command line OCR)

## What is this project?

This project aims to implement a multi-layer perceptron network in the C programming language, for speed and memory efficiency. Using this network implementation, the goal of this project is to re-create an OCR capable of recognizing hand-written digits with high accuracy, and low compute time.

The network is matrix based, calculations are performed via matrix operations, allowing for easy implementation of feed forward, backpropagation and gradient descent algorithms.

## MNIST dataset

The network was tested and trained with the MNIST hand-written digit dataset. The accuracy of the network has yet to be benchmarked.

To download the dataset, simply run the `download_data.sh` script.

## Running the project

* Compile the source code
```bash
make mkdir # Generates the bin/ and obj/ folders
           # Only needs to be run once.
make
```

* Run the binary
```bash
./bin/ocr
```

## Todo list

* [x] Design efficient memory based neural network structure
* [x] Implement matrix library
* [x] Implement feed forward algorithm
* [x] Implement backpropagation algorithm
* [x] Test neural network with basic XOR network
* [x] Add saving and loading capabilities for the network
* [x] Implement mini batch training in the network for prediction versatility
* [X] Implement dataset API to easily pass data into the network
* [x] Test neural network with OCR network trained to recognize handwritten digits
* [ ] Speed up matrix multiplication with multithread implementation (CPU or GPU)
* [ ] Implement basic cli interface to interact with the network