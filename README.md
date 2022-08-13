# C Neural Network

## What is this project?

This project implements a multi-layer perceptron neural network architecture in the C programming language.

The network has been tested as an OCR to recognize hand-written digits from the MNIST dataset.

The network is matrix based, calculations are performed via matrix operations, allowing for easy implementation of feed forward, backpropagation and gradient descent algorithms.

## MNIST dataset

The network was tested and trained with the MNIST hand-written digit dataset. The accuracy of the network has yet to be benchmarked.

To download the dataset, simply run the `download_data.sh` script.

## Running the project

No additional dependencies are required to compile the project.

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
* [ ] Speed up matrix multiplication with multithread implementation (CPU or GPU)
* [ ] Implement basic cli interface to interact with the network