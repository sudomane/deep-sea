# cl-OCR (Command line OCR)

## What is this project?

This project aims to implement a multi-layer perceptron network in the C programming language, for speed and memory efficiency. Using this network implementation, the goal of this project is to re-create an OCR capable of recognizing hand-written digits with high accuracy, and low compute time.

The network is matrix based, calculations are performed via matrix operations, allowing for easy implementation of feed forward, backpropagation and gradient descent algorithms.

This project is still a work in progress.

## Running the project

* Compile the source code
```bash
make mkdir # Only needs to be executed once
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
* [ ] Add saving and loading capabilities for the network
* [ ] Refactor O(n^3) matrix multiplication to O(n^1) with multithread implementation
* [ ] Implement mini batch training in the network for prediction versatility
* [ ] Test neural network with OCR network trained to recognize handwritten digits
* [ ] Implement basic cli interface to interact with the network