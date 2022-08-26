# DeepSea

## What is DeepSea?

DeepSea is a play on words. I originally intended to name this "DeepC", however that name was taken, so I chose the next best thing.

DeepSea is an experimental deep learning framework for the C programming language.

The framework's network is matrix based, calculations are performed via matrix operations, allowing for easy implementation of feed forward, backpropagation and gradient descent algorithms.

## Network accuracy
An OCR was implemented and tested with hand written digits from the MNIST dataset, and has achieved an accuarcy of 82% on a test set of 10000 images, while only being trained on 4096 images out of the 60000 total images of the MNIST dataset, due to CPU limitations. Plans to implement CPU/GPU acceleration are currently a work in progress.

![](https://i.imgur.com/xa4Z45A.png)

## MNIST dataset

To test the network with the dataset, simply execute the `fetch_data` script to download the MNIST dataset, and execute the ocr binary. Realistically, the framework can be tested on any MNIST dataset.

## Running the project

No additional dependencies are required to compile the project.

* Compile the source code
```bash
make mkdir # Generates the bin/ and obj/ folders
           # Only needs to be run once.
make
```

* Execute the binary: Optional `network.save` argument to load and evaluate an existing network. If no arguments are provided, the program will create, train and save a new network.
```bash
./bin/deepsea [network.save]
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
