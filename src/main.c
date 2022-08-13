#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "matrix.h"
#include "utils.h"

#define IMAGE_DATA "data/train-images-idx3-ubyte"
#define LABEL_DATA "data/train-labels-idx1-ubyte"

int main(int argc, char* argv[])
{
	if (argc != 1 && argc != 3)
	{
		errx(-42, "Correct usage:\n\t./main (no arguments, default MNIST in data/)"
				  "\n\t./main [MNIST IMAGE PATH] [MNIST LABEL PATH]");
	}

	srand(0);
	
	const char* image_data = IMAGE_DATA;
	const char* label_data = LABEL_DATA;

	if (argc == 3)
	{
		image_data = argv[1];
		label_data = argv[2];
	}
	
	size_t epochs = 10;
	
	size_t L = 3;
	size_t input_size = 784;
	size_t hidden_size = 16;
	size_t output_size = 10;
	
	size_t n_data = 512;
	size_t batch_size = 64;
	double lr = 0.1f;
	
	dataset_t* train_dataset = data_init(n_data, input_size, output_size);
	network_t* net = net_init(L, input_size, hidden_size, output_size, batch_size, lr);
	
	data_load_mnist(image_data, train_dataset, LOAD_IMAGES);
	data_load_mnist(label_data, train_dataset, LOAD_LABELS);

	net_train(net, train_dataset, epochs);

	net_free(net);
	data_free(train_dataset);

	return 0;
}