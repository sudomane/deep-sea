#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "matrix.h"
#include "utils.h"

#define TRAIN_IMAGE_DATA "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_DATA "data/train-labels-idx1-ubyte"

#define TEST_IMAGE_DATA "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_DATA "data/t10k-labels-idx1-ubyte"

int main(int argc, char* argv[])
{
	if (argc != 1 && argc != 3)
	{
		errx(-42, "Correct usage:\n\t./main (no arguments, default MNIST in data/)"
				  "\n\t./main [MNIST IMAGE PATH] [MNIST LABEL PATH]");
	}

	srand(time(NULL));
	
	const char* train_image_data = 	TRAIN_IMAGE_DATA;
	const char* train_label_data =	TRAIN_LABEL_DATA;

	const char* test_image_data =	TEST_IMAGE_DATA;
	const char* test_label_data =	TEST_LABEL_DATA;

	if (argc == 3)
	{
		train_image_data = argv[1];
		train_label_data = argv[2];
	}
	
	size_t epochs =		 	100;
	
	size_t L =			 	4;
	size_t input_size =		784;
	size_t hidden_size = 	32;
	size_t output_size = 	10;
	
	size_t n_train_data = 	256;
	size_t n_test_data = 	1;

	size_t batch_size = 	64;

	double lr = 			0.1f;
	
	dataset_t* train_dataset = data_init(n_train_data, input_size, output_size);
	dataset_t* test_dataset = data_init(n_test_data, input_size, output_size);
	
	network_t* net = net_init(L, input_size, hidden_size, output_size, batch_size, lr);
	
	data_load_mnist(train_image_data, train_dataset, LOAD_IMAGES);
	data_load_mnist(train_label_data, train_dataset, LOAD_LABELS);

	net_train(net, train_dataset, epochs);

	data_load_mnist(test_image_data, test_dataset, LOAD_IMAGES);
	data_load_mnist(test_label_data, test_dataset, LOAD_LABELS);

	net_evaluate(net, test_dataset);

	//net_save(net, "test.save");

	net_free(net);
	data_free(train_dataset);

	return 0;
}