/**
 * @file	main.c
 * @author 	Philippe Bouchet (philippe.bouchet@epita.fr)
 * @version 0.1
 * 
 * @brief		The goal of the main.c file is to serve as an example as to how
 * 				DeepSea can be implemented, by showing the workflow expected in an
 * 				implementation. 
 * @copyright 	Copyright (c) 2022
 * 
 */

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"

#define TRAIN_IMAGE_DATA "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_DATA "data/train-labels-idx1-ubyte"

#define TEST_IMAGE_DATA "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_DATA "data/t10k-labels-idx1-ubyte"

static void train_network();
static void evaluate_network(char* network_path);

int main(int argc, char* argv[])
{
	if (argc > 2)
		errx(-1, "Invalid arguments provided. Correct use:\n.\n"
				 "./main network.save - Evaluate existing network model\n"
				 "./main - Train and save new network.");

	srand(0);	
	system("clear");
	
	if (argc == 2)
		evaluate_network(argv[1]);
	else
		train_network();
	
	return 0;
}

static network_t* _configure_network()
{
	size_t L, input_size, hidden_size, output_size, batch_size;
	
	double lr;

	printf("\n[CONFIGURE NETWORK PARAMETERS]\n\n");

	printf("Hidden layers:\t");
	scanf("%zu", &L);
	printf("Input size:\t");
	scanf("%zu", &input_size);
	printf("Hidden size:\t");
	scanf("%zu", &hidden_size);
	printf("Output size:\t");
	scanf("%zu", &output_size);
	printf("Batch size:\t");
	scanf("%zu", &batch_size);
	printf("Learning rate:\t");
	scanf("%lf", &lr);
	printf("\n");
	
	network_t* net = net_init(L+1, input_size,
								   hidden_size,
								   output_size,
								   batch_size, lr);

	return net;
}

static void evaluate_network(char* network_path)
{
	size_t n_test_data;

	printf("\n[NETWORK EVALUATION]\n\n");
	printf("N testing data:\t");
	scanf("%zu", &n_test_data);
	
	network_t* net = net_load(network_path);
	dataset_t* test_dataset = data_init(n_test_data,
										net->input_size,
										net->output_size);
	
	data_load_mnist(TEST_IMAGE_DATA, test_dataset, LOAD_IMAGES);
	data_load_mnist(TEST_LABEL_DATA, test_dataset, LOAD_LABELS);

	net_evaluate(net, test_dataset);

	int r;
	
	while (1)
	{
		printf("Image to predict [1 - %zu] (0 to exit):\t", test_dataset->n);
		scanf("%d", &r);
		
		if (r <= 0)
		{
			break;
		}
		
		if ((size_t) r > test_dataset->n)
		{
			warnx("MAIN::PREDICTION: Selection out of bounds!");
			continue;
		}

		printf("Selected image:\t%d/%zu\n", r, test_dataset->n);
		net_predict(net, test_dataset->X[r-1], test_dataset->y[r-1]);
	}

	net_free(net);
	data_free(test_dataset);
}

static void train_network()
{
	size_t epochs;
	size_t n_train_data;
		
	printf("Epochs to train:\t");
	scanf("%zu", &epochs);
	printf("N training data:\t");
	scanf("%zu", &n_train_data);

	network_t* net = _configure_network();

	if (net->batch_size > n_train_data)
	{
		errx(-1, "MAIN::NETWORK: "
				 "Batch size greater than train data quantity.");
	}

	dataset_t* train_dataset = data_init(n_train_data,
										 net->input_size,
										 net->output_size);
	
	data_load_mnist(TRAIN_IMAGE_DATA, train_dataset, LOAD_IMAGES);
	data_load_mnist(TRAIN_LABEL_DATA, train_dataset, LOAD_LABELS);

	net_train(net, train_dataset, epochs);
	net_save(net, "network.save");

	net_free(net);
	data_free(train_dataset);
}