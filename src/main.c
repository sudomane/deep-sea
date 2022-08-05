#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "utils.h"

void start()
{
	// Static seed, easier for testing
	srand(0);
	system("clear");
	
	double X[INPUT_SIZE] = { 0.0f, 0.0f };
	double y[OUTPUT_SIZE] = { 0.0f };

	// Initialize network with activation function
	network_t* network = init_network(X, sigmoid, 0.1f);

	train(network, y, 1000);
	predict(network, X);

	free_network(network);
}

int main(int argc, char* argv[])
{
	start();

	return 0;
}
