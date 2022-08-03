#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "utils.h"

int main()
{
	// Static seed, easier for testing
	srand(0);

	network_t* network = init_network();
	
	double X[INPUT_SIZE] = { 1.0f, 0.0f };
	init_input(network, X);
	summary(network, 0);

	feed_forward(network);
	summary(network, 1);
	
	double y[1] = { 1.0f };
	double mse = cost_function(network, y);
	printf("MSE: %f\n", mse);
	free_network(network);

	return 0;
}
