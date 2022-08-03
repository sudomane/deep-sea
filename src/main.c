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
	
	double input[INPUT_SIZE] = { 1.0f, 0.0f };
	init_input(network, input);
	summary(network, 1);

	feed_forward(network);
	summary(network, 1);
	
	free_network(network);

	return 0;
}
