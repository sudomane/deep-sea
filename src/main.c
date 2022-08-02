#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"

int main()
{
	srand(0);

	network_t* network = init_network();
	
	double input[INPUT_SIZE] = { 1.0f, 0.0f };
	init_input(network, input);

	display_network(network);
	feed_forward(network);
	printf("\nAFTER FEEDFORWARD\n\n");
	display_network(network);
	free_network(network);

	return 0;
}
