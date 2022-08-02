#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "network.h"
#include "utils.h"

int main()
{
	// Static seed, easier for testing
	srand(0);

	network_t* network = init_network();
	
	double input[INPUT_SIZE] = { 1.0f, 0.0f };
	init_input(network, input);
	summary(network, 0);

	clock_t t_1, t_2;

	t_1 = clock();
	feed_forward(network);
	t_2 = clock();

	clock_t delta = t_2 - t_1;
	printf("Time to compute: %ld ms\n", delta);

	free_network(network);

	return 0;
}
