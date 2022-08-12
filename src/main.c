#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "matrix.h"
#include "utils.h"

int main(int argc, char* argv[])
{	
	srand(time(NULL));

	// Optimal XOR configuration
	size_t L = 2;
	size_t input_size = 2;
	size_t hidden_size = 2;
	size_t output_size = 1;
	
	network_t* net = net_init(input_size, hidden_size, output_size, L);
	
	net_train(net, 100);
	net_display(net);

	net_free(net);

	return 0;
}
