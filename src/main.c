#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "matrix.h"
#include "utils.h"

void interactive_mode(network_t* net)
{
	double X[2];
	
	while(1)
	{
		//system("clear");
		
		for (size_t i = 0; i < net->input_size; i++)
		{
			printf("\tParameter [%zu] >> ", i+1);
			scanf("%lf", &X[i]);
		}

		printf("\n");
		net_predict(net, X);

		//printf("\nNew prediction? [y/N] >> ");
		
		//if (getchar() == 	'n')
			//break;
	}
}

int main(int argc, char* argv[])
{	
	(void) argc;
	(void) argv;
	
	srand(0);

	size_t L = 3;
	size_t input_size = 2;
	size_t hidden_size = 4;
	size_t output_size = 1;
	
	size_t n_data = 4;
	size_t batch_size = 2;
	double lr = 0.1f;
	
	dataset_t* data = data_init(n_data, input_size, output_size);
	network_t* net = net_init(L, input_size, hidden_size, output_size, batch_size, lr);
	
	net_train(net, data, 100000);

	interactive_mode(net);

	net_free(net);
	data_free(data);

	return 0;
}