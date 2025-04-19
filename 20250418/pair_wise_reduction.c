#include <stdio.h>
#include <time.h>
#include <stdlib.h>


void reduce(const float* data_h, float* result, int n)
{
	for (unsigned int i = 0; i < n; i++) 
	{
		result[i] = data_h[i*2] + data_h[i*2+1];
	}
	if (n%2!=0)
	{
		result[0] = data_h[-1];
	}
}


int main() 
{
	unsigned const int n = 10 << 2;

	float * data_h = (float*)calloc(n, sizeof(float));

	srand(42);
	for (unsigned int i = 0; i < n; i++)
	{
		data_h[i] = (float)rand()/RAND_MAX;
	}

	float * result_h = (float*)calloc(n,sizeof(float));

	reduce(data_h,result_h,n);
	for (int n_c = n/2; n_c > 1; n_c/=2)
	{
		reduce(result_h, result_h,n);
	}

	free(data_h);
	free(result_h);

}
