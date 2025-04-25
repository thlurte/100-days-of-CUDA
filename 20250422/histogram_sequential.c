#include <stdio.h>
#include <stdlib.h>

int main() 
{
	const unsigned int n = 1000000;
	char* data = (char *)malloc(n*sizeof(char));
	unsigned int* hist = (int *)calloc(n,sizeof(int));

	char alphabet[] = {"abcdefghijklmnopqrstuvwxyz"};

	srand(42);
	for (unsigned int i = 0; i < n; i++) {
		int idx = rand() % 26;
		data[i] = alphabet[idx];
		hist[idx]++;
	}

	for (unsigned int i = 0; i < 26 ; i++) {
		printf("%c, %u \n", alphabet[i], hist[i]);
	}

	free(data);
	free(hist);
	return 0;

}



