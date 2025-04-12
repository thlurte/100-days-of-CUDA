#include <stdio.h>

void vectorAdd(int a[], int b[], int c[], int n) {
	for (int i = 0; i<n; i++) {
		c[i] = a[i]+b[i];
	}
}

int main() {

	int a[] = {1,2,3};
	int b[] = {4,5,6};
	int c[3];
	int n = 3;	

	vectorAdd(a,b,c,n);

	for (int i = 0; i<n; i++) {
		printf("%d\n",c[i]);
	}


	



	return 0;
}
