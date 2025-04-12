#include <stdio.h>

void swap(int *a, int *b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

int main() {

	int a = 15;
	int b = 16;

	printf("a->%d\tb->%d\n",a,b);
	swap(&a,&b);
	printf("a->%d\tb->%d\n",a,b);

	return 0;
}
