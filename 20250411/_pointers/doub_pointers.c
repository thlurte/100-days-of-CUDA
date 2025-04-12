#include <stdio.h>

int main() {

	int a = 10;

	int *aa = &a;
	int **aaa = &aa;
	int ***aaaa = &aaa;


	printf("%d\t-> %d\t -> %d\t -> %d\t",a,*aa, **aaa, ***aaaa);

}
