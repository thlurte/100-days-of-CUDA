#include <stdio.h>

void foo(int *ab);

int main() {

	int ab = 42;
	foo(&ab);
	printf("%d\n",ab);
	

	return 0;
}

void foo(int *ab) {
	*ab = 1337;
}
