#include <stdio.h>

unsigned int ord(char c) {
	return (unsigned int)c;
}

int main(){

	char a[] = "anagram";
	char b[] = "aaangrm";

	int _a[7];
	int _b[7];

	for (int i = 0; i<7; i++) {
		_a[i] = ord(a[i]);
		printf("%d \n",ord(a[i]));
		_b[i] = ord(b[i]);
	}

	

	





	return sort(_a) == sort(_b);
}
