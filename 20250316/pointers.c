#include <stdio.h>

int main() {
    int a = 15;
    int *pA = &a;  // Pointer stores address of a

    printf("Variable value: %d \n", a);         // Correct
    printf("Variable address: %p \n", &a);  // %p for addresses, cast to void*
    printf("Pointer value (address of a): %p \n", pA);  // pA is a pointer, use %p
    printf("Value at pointer (dereferenced): %d \n", *pA); // *pA gives value stored at address
    printf("Pointer variable's own address: %p \n", &pA);  // Address of the pointer itself

    return 0;
}
