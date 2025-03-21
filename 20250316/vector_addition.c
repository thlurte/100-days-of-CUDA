#include <stdio.h>

void add_vectors(int a[], int b[], int c[], int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {6, 7, 8, 9, 10};
    int c[5];

    add_vectors(a, b, c, 5);

    for (int i = 0; i < 5; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");
    return 0;
}