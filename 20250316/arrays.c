#include <stdio.h>

int main(){
    int a[5] = {1, 2, 3, 4, 5};
    printf("%d", a[0]);

    int b[5];
    b[4] = 4;

    printf("%d", b[1]);
    printf("%d \n", b[4]);
}