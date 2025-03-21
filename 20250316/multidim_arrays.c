#include <stdio.h>

int main(){


    char c[5][4];

    c[0][0] = 1;
    printf("%d \n", c[0][0]);
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    
    printf("%d \n", v[0][2]);
    
    return 0;

}