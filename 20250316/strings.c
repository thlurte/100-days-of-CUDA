#include <stdio.h>
#include <string.h>
int main() {
    char * first_name = "John";
    char * last_name = "Doe";

    printf("Hello, %s %s \n", first_name, last_name);

    char name[] = "John Doe";
    printf("Hello, %s \n", name);

    if (strncmp(strcat(strcat(first_name, " "), last_name),name, 3)==0) {
        printf("The first 5 characters of first_name and last_name are the same \n ");
    } else {
        printf("The first 5 characters of first_name and last_name are different \n ");
    }

    return 0;
}