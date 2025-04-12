#include <stdio.h>


void foo(int num) {
    for (int i = 1; i<=num; i++) {
        if (i%3 == 0 && i%5 == 0) {
            printf("fizzBuzz\n");
        } else if (i%3 == 0) {
            printf("fizz\n");
        } else if (i%5 == 0) {
            printf("buzz\n");    
        } else {
            printf("%d\n",i);
        }
    }

}

int main() {
    foo(15);
    return 0;
}
