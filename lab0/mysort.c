#include <stdio.h>
#include <stdlib.h>

void insertion_sort(int *a, int n) {
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <n> <input_file>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    char *filename = argv[2];

    int *a = malloc(n * sizeof(int));
    if (!a) return 1;

    FILE *fp = fopen(filename, "r");
    if (!fp) return 1;

    for (int i = 0; i < n; i++) {
        fscanf(fp, "%d,", &a[i]);
    }
    fclose(fp);

    insertion_sort(a, n);

   // for (int i = 1; i < n; i++) {
  //      if (a[i] < a[i - 1]) {
  //          printf("array is NOT correctly sorted\n");
    //        free(a);
     //       return 1;
  //      }
  //  }

 //   printf("array is correctly sorted\n");
    free(a);
    return 0;
}
