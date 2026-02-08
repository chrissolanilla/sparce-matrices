#include <stdio.h>
#include <stdlib.h>
//sparse matrices, we have CRS, CsS, JDS, and TJDS

//Compressed Row storage:
typedef struct {
	int rows;
	int columns;
	int array[];
} CRS;

//Compressed column storage:
typedef struct {
    int columns;
    int array[];
} CsS;

//Jagged dense storage:
typedef struct {
	int rows;
	int columns;
    int array[];
} JDS;

//Transpose jagged dense storage
typedef struct {
	int rows;
	int columns;
	int array[];
} TJDS;

int main(void) {
	//A is an 6x6 matrix
	int A[6][6] = {{10,0,0,0,-2,0},
					{3,9,0,0,0,0},
					{0,0,8,7,3,0},
					{0,0,8,7,0,0},
					{0,8,0,9,9,0},
					{0,5,0,0,2,-1}
				};

	for(int i=0; i<6; i++) {
		for(int j=0; j<6; j++) {
			/* printf("row %d column %d: ",i,j); */
			printf("%d ",A[i][j]);
		}
		printf("\n");
	}
	return 0;
}
