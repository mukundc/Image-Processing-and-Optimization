/* Parul Singh cs61c-hz
   Mukund Chillakanti cs61c-oz
*/
#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{	
	// the x coordinate of the kernel's center
	int kern_cent_X = (KERNX - 1)/2;
	// the y coordinate of the kernel's center
	int kern_cent_Y = (KERNY - 1)/2;
	float outSum;
	// the x/y size of the padded matrix
	int padded_X = calcExtra(data_size_X+(kern_cent_X*2));
	int padded_Y = data_size_Y+(kern_cent_X*2);
	// zero initiializng padded matrix
	float padded_matrix[padded_X*padded_Y];
	memset(padded_matrix, 0, sizeof(float) * padded_X * padded_Y);
	int padded_start = kern_cent_X+kern_cent_X*(padded_X);
	// moving in values into paddedmatrix from in

	for (int m = 0; m < data_size_Y; m++)
		memcpy(padded_start+padded_matrix+padded_X*m, in+m*data_size_X, sizeof(float)*data_size_X);

	// inverting kernel
	float newKern[KERNX*KERNY];
	for (int i = 0; i < KERNX; i++)
	{
		for (int j = 0; j < KERNY; j++)
		{
			newKern[i+j*KERNX] = kernel[(KERNX-1-i)+(KERNY-1 - j)*KERNX];		
		}
	}
	
	//matrix convolution
	__m128 paddedrow,multipliedvalues, kernelvalue;
	int x;
	for (int y = 0; y < data_size_Y; y++)
	{
		for (x = 0; x <= data_size_X-4; x+=4) {
			__m128 total = _mm_setzero_ps();
			for (int j = 0; j < KERNY; j++) {
				for(int i = 0; i < KERNX; i++) {
					kernelvalue = _mm_load1_ps(newKern + (i+j*KERNX));
					paddedrow = _mm_loadu_ps(padded_matrix+x+i+(y+j)*padded_X);
					multipliedvalues = _mm_mul_ps(paddedrow, kernelvalue);
					total = _mm_add_ps(total,multipliedvalues);
				}		 
			}
			_mm_storeu_ps(out+(x+y*data_size_X), total);
		}
		for (; x < data_size_X; x++)
		{
			outSum = 0;
			for (int j = 0; j < KERNY; j++) {
				for(int i = 0; i < KERNX; i++) {
					outSum += newKern[i+j*KERNX] * padded_matrix[x+i+(y+j)*padded_X];	
				}		 
			}
			out[x+y*data_size_X] = outSum;

		}
	}
	return 1;
}


int calcExtra(int size){
	while (size%4 != 0)
		size+=1;
	return size;
}

void printMatrix(float* matrix, int i, int j){
	for(int y = 0; y <j; y++){
		for(int x = 0; x <i; x++)
			printf("%f ", matrix[x+y*i]);
		printf("\n");	
	}
}




