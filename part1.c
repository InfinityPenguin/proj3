#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    //NOTE: right now the code only works with matrices of length divisible by four!!!!
    
    //flipping the kernel
    float k[KERNX*KERNY];
    for (int i = 0; i<KERNX*KERNY/2; i++) {
	k[i] = kernel[KERNX*KERNY-1-i]; //k is flipped version of kernel
    }

    //zero pad the matrix "in" and call it "buf". Example: if "in" was 4x4, then copy it onto buf while padding it so that it is 6x6. remember to keep buf as a 1-D array, row-wise implemented
    
    
    // the x coordinate of the kernel's center
    // int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    //int kern_cent_Y = (KERNY - 1)/2;
    
    // main convolution loop
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
	for(int x = 0; x + 3 < data_size_X; x+=4){ // the x coordinate of the output location we're focusing on
	    __m128i n = _mm_loadu_si128((__m128i*) (out+x+y*data_size_X)); //getting the next four values of output matrix
	    for (int i = 0; i < KERNX*KERNY;i++) { //iterating through kernal	       
		__m128i kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
		__m128i m = _mm_loadu_ps(buf);
		//	out[x+y*data_size_X] += 
		//	  kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
		  }
	  }
	}
    
    return 1;

    
}
