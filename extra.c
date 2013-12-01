#include <emmintrin.h>
#include<string.h>
#include <omp.h>
#include <stdio.h>
//#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
//#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define blocksize 390000 


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
	   float* kernel, int kernel_x, int kernel_y)

{

    #define KERNX kernel_x
    #define KERNY kernel_y
    #define kernsize KERNX*KERNY        
    #define kernmid kernsize/ 2
    
        //flipping the kernel

        float k[kernsize];
        for (int i = 0; i<kernsize; i++) { 
                k[i] = kernel[kernsize-1-i]; //k is flipped version of kernel
        }

        float* out_index; // all variables we need to use later
        __m128 n;
        __m128 kk;
        __m128 m;
        int dy;

#define kern_cent_X ((KERNX - 1) / 2)
#define kern_cent_Y ((KERNY - 1) / 2)
	/*
	//top edge first
	int y = 0;
	for (int x = kern_cent_X; x+3 <data_size_X-kern_cent_X ;x+=4) {
	    n = _mm_setzero_ps();
	    for (int i=KERNX; i< kernsize;i++) {
		kk = _mm_load1_ps(k+i);
		m = _mm_loadu_ps(in+(x-kern_cent_X+i%KERNX)+(y-1+i/KERNX)*data_size_X);
		n=_mm_add_ps(n,_mm_mul_ps(kk,m));
	    }
	    _mm_storeu_ps(out+x,n);
	}
	//edge of two sides

	//bottom edge
		y = data_size_Y-1
	        for (int x = 0; x < data_size_X; x++) {
                        for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                                for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
                                        if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
                                                out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
                                        }
                                }
                        }
                }
	*/

	
	//top part first
	for (int y = 0; y < kern_cent_Y; y ++) {
                for (int x = 0; x < data_size_X; x++) {
                        for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                                for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
                                        if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
                                                out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
                                        }
                                }
                        }
                }
        }

	 //the two sides
#pragma omp parallel for num_threads(16) firstprivate(data_size_Y, data_size_X,kernel)
        for (int y = kern_cent_Y; y < data_size_Y - kern_cent_Y; y++) {
                for (int x = 0; x < kern_cent_X; x ++) {
		    for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                                for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
                                        if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
                                                out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
                                        }
                                }
                        }
                }

		for (int x = data_size_X-kern_cent_X; x < data_size_X; x ++) {
		    for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                                for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
                                        if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
                                                out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
                                        }
                                }
                        }
                }
        }        

	//bottom part
	for (int y = data_size_Y-kern_cent_Y; y < data_size_Y; y ++) {
                for (int x = 0; x < data_size_X; x++) {
                        for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
                                for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
                                        if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
                                                out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
                                        }
                                }
                        }
                }
        }   

        // main convolution loop
        int x;
        int i;
#pragma omp parallel for num_threads(16) firstprivate( x, i, k,out_index, data_size_X,n,kk,m,dy)//lots of privatizing
        for(int y = kern_cent_Y; y < data_size_Y - kern_cent_Y; y++){ // the y coordinate of the output location we're focusing on
                for(x = kern_cent_X; x + 31 < data_size_X - kern_cent_X; x+=32){ // the x coordinate of the output location we're focusing on
                        out_index = out+x+y*data_size_X;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x - kern_cent_X+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+4+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+8+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+12+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+16+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+20+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+24+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);

                        out_index += 4;
                        n =  _mm_setzero_ps(); //getting the next four values of output matrix
                        for (i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+28+i%KERNX)+(y - kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);
                }

                for (;x+3<data_size_X - 1;x+=4) {
                        out_index = out+x+y*data_size_X;
                        n = _mm_setzero_ps();
                        for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                                m = _mm_loadu_ps(in+(x-kern_cent_X+i%KERNX)+(y-kern_cent_Y+i/KERNX)*(data_size_X)); //load four adjacent values from in
                                n = _mm_add_ps(n, _mm_mul_ps(kk, m));//multiply and add the product to n
                        }
                        _mm_storeu_ps(out_index,n);
                }

                for(;x<data_size_X - 1;x++) {
                        float sum = 0.0;
                        for (int i = 0; i<kernsize;i++) {
                                sum += in[x-kern_cent_X+i%KERNX+(y-kern_cent_Y+i/KERNX)*data_size_X]*k[i];
                        }
                        out[x+y*data_size_X] = sum;
                }
        }
        //end convo loop
        //y_prev = y_cap - 2;
        //}
return 1;
}
