#include <emmintrin.h>
#include<string.h>
#include <omp.h>
#include <stdio.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define kernsize KERNX*KERNY	
#define kernmid kernsize >> 1
#define numthreads 16

int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel) {
	// flipping the kernel
	float k[kernsize];
	for (int i = 0; i < kernsize; i++) { 
		k[i] = kernel[kernsize - 1 - i]; // k is the flipped version of kernel
	}

	float* out_index; // variables needed later
	__m128 n;
	__m128 kk;
	__m128 m;
	int dy;
	int x;
	int i;

	int kern_cent_X = KERNX >> 1;
	int kern_cent_Y = KERNY >> 1;

	for (int y = 0; y < data_size_Y; y++) { // left and right edge cases 
		for (int x = 0; x < data_size_X; x += data_size_X - 1) {
			for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
					if (x + i > -1 && x + i < data_size_X && y + j > -1 && y + j < data_size_Y) {
						out[x + y * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (y + j) * data_size_X];
					}
				}
			}
		}
	}

	for (x = 1; x + 3 < data_size_X - 1; x += 4) { // top row edge cases
		n =  _mm_setzero_ps();		
		for (int i = (KERNY >> 1) * KERNX; i < kernsize; i++) {
			kk = _mm_load1_ps(k + i);
			m = _mm_loadu_ps(in + x - 1 + i % KERNX + (i / KERNY - 1) * data_size_X);
			n = _mm_add_ps(n, _mm_mul_ps(kk, m));
		}
		_mm_storeu_ps(out + x, n);
	}
	for (; x < data_size_X - 1; x++) {
		for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (j > -1 && j < data_size_Y) {
					out[x] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + j * data_size_X];
				}
			}
		}
	}

	for (x = 1; x + 3 < data_size_X - 1; x += 4) { // bottom row edge cases
		n =  _mm_setzero_ps();		
		for (int i = 0; i < ((KERNY >> 1) + 1) * KERNX; i++) {
			kk = _mm_load1_ps(k + i);
			m = _mm_loadu_ps(in + x - 1 + i % KERNX + (data_size_Y - 2 + i / KERNX) * data_size_X);
			n = _mm_add_ps(n, _mm_mul_ps(kk, m));
		}
		_mm_storeu_ps(out + x + (data_size_Y - 1) * data_size_X, n);
	}
	for (; x < data_size_X - 1; x++) {
		for (int i = -kern_cent_X; i <= kern_cent_X; i++) {
			for (int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
				if (data_size_Y - 1 + j > -1 && data_size_Y - 1 + j < data_size_Y) {
					out[x + (data_size_Y - 1) * data_size_X] += kernel[(kern_cent_X - i) + (kern_cent_Y - j) * KERNX] * in[(x + i) + (data_size_Y - 1 + j) * data_size_X];
				}
			}
		}
	}

	// main convolution loop
#pragma omp parallel for num_threads(numthreads) firstprivate(x, i, k, out_index, data_size_X, n, kk, m, dy)
	for(int y = 1; y < data_size_Y - 1; y++) { // the y coordinate of the output location we're focusing on
		for(x = 1; x + 31 < data_size_X - 1; x+=32){ // the x coordinate of the output location we're focusing on
			out_index = out+x+y*data_size_X;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x - 1+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+4+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+8+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+12+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+16+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+20+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+24+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);

			out_index += 4;
			n =  _mm_setzero_ps(); //getting the next four values of output matrix
			for (i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+28+i%KERNX)+(y - 1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);
		}

		for (;x+3<data_size_X - 1;x+=4) {
			out_index = out+x+y*data_size_X;
			n = _mm_setzero_ps();
			for (int i = 0; i < kernsize;i++) { //iterating through kernal               
				kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
				m = _mm_loadu_ps(in+(x-1+i%KERNX)+(y-1+i/KERNY)*(data_size_X)); //load four adjacent values from in
				n = _mm_add_ps(n, _mm_mul_ps(kk, m));//multiply and add the product to n
			}
			_mm_storeu_ps(out_index,n);
		}

		for(;x<data_size_X - 1;x++) {
			float sum = 0.0;
			for (int i = 0; i<kernsize;i++) {
				sum += in[x-1+i%KERNX+(y-1+i/KERNY)*data_size_X]*k[i];
			}
			out[x+y*data_size_X] = sum;
		}
	}
	// end main convo loop

	return 1;
}
