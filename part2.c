#include <emmintrin.h>
#include<string.h>
#include <omp.h>
#include <stdio.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{

    //flipping the kernel
    
   float k[KERNX*KERNY];
   for (int i = 0; i<KERNX*KERNY; i++) { 
        k[i] = kernel[KERNX*KERNY-1-i]; //k is flipped version of kernel
   }

    //zero pad the matrix "in" and call it "buf". Example: if "in" was 4x4, then copy it onto buf while padding it so that it is 6x6. remember to keep buf as a 1-D array, row-wise implemented
    int j;
    int y_cap;
    if (data_size_X*data_size_Y< 390000) { //check if the matrix is too large to fit in cache
	y_cap = data_size_Y;
    }
    else
	y_cap = (390000/data_size_X); //the maximum y so there's no cache overflow
    
    int buf_x = data_size_X+2;
    int buf_y = y_cap+2;
    int buf_size = buf_x*buf_y;
    
    float buf[buf_size]; //create a buf with only y_cap as its y
    memset(buf, 0.0, buf_size*sizeof(float));
    
#pragma omp parallel for num_threads(16) firstprivate(buf_x,data_size_X)
    for (j = 0; j < y_cap; j++) {
       int i;
       for (i = 0; i + 3 < data_size_X; i+=4) {
                _mm_storeu_ps(buf + (i + 1) + (j + 1) * buf_x, _mm_loadu_ps(in + i + j*data_size_X));        
        }
        for (; i < data_size_X; i++) {
            buf[(i+1)+(j+1)*(buf_x)] = in[i + j*data_size_X]; //update the elements with in
        }
    }

    float* out_index; //all variables we need to use later
    __m128 n;
    __m128 kk;
    __m128 m;
    int dy;

    #define kernsize KERNX*KERNY	
    //first part of convolution
    int y_limit; //the last y we can update to out_index
    if (y_cap == data_size_Y)
	y_limit = data_size_Y;
    else
	y_limit = y_cap-1; //because we've reached the bottom of the matrix; the kernel takes up another row below
#pragma omp parallel for num_threads(16) firstprivate(k,data_size_X,out_index, buf_x,n,kk,m,dy)//lots of privatizing
    for(int y=0; y < y_limit; y++){ // the y coordinate of the output location we're focusing on
	int i;
	int x;
        for(x = 0; x + 31 < data_size_X; x+=32){ // the x coordinate of the output location we're focusing on
            out_index = out+x+y*data_size_X;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+4+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+8+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+12+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
	     
           out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+16+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i <kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+20+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+24+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+28+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
          }

        for (;x+3<data_size_X;x+=4) {
            out_index = out+x+y*data_size_X;
            n = _mm_setzero_ps();
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf+(x+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk, m));//multiply and add the product to n
             }
            _mm_storeu_ps(out_index,n);
        }

	for(;x<data_size_X;x++) {
	    float sum = 0.0;
	    for (int i = 0; i<kernsize;i++) {
		sum += buf[x+i%KERNX+(y+i/KERNY)*buf_x]*k[i];
	    }
	    out[x+y*data_size_X] = sum;
	}
        }
    //printf("finished the loop!\n");
    if (y_limit == data_size_Y) //if y_limit is the number of rows in the in matrix, then we are done
	return 1;
    
    int buf2_y;
    int y_limit_2;
    int y_load;
    if ((data_size_Y-y_cap)*data_size_X>390000) {
	buf2_y = y_cap+2;
	y_limit_2 = y_limit+y_cap;
	y_load = buf2_y-2;
    }
    else {
	buf2_y = data_size_Y-y_cap+1+2;
	y_limit_2 = data_size_Y;
	y_load = buf2_y-2;
    }
    
//else we need to finish loading the rest of the in matrix into buf2, and then run through the convolution
    int buf2_x = data_size_X+2;
    float buf2[buf2_x*buf2_y]; //now create a second buffer matrix with rest of in matrix
    memset(buf2, 0.0, (buf2_x*buf2_y)*sizeof(float));
    //printf("made buf2!\n");
    
    
#pragma omp parallel for num_threads(16) firstprivate(buf_x,data_size_X,y_limit)
    for (j = -1; j < y_load; j++) {
       int i;
       for (i = 0; i + 3 < data_size_X; i+=4) {
	   _mm_storeu_ps(buf2 + (i + 1) + (j + 1) * buf2_x, _mm_loadu_ps(in + i + (j+y_limit)*data_size_X));        
        }
        for (; i < data_size_X; i++) {
            buf2[(i+1)+(j+1)*(buf2_x)] = in[i + (j+y_limit)*data_size_X]; //update the elements with in
        }
    }

#pragma omp parallel for num_threads(16) firstprivate(k,data_size_X,out_index, buf_x,n,kk,m,dy,y_limit)
    for(int y=y_limit; y < y_limit_2; y++){ // the y coordinate of the output location we're focusing on
	int i;
	int x;
	//printf("y = %d, y in in array is %f\n", y, in[0+y*data_size_X]);
	//printf("y in buf array is %f\n", buf2[(0+1)+(y-y_limit+1)*buf2_x]);
        for(x = 0; x + 31 < data_size_X; x+=32){ // the x coordinate of the output location we're focusing on
            out_index = out+x+y*data_size_X;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

	     //   printf("finished x=0\n");

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+4+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+8+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+12+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
	     
           out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+16+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i <kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+20+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+24+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+28+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
          }

        for (;x+3<data_size_X;x+=4) {
            out_index = out+x+y*data_size_X;
            n = _mm_setzero_ps();
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf2+(x+i%KERNX)+((y-y_limit)+i/KERNY)*(buf2_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk, m));//multiply and add the product to n
             }
            _mm_storeu_ps(out_index,n);
        }

	for(;x<data_size_X;x++) {
	    float sum = 0.0;
	    for (int i = 0; i<kernsize;i++) {
		sum += buf2[x+i%KERNX+((y-y_limit)+i/KERNY)*buf2_x]*k[i];
	    }
	    out[x+y*data_size_X] = sum;
	}
        }
    if (y_limit_2==data_size_Y)
	return 1;

    int buf3_x=buf2_x;
    int buf3_y=data_size_Y-2*y_cap+1+2;
    float buf3[buf3_x*buf3_y]; //now create a second buffer matrix with rest of in matrix
    memset(buf2, 0.0, (buf3_x*buf3_y)*sizeof(float));
    printf("made buf3!\n");
    
    
#pragma omp parallel for num_threads(16) firstprivate(buf_x,data_size_X,y_limit_2)
    for (j = -1; j < data_size_Y-2*y_cap+1; j++) {
       int i;
       //printf("j is %d\n", j);
       for (i = 0; i + 3 < data_size_X; i+=4) {
	   //printf("loading in matrix location i = %d, value = %f\n",i,in[i+(j+y_limit_2)*data_size_X]);
	   _mm_storeu_ps(buf3 + (i + 1) + (j + 1) * buf3_x, _mm_loadu_ps(in + i + (j+y_limit_2)*data_size_X));        
        }
        for (; i < data_size_X; i++) {
            buf3[(i+1)+(j+1)*(buf3_x)] = in[i + (j+y_limit_2)*data_size_X]; //update the elements with in
        }
    }
    printf("finished creating buf3!\n");
#pragma omp parallel for num_threads(16) firstprivate(k,data_size_X,out_index, buf_x,n,kk,m,dy,y_limit_2)
    for(int y=y_limit_2; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
	int i;
	int x;
	//printf("y = %d, y in in array is %f\n", y, in[1+y*data_size_X]);
	//printf("y in buf array is %f\n", buf3[(1+1)+(y-y_limit_2+1)*buf3_x]);
        for(x = 0; x + 31 < data_size_X; x+=32){ // the x coordinate of the output location we're focusing on
            out_index = out+x+y*data_size_X;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
	    //  printf("buf is %f, in is %f\n",buf3[x+1+((y-y_limit_2)+1)*(buf3_x)], in[x+y*data_size_X]);
	     
             _mm_storeu_ps(out_index,n);

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+4+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
            n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+8+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+12+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
	     
           out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+16+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (i = 0; i <kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+20+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);

            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+24+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
        
            out_index += 4;
             n =  _mm_setzero_ps(); //getting the next four values of output matrix
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+28+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk,m));//multiply and add the product to n
             }
             _mm_storeu_ps(out_index,n);
          }

        for (;x+3<data_size_X;x+=4) {
            out_index = out+x+y*data_size_X;
            n = _mm_setzero_ps();
            for (int i = 0; i < kernsize;i++) { //iterating through kernal               
                kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
                m = _mm_loadu_ps(buf3+(x+i%KERNX)+((y-y_limit_2)+i/KERNY)*(buf3_x)); //load four adjacent values from buf
                n = _mm_add_ps(n, _mm_mul_ps(kk, m));//multiply and add the product to n
             }
            _mm_storeu_ps(out_index,n);
        }

	for(;x<data_size_X;x++) {
	    float sum = 0.0;
	    for (int i = 0; i<kernsize;i++) {
		sum += buf3[x+i%KERNX+((y-y_limit_2)+i/KERNY)*buf3_x]*k[i];
	    }
	    out[x+y*data_size_X] = sum;
	}
        }

    
    return 1;
}

