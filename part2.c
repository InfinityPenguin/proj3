
#include <emmintrin.h>
#include<string.h>
#include <omp.h>
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
/*        printf("flipped kernel: "); // debugging: print the flipped kernel
        for (int i = 0; i < KERNX*KERNY; i++) {
                  printf("%.2f ", k[i]); 
        }
        printf("\n");
*/

    //zero pad the matrix "in" and call it "buf". Example: if "in" was 4x4, then copy it onto buf while padding it so that it is 6x6. remember to keep buf as a 1-D array, row-wise implemented
    int buf_x = data_size_X+2;//account for cases where length is not divisible by 4
    int buf_y = data_size_Y+2;
    int buf_size = buf_x*buf_y;

    float buf[buf_size];
    memset(buf, 0.0, buf_size*sizeof(float));
    
    int j;
#pragma omp parallel for num_threads(16) firstprivate(buf_x,data_size_X)
    for (j = 0; j < data_size_Y; j++) {
       int i;
       for (i = 0; i + 3 < data_size_X; i+=4) {
                _mm_storeu_ps(buf + (i + 1) + (j + 1) * buf_x, _mm_loadu_ps(in + i + j*data_size_X));        
        }
        for (; i < data_size_X; i++) {
            buf[(i+1)+(j+1)*(buf_x)] = in[i + j*data_size_X]; //update the elements with in
        }
    }

/*        printf("buffered input:\n"); // debugging: print the buffered input
        for (int i = 0; i < buf_x * buf_y; i++) {
                printf("%.2f ", buf[i]);
                if ((i + 1) % buf_x == 0)
                        printf("\n");
        }
*/
    
    // main convolution loop
    
    int chunk;
    if (data_size_Y*data_size_X<=390000) {
	chunk = data_size_Y/16;
    }
    else {
	chunk =(390000/data_size_X)/16;
	} 

    float* out_index;
    __m128 n;
    __m128 kk;
    __m128 m;
    int dy;
    /*
    if (data_size_X*data_size_Y>=400000)
	dy = 400000/data_size_X;
    else
	dy = data_size_Y;
    */
    #define kernsize KERNX*KERNY
    /*
#pragma omp parallel for num_threads(16) firstprivate(k,data_size_X,out_index, buf_x,n,kk,m) //schedule(static, chunk) //lots of privatizing
    for(int y = 0; y <data_size_Y; y++){ // the y coordinate of the output location we're focusing on
	int i;
	int x;
	__m128 o;
	for (i=0;i<kernsize;i++){
	    kk =  _mm_load1_ps(k+i); //load four of the same value of k[i]
	    for(x = 0; x + 31 < data_size_X; x+=32){ // the x coordinate of the output location we're focusing on
		int tmp = i%KERNX+(y+i/KERNY)*(buf_x);
		out_index = out+x+y*data_size_X;
		m = _mm_loadu_ps(buf+x+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);//load four adjacent values from out
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);//store n back to out

		out_index += 4;
		m = _mm_loadu_ps(buf+x+4+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);

		out_index += 4;
	        m = _mm_loadu_ps(buf+x+8+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);

		out_index += 4;
		m = _mm_loadu_ps(buf+x+12+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);
			     
		out_index += 4;
		m = _mm_loadu_ps(buf+x+16+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);
        
		out_index += 4;
		m = _mm_loadu_ps(buf+x+20+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);

		out_index += 4;
		m = _mm_loadu_ps(buf+x+24+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);
        
		out_index += 4;
		m = _mm_loadu_ps(buf+x+28+tmp); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk,m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);
		
	    }

	    for (;x+3<data_size_X;x+=4) {
		out_index = out+x+y*data_size_X;
		m = _mm_loadu_ps(buf+(x+i%KERNX)+(y+i/KERNY)*(buf_x)); //load four adjacent values from buf
		o = _mm_loadu_ps(out_index);
		n = _mm_add_ps(o, _mm_mul_ps(kk, m));//multiply and add the product to n
		_mm_storeu_ps(out_index,n);
	    }

	    for(;x<data_size_X;x++) {
		out[x+y*data_size_X] += buf[x+i%KERNX+(y+i/KERNY)*buf_x]*k[i];
	    }
	}
        }
	*/
#pragma omp parallel for num_threads(16) firstprivate(k,data_size_X,out_index, buf_x,n,kk,m,dy) schedule(static,chunk)//lots of privatizing
    for(int y=0; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
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
    
    return 1;
}
