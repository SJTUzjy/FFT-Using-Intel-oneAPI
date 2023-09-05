#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>

#include "mkl_vsl.h"
#include "mkl_dfti.h"

#define N (2048)

int const test_num = 1000;

float r[N * N];
float r2cData[N][N];
MKL_Complex8 r2cResult[N * (N / 2 + 1)];

struct complex {
	float x, y;
};

struct complex srcData[N][N];
struct complex destData[N][N];


void SplitArray(struct complex *src, struct complex *dest, int xn, int yn, int flag) {
	struct complex tmp[flag ? (yn >> 1) : (xn >> 1)], *s = src, *d = dest;
	if(!flag) {
		if(xn <= 1) return;
		for (int i = 0; i < xn >> 1; i++) {
			tmp[i].x = s[(i << 1)|1].x;
			tmp[i].y = s[(i << 1)|1].y;

			d[i].x = s[i << 1].x;
			d[i].y = s[i << 1].y;
		}
		for (int i = 0; i < xn >> 1; i++) {
			d[i + (xn >> 1)].x = tmp[i].x;
			d[i + (xn >> 1)].y = tmp[i].y;
		}
		SplitArray(dest, dest, xn >> 1, yn, flag);
		SplitArray(dest + (xn >> 1), dest + (xn >> 1), xn >> 1, yn, flag);
	}
	else {
		if(yn <= 1) return;
		for (int i = 0; i < yn >> 1; i++) {
			tmp[i].x = s[((i << 1)|1) * N].x;
			tmp[i].y = s[((i << 1)|1) * N].y;

			d[i * N].x = s[(i << 1) * N].x;
			d[i * N].y = s[(i << 1) * N].y;
		}
		for (int i = 0; i < yn >> 1; i++) {
			d[(i + (yn >> 1)) * N].x = tmp[i].x;
			d[(i + (yn >> 1)) * N].y = tmp[i].y;
		}
		SplitArray(dest, dest, xn, yn >> 1, flag);
		SplitArray(dest + N * (yn >> 1), dest + N * (yn >> 1), xn, yn >> 1, flag);
	}
}


void naiveFFT(struct complex *src, struct complex *dest)
{
    struct complex t1, t2, odd, even;
    for(int y = 0; y < N; y++) {
        SplitArray(src + y*N, dest + y*N , N, 0, 0);
        for(int i = 0; i < log2(N); i++) {
            int n = 2 * pow(2, i);
            for (int j = 0; j < N / n; j++) {
                for (int k = 0; k < n / 2; k++) {
                    float t = -2 * M_PI * k / n;

					t1.x = cos(t);
					t1.y = sin(t);

					odd = dest[y * N + j * n + k];
					even = dest[y * N + j * n + k + n / 2];

					t2.x = t1.x * even.x - t1.y * even.y;
					t2.y = t1.x * even.y + t1.y * even.x;

					dest[y * N + j * n + k].x = odd.x + t2.x;
					dest[y * N + j * n + k].y = odd.y + t2.y;
					dest[y * N + j * n + k + n / 2].x = odd.x - t2.x;
					dest[y * N + j * n + k + n / 2].y = odd.y - t2.y;
				}
            }
        }
    }
    for(int x = 0; x < N; x++){
        SplitArray(dest + x, dest + x, 0, N, 1);
        for(int i = 0; i < log2(N); i++) {
            int n = 2 * pow(2, i);
            for(int j = 0; j < N/n; j++) {
                for(int k = 0; k < n / 2; k++) {
                    float t =  -2 * M_PI * k / n;

					t1.x = cos(t);
					t1.y = sin(t);

					odd = dest[(j * n + k) * N + x];
					even = dest[(j * n + k + n / 2) * N + x];

					t2.x = t1.x * even.x - t1.y * even.y;
					t2.y = t1.x * even.y + t1.y * even.x;

					dest[(j * n + k) * N + x].x = odd.x + t2.x;
					dest[(j * n + k) * N + x].y = odd.y + t2.y;
					dest[(j * n + k + n / 2) * N + x].x = odd.x - t2.x;
					dest[(j * n + k + n / 2) * N + x].y = odd.y - t2.y;

                }
            }
        }
    }
}


int main() {

	printf("Error measurement start...\n");
	// Set Random Number Seed
	int seed = (unsigned) time(NULL) % 100;
	printf("Random seed: %d\n", seed);

	// Random Number Generation
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * N, r, 0.0, 1.0);
	vslDeleteStream(&stream);

	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++) {
			srcData[i][j].x = r2cData[i][j] = r[i * N + j];
			destData[i][j].x = destData[i][j].y = 0;
		}
	
	// Using oneMKL to do FFT
	DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
	MKL_LONG status;
	MKL_LONG dim_sizes[2] = {N, N};
	MKL_LONG rs[3] = {0, N, 1};
    MKL_LONG cs[3] = {0, N/2+1, 1};

	status = DftiCreateDescriptor(&my_desc_handle, DFTI_SINGLE, DFTI_REAL, 2, dim_sizes);
	status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiSetValue(my_desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, rs);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, cs);
	status = DftiCommitDescriptor(my_desc_handle);
	status = DftiComputeForward(my_desc_handle, r2cData, r2cResult);

	status = DftiFreeDescriptor(&my_desc_handle);

	// Inplement the naive FFT
	naiveFFT(srcData, destData);

	float error_sum = 0.0;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N / 2 + 1; j++) {
			int index = i * (N /2 + 1) + j;
            float re_got = r2cResult[index].real;
            float im_got = r2cResult[index].imag;
			// printf("%.4f+%.4fj ", destData[i][j].x, destData[i][j].y);
			// printf("%.4f+%.4fj\n", re_got, im_got);

			//Compute the error
			float dx = destData[i][j].x - re_got;
			float dy = destData[i][j].y - im_got;
			error_sum += sqrt(dx * dx + dy * dy) / sqrt(destData[i][j].x * destData[i][j].x + destData[i][j].y * destData[i][j].y);
		}
	}

	// Output the error result
	float avg_error = error_sum / N / (N / 2 + 1);
	printf("Average error: %.4f\n", avg_error);
	if (avg_error < 0.05) printf("The result is correct!\n");
	else printf("The result is incorrect!\n");

	// Compare the efficiency
	printf("Efficiency comparison begin...\n");
	printf("Amount of test: %d\n", test_num);
	printf("Input matrix size: %d x %d\n", N, N);

	struct timeb sTimeMil, tTimeMil;
	time_t sTimeSec, tTimeSec; 
	float sumTime1 = 0, sumTime2 = 0;
	for(int i = 0; i < test_num; i++) {
		VSLStreamStatePtr stream;
		vslNewStream(&stream, VSL_BRNG_MT19937, i);
		vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * N, r, 0.0, 1.0);
		vslDeleteStream(&stream);

		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++) {
				srcData[i][j].x = r2cData[i][j] = r[i * N + j];
				destData[i][j].x = destData[i][j].y = 0;
			}
		
		DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
		MKL_LONG status;
		MKL_LONG dim_sizes[2] = {N, N};
		MKL_LONG rs[3] = {0, N, 1};
		MKL_LONG cs[3] = {0, N/2+1, 1};

		status = DftiCreateDescriptor(&my_desc_handle, DFTI_SINGLE, DFTI_REAL, 2, dim_sizes);
		status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
		status = DftiSetValue(my_desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
		status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, rs);
		status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, cs);
		status = DftiCommitDescriptor(my_desc_handle);

		// oneMKL FFT time consumption
		ftime(&sTimeMil);
		time(&sTimeSec);
		status = DftiComputeForward(my_desc_handle, r2cData, r2cResult);
		ftime(&tTimeMil);
		time(&tTimeSec);
		status = DftiFreeDescriptor(&my_desc_handle);
		sumTime1 += (float) (tTimeMil.millitm - sTimeMil.millitm + (float) (1000 * (tTimeSec - sTimeSec)));

		// naive FFT time consumption
		ftime(&sTimeMil);
		time(&sTimeSec);
		naiveFFT(srcData, destData);
		ftime(&tTimeMil);
		time(&tTimeSec);
		sumTime2 += (float) (tTimeMil.millitm - sTimeMil.millitm + (float) (1000 * (tTimeSec - sTimeSec)));
	}

	// Output the efficiency result
	printf("The naive FFT spend %.4f miliseconds per running in average \n", sumTime2 / test_num);
	printf("The oneMKL FFT spend  %.4f miliseconds per running in average\n", sumTime1 / test_num);

	return 0;
}
