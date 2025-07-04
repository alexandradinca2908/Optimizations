/*
 * Tema 2 ASC
 * 2025 Spring
 * !!! Do not modify this file !!!
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/* 
 * Compares two upper matrixes withtin a tolerance limit
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	N = fileInfo1.st_size / sizeof(double);

	for (i = 0; i < N; i++ ) {
		ret = check_err(mat1[i ], mat2[i ], precision); 
		if (ret != 0) {
			printf("Arrays differ on index [%d]. Expected %.8lf got %.8lf\n",
					i, mat1[i], mat2[i]);
			goto done;
		}
	}

done:
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);

	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
