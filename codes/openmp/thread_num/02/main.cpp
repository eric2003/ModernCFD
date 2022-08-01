#include <omp.h>
#include <iostream>
#include <string>

int GetThreadCount( int argc, char** argv )
{
	int nThreads = 1;
	if ( argc < 2 )
	{
		return nThreads;
	}
	std::size_t pos{};
	nThreads = std::stoi(argv[1], &pos);
	return nThreads;
}

void hello()
{
	int my_id = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	std::cout << "Hello from thread " << my_id << " of " << num_threads << "." << std::endl;
}

int main(int argc, char** argv)
{
    {
		int nThreads = GetThreadCount( argc, argv );
		std::cout << " nThreads = " << nThreads << std::endl;
		#pragma omp parallel num_threads( nThreads )
		hello();
    }

    return 0;
}