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

int main(int argc, char** argv)
{
    {
		int nThreads = GetThreadCount( argc, argv );
		std::cout << " nThreads = " << nThreads << std::endl;
    }

    return 0;
}