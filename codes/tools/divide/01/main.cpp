#include <iostream>
 
int main( int argc, char **argv )
{
    while ( true )
    {
		int ni_global = -1;
        std::printf( "please input ni_global:\n" );
        std::scanf("%d", &ni_global); 
        if ( ni_global < 0 ) break;
        int nZones = 4;
        int grid_ni = ( ni_global + nZones - 1 ) / nZones;
        int count = 0;
        for ( int i = 0; i < nZones; ++ i )
        {
             count += ( grid_ni - 1 );
        }
        count += 1;
        std::printf( "grid_ni = %d, ni_global = %d, count = %d\n", grid_ni, ni_global, count );
    };
    
    return 0;
}
