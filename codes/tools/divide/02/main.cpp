#include <iostream>
#include <vector>
 
int main( int argc, char **argv )
{
    while ( true )
    {
		int ni_global = -1;
        std::printf( "please input ni_global:\n" );
        std::scanf("%d", &ni_global); 
        if ( ni_global < 0 ) break;
        int nZones = 4;
		std::vector<int> ni_array(4);
        int grid_ni = ( ni_global + nZones - 1 ) / nZones;
		int ni_last = ni_global - ( nZones - 1 ) * ( grid_ni - 1 );
        int count = 0;
        for ( int i = 0; i < nZones; ++ i )
        {
            count += ( grid_ni - 1 );
        }
        count += 1;
		
        for ( int i = 0; i < nZones - 1; ++ i )
        {
			ni_array[i] = grid_ni;
        }
		ni_array[nZones - 1] = ni_last;
		
		int count1 = 0;
        for ( int i = 0; i < nZones; ++ i )
        {
			count1 += ( ni_array[i] - 1 );
        }
		count1 += 1;
		
        std::printf( "grid_ni = %d, ni_global = %d, count = %d, count1 = %d\n", grid_ni, ni_global, count, count1 );
        for ( int i = 0; i < nZones; ++ i )
        {
			std::printf( "%d ", ni_array[i] );
        }
    };
    
    return 0;
}
