kernel void ArrayMultReduce( global const float *dA, global const float *dB, local float *prods, global float *dC )
{
    // first the multiplication of arrays
	int gid = get_global_id( 0 ); 						// 0 .. total_array_size-1
    int numItems = get_local_size( 0 ); 				// # work-items per work-group
    int tnum = get_local_id( 0 ); 						// thread (i.e., work-item) number in this work-group
														// 0 .. numItems-1
    int wgNum = get_group_id( 0 ); 						// which work-group number this is in

    prods[ tnum ] = dA[ gid ] * dB[ gid ];				// multiply the 2 arrays and store results in products[tnum]


	// then the reduction - keep on device to avoid passing prods back and forth to memory
	// all threads (i.e., work-items) in work group execute this code simultaneously 
    for( int offset = 1; offset < numItems; offset *= 2 )
    {
        int mask = 2 * offset - 1;						// refer to slide #7 for mask and offset logic
        barrier( CLK_LOCAL_MEM_FENCE );					// wait for completion
        if( ( tnum & mask ) == 0 )						// use offset and mask to perform only log_2(numItems) times
        {
            prods[ tnum ] += prods[ tnum + offset ];	// accumulate thread (i.e., work-item) result
        }       
    }
    
    barrier( CLK_LOCAL_MEM_FENCE );						// wait for all threads to finish			
    if( tnum == 0 )
        dC[ wgNum ] = prods[ 0 ];						// final sum for this work-group
}

