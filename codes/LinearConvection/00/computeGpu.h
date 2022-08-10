#pragma once
// Forward declarations
extern "C" {
void initData( float *data, int N );
void computeGPU(float *hostData, int blockSize, int gridSize);
float sum(float *data, int size);
}
