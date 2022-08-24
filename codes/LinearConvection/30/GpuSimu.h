#pragma once

class GpuSimu
{
public:
    GpuSimu();
    ~GpuSimu();
public:
    void Init(int argc, char **argv);
    void Run();
public:
    int blockSize;
    int gridSize;
    int dataSizePerNode;
    int dataSizeTotal;
    float * dataRoot;
    float * dataNode;
};

