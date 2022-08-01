#pragma once

#include <omp.h>
#include <string>

class TimeOmp
{
public:
    TimeOmp();
    ~TimeOmp();
public:
    void Restart();
    void Stop();

    double ElapsedMilliseconds();
    double ElapsedSeconds();
    void ShowTimeSpan( const std::string & title = "" );
private:
    double time_old;
    double time_now;
    bool   bRunning = false;
};