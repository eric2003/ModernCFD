#include "TimeOmp.h"
#include <iostream>
#include <thread>

TimeOmp::TimeOmp()
{
    this->Restart();
}

TimeOmp::~TimeOmp()
{
    ;
}

void TimeOmp::Restart()
{
    this->time_old = omp_get_wtime();
}

void TimeOmp::Stop()
{
    this->time_now = omp_get_wtime();
}

double TimeOmp::ElapsedMilliseconds()
{
    this->time_now = omp_get_wtime();

    return this->time_now - this->time_old;
}

double TimeOmp::ElapsedSeconds()
{
    return this->ElapsedMilliseconds() / 1000.0;
}

void TimeOmp::ShowTimeSpan( const std::string & title )
{
    this->time_now = omp_get_wtime();

    double elapsed = this->time_now - this->time_old;

    std::cout << title << " time elapsed : " << elapsed << " seconds" << "\n";

    this->time_old = this->time_now;
}