#include "TimeSpan.h"
#include <iostream>
#include <thread>

TimeSpan::TimeSpan()
{
    this->Restart();
}

TimeSpan::~TimeSpan()
{
    ;
}

void TimeSpan::Restart()
{
    this->time_old = std::chrono::system_clock::now();
}

void TimeSpan::Stop()
{
    this->time_now = std::chrono::system_clock::now();
}

double TimeSpan::ElapsedMilliseconds()
{
    this->time_now = std::chrono::system_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>( this->time_now - this->time_old ).count();
}

double TimeSpan::ElapsedSeconds()
{
    return this->ElapsedMilliseconds() / 1000.0;
}

void TimeSpan::ShowTimeSpan( const std::string & title )
{
    this->time_now = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = this->time_now - this->time_old;

    std::cout << title << " time elapsed : " << elapsed.count() << " seconds" << "\n";

    this->time_old = this->time_now;
}

void TimeSpan::RunTest()
{
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        << " ns" << std::endl;

    std::cout << "Elapsed time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        << " mus" << std::endl;

    std::cout << "Elapsed time in milliseconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;

    std::cout << "Elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " sec" << std::endl;
}