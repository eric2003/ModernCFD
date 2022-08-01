#pragma once

#include <chrono>
#include <string>

class TimeSpan
{
public:
    using clock_type = std::chrono::time_point<std::chrono::system_clock>;
public:
    TimeSpan();
    ~TimeSpan();
public:
    void Restart();
    void Stop();

    double ElapsedMilliseconds();
    double ElapsedSeconds();
    void ShowTimeSpan( const std::string & title = "" );
public:
    void RunTest();
private:
    clock_type time_old;
    clock_type time_now;
    bool       bRunning = false;
};