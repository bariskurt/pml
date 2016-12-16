#ifndef PML_TIME_H_
#define PML_TIME_H_

#include <ctime>
#include <string>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

namespace pml {

  // A unified Time strucre for pml library.
  // Provides Time arithmetic and comparison.
  class Time {
    public:
      Time(): sec(0), usec(0) {}

      Time(uint32_t sec_, uint32_t usec_) : sec(sec_), usec(usec_){}

      // Time from microseconds.
      explicit Time(uint64_t microseconds_) {
        *this = microseconds_;
      }

      // Time from string: ssssssssss.mmmmmm  (seconds.microseconds)
      explicit Time(std::string str) {
        char *p;
        sec = (uint32_t)std::strtoul(str.c_str(), &p, 10);
        usec = (uint32_t)std::strtoul(str.c_str() + 11, &p, 10);
      }


      // A time value that is accurate to the nearest
      // microsecond but also has a range of years
      //
      // struct timeval
      // {
      //    __time_t tv_sec;        // Seconds.
      //    __suseconds_t tv_usec;  // Microseconds.
      // };
      explicit Time(const struct timeval &tval) :
              sec(tval.tv_sec), usec(tval.tv_usec){}

      // struct timespec
      // {
      //    __time_t tv_sec;            // Seconds.
      //    __syscall_slong_t tv_nsec;  // Nanoseconds.
      // };
      explicit Time(const struct timespec &tspec) :
              sec(tspec.tv_sec), usec(tspec.tv_nsec / (uint32_t)1e3){}

      // Time to microseconds.
      uint64_t microseconds() const{
        return (usec + sec * (uint64_t) 1e6);
      }

      void operator=(const struct timeval &t) {
        sec = t.tv_sec;
        usec = t.tv_usec;
      }

      void operator=(const struct timespec &t) {
        sec = t.tv_sec;
        usec = t.tv_nsec / (uint32_t)1e3;
      }

      void operator=(uint64_t microseconds_) {
        sec = (uint32_t) (microseconds_ / (uint32_t)1e6);
        usec = (uint32_t)(microseconds_ % (uint32_t)1e6);
      }

      // Time arithmetic:
      Time operator+(const Time& t) const {
        return Time(microseconds() + t.microseconds());
      }

      Time operator-(const Time& t) const {
        return Time(microseconds() - t.microseconds());
      }

      bool operator<(const Time &t) const {
        return microseconds() < t.microseconds();
      }

      bool operator>(const Time &t) const {
        return microseconds() > t.microseconds();
      }

      // Time arithmetic with microseonds
      bool operator < (uint64_t microseconds_) {
        return microseconds() < microseconds_;
      }

      bool operator > (uint64_t microseconds_) {
        return microseconds() > microseconds_;
      }

      // Time to string of format "YYYY-MM-DD_HH:MM::SS.mmmmmm"
      std::string to_date() const {
        time_t t = sec;
        char date[256], microseconds_[8];
        strftime(date, sizeof(date), "%F_%T", localtime(&t));
        sprintf(microseconds_, ".%06u", usec);
        return std::string(date) + std::string(microseconds_);
      }

      // Time to string of format "ssssssssss.mmmmmm"
      std::string to_string() const {
        char buffer[250];
        snprintf(buffer, 32, "%010u.%06u", sec, usec);
        return std::string(buffer);
      }

      // Returns microseconds from the EPOCH for the current moment.
      static Time now() {
        struct timespec ts;
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        clock_serv_t cclock;
        mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        ts.tv_sec = mts.tv_sec;
        ts.tv_nsec = mts.tv_nsec;
#else
        clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
        return Time(ts);
      }

    public:
      uint32_t sec;   // seconds
      uint32_t usec;  // microseconds
  };


  // A Tic-Toc timer class.
  class TicTocTimer {
    public:

      TicTocTimer(){
        tic();
      }

      // Start timer.
      void tic() {
        t_start = Time::now();
      }

      // Return time elapsed since last tic.
      Time toc() {
        return  Time::now()  - t_start;
      }

    private:
      Time t_start;
  };


  class ProgressBar{

    public:
      ProgressBar(uint64_t max_iter_):
              max_iter(max_iter_), current_iter(0), current_percentage(0){
        timer.tic();
        draw();
      }

      void update(uint64_t increment = 1){
        current_iter += increment;
        uint16_t next_percentage = 100 * ((double)current_iter/max_iter);
        if(next_percentage > current_percentage){
          current_percentage = next_percentage;
          draw();
        }
      }

      void finished(){
        update(max_iter - current_iter);
      }

    private:
      void draw(){
        //printf("\r%3lu %% [", current_percentage);
        std::cout<<"\r"<< current_percentage << "% [";
        std::cout << std::string(current_percentage, '=')
        << std::string(100-current_percentage, ' ')
        << "] Elapsed time: "
        << timer.toc().to_string() << " seconds.";
        if( current_percentage == 100 ){
          std::cout << std::endl;
        }
        fflush(stdout);
      }

    private:
      uint64_t max_iter;
      uint64_t current_iter;
      uint64_t current_percentage;

      // Timer:
      TicTocTimer timer;
  };

} // namespace pml

#endif