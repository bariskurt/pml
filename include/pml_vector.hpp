#ifndef PML_VECTOR_H_
#define PML_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_psi.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define DEFAULT_PRECISION 6

namespace pml {

  // Helpers:
  inline void ASSERT_TRUE(bool condition, const std::string &message) {
    if (!condition) {
      std::cout << "FATAL ERROR: " << message << std::endl;
      exit(-1);
    }
  }

  // Defines a series from start to stop exclusively with step size step:
  // [start, start+step, start+2*step, ...]
  struct Range {
      Range(size_t start_, size_t stop_, size_t step_ = 1)
              : start(start_), stop(stop_), step(step_) { }
      size_t start, stop, step;
  };

  inline bool fequal(double a, double b) {
    return fabs(a - b) < 1e-6;
  }


  inline double add(double x, double y){ return x + y; }
  inline double sub(double x, double y){ return x - y; }
  inline double mul(double x, double y){ return x * y; }
  inline double div(double x, double y){ return x / y; }

  class Vector {

    public:

      template <bool is_const = true>
        class generic_iterator {

          friend class generic_iterator<true>;

          typedef typename std::conditional<is_const, const double*,
              double*>::type ValuePointerType;

          typedef typename std::conditional<is_const, const double&,
              double&>::type ValueReferenceType;

        public:
          generic_iterator<is_const>(ValuePointerType data_, size_t stride_)
              : data(data_), stride(stride_){}

          generic_iterator<is_const>(
              const generic_iterator<false>& it)
              : data(it.data), stride(it.stride){}

          bool operator== (const generic_iterator& other) const {
            return data == other.data;
          }

          bool operator!= (const generic_iterator& other) const {
            return data != other.data;
          }

          ValueReferenceType operator*() {
            return *data;
          }

          generic_iterator& operator--(){
            data -= stride;
            return *this;
          }

          generic_iterator& operator++(){
            data += stride;
            return *this;
          }

          generic_iterator operator--(int){
            generic_iterator temp(*this);
            data -= stride;
            return temp;
          }

          generic_iterator operator++(int){
            generic_iterator temp(*this);
            data += stride;
            return temp;
          }

        private:
          ValuePointerType data;
          size_t stride;
      };

      typedef generic_iterator<true> const_iterator;
      typedef generic_iterator<false> iterator;

    public:

      class const_view {

        public:
          const_view(const double *data, size_t size, size_t stride = 1):
                  cdata_(data), size_(size), stride_(stride){}

          const_view(const Vector &vector)
                  : cdata_(vector.data()), size_(vector.size()), stride_(1){}

        public:
          Vector apply(double (*func)(double)) const {
            Vector result(*this);
            view(result).apply(func);
            return result;
          }

          Vector apply(double value,
                       double (*func)(double, double)) const {
            Vector result(*this);
            view(result).apply(value, func);
            return result;
          }

          Vector apply(const const_view &cview,
                       double (*func)(double, double)) const {
            Vector result(*this);
            view(result).apply(cview, func);
            return result;
          }

        public:

          // ------ Operators -------
          Vector operator+(double value) const { return apply(value, add); }
          Vector operator-(double value) const { return apply(value, sub); }
          Vector operator*(double value) const { return apply(value, mul); }
          Vector operator/(double value) const { return apply(value, div); }

          Vector operator+(const const_view &cview) const {
            return apply(cview, add);
          }

          Vector operator-(const const_view &cview) const {
            return apply(cview, sub);
          }

          Vector operator*(const const_view &cview) const {
            return apply(cview, mul);
          }

          Vector operator/(const const_view &cview) const {
            return apply(cview, div);
          }

          // ----- Comparison -------
          bool operator==(const const_view &other) const {
            if( size() != other.size() )
              return false;
            auto it2 = other.begin();
            for(auto it1 = begin(); it1 != end(); ++it1, ++it2)
              if( ! fequal(*it1, *it2) )
                return false;
            return true;
          }

          bool operator!=(const const_view &other) const {
            return !(*this == other);
          }

          bool operator==(double value) const {
            for(auto it = begin(); it != end(); ++it)
              if( !fequal(*it, value) )
                return false;
            return true;
          }

          bool operator!=(double value) const {
            return !(*this == value);
          }

          // ----- Iterators -------
          const_iterator begin() const {
            return const_iterator(cdata_, stride_);
          }

          const_iterator end() const {
            return const_iterator(cdata_ + size_*stride_, stride_);
          }

          // ----- Getters -----
          size_t size() const {
            return size_;
          }

          size_t stride() const {
            return stride_;
          }

          // ------ Save ----
          void saveTxt(const std::string &filename,
                       int precision = DEFAULT_PRECISION) const {
            std::ofstream ofs(filename);
            if (ofs.is_open()) {
              ofs << 1 << std::endl;      // dimension
              ofs << size() << std::endl; // size
              ofs << std::setprecision(precision) << std::fixed;
              for (auto it = begin(); it != end(); ++it) {
                ofs << *it << std::endl;
              }
              ofs.close();
            }
          }

          void save(const std::string &filename) const {
            std::ofstream ofs(filename, std::ios::binary | std::ios::out);
            if (ofs.is_open()) {
              double dim = 1;
              double length = size();
              ofs.write(reinterpret_cast<char*>(&dim), sizeof(double));
              ofs.write(reinterpret_cast<char*>(&length), sizeof(double));
              if(stride() == 1) {
                ofs.write(reinterpret_cast<const char *>(cdata_),
                          sizeof(double) * length);
              } else {
                for(size_t i = 0; i < size_ * stride_; i+=stride_)
                  ofs.write(reinterpret_cast<const char *>(&cdata_[i]),
                            sizeof(double));
              }
              ofs.close();
            }
          }

          friend std::ostream& operator<<(std::ostream &out,
                                          const const_view &v) {
            out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
            for (auto value : v) {
              out << value << "  ";
            }
            return out;
          }

        protected:
          const double *cdata_;
          size_t size_;
          size_t stride_;
      };

      class view : public const_view{

        public:
          view(double *data, size_t size, size_t stride = 1) :
              const_view(data, size, stride), data_(data){}

          explicit view(Vector &v) : const_view(v), data_(v.data()){}

        public:
          void apply(double (*func)(double)){
            for(auto it = begin(); it != end(); ++it)
              *it = func(*it);
          }

          void apply(double value, double (*func)(double, double)){
            for(auto it = begin(); it != end(); ++it)
              *it = func(*it, value);
          }

          void apply(const const_view &cv,
                     double (*func)(double, double)){
            ASSERT_TRUE(size() == cv.size(),
                        "apply:: Vector view sizes mismatch.");
            auto it1 = begin();
            auto it2 = cv.begin();
            for(; it1 != end(); ++it1, ++it2)
              *it1 = func(*it1, *it2);
          }

          // ----- Iterators -------
          iterator begin() {
            return iterator(data_, stride_);
          }

          iterator end() {
            return iterator(data_ + size_*stride_, stride_);
          }

        public:
          // ------ Self Assignment Operators -------
          void operator+=(double value) { apply(value, add); }
          void operator-=(double value) { apply(value, sub); }
          void operator*=(double value) { apply(value, mul); }
          void operator/=(double value) { apply(value, div); }
          void operator+=(const const_view &other) { apply(other, add); }
          void operator-=(const const_view &other) { apply(other, sub); }
          void operator*=(const const_view &other) { apply(other, mul); }
          void operator/=(const const_view &other) { apply(other, div); }

          view& operator=(const const_view &cv){
            ASSERT_TRUE(size() == cv.size(),
                        "apply:: Vector view sizes mismatch.");
            auto it = begin();
            auto cit = cv.begin();
            for(; it != end(); ++it, ++cit)
              *it = *cit;
            return *this;
          }

          friend std::istream& operator>>(std::istream &in, view &v) {
            for (auto &value : v) {
              in >> value;
            }
            return in;
          }

        private:
          double *data_;
      };

    public:
      // Empty Vector
      Vector() { }

      // Vector of given length and default value.
      explicit Vector(size_t length, double value = 0)
          : data_(length, value) { }

      // Vector from given array
      Vector(size_t length, const double *values)
          : data_(length) {
        memcpy(this->data(), values, sizeof(double) * length);
      }

      // Vector from initializer lsit
      Vector(const std::initializer_list<double> &values)
          : data_(values) { }

      // Vector from const view
      Vector(const const_view &v) : data_(v.size()){
        view(*this) = v;
      }

      Vector& operator=(const const_view &v){
        data_.resize(v.size());
        view(*this) = v;
        return *this;
      }

      // Vector of zeros of given length.
      static Vector zeros(size_t length) {
        return Vector(length, 0.0);
      }

      // Vector of ones of given length.
      static Vector ones(size_t length) {
        return Vector(length, 1.0);
      }

    public:
      // Returns length of the Vector.
      size_t size() const {
        return data_.size();
      }

      // Checks empty.
      bool empty() const {
        return data_.empty();
      }

      // Vector resize. If new size is smaller, the data_ is cropped.
      // If new size is larger, garbage values are appended.
      void resize(size_t new_size) {
        data_.resize(new_size);
      }

    public:
      // Append a single value. (same as push_back)
      void append(double value) {
        data_.push_back(value);
      }

      // Append a Vector
      void append(const const_view &cv) {
        for(auto it = cv.begin(); it != cv.end(); ++it)
          data_.push_back(*it);
      }

    public:
      bool operator==(const Vector& v) const {
        return const_view(*this) == const_view(v);
      }

      bool operator!=(const Vector& v) const {
        return const_view(*this) != const_view(v);
      }

      bool operator==(double value) const {
        return const_view(*this) == value;
      }

      bool operator!=(double value) const {
        return const_view(*this) != value;
      }


    public:
      //  -------- Iterators--------
      std::vector<double>::iterator begin() {
        return data_.begin();
      }

      std::vector<double>::const_iterator begin() const {
        return data_.cbegin();
      }

      std::vector<double>::iterator end() {
        return data_.end();
      }

      std::vector<double>::const_iterator end() const {
        return data_.cend();
      }

      // ------- Accessors -------

      double &operator[](const size_t i0) {
        return data_[i0];
      }

      double operator[](const size_t i0) const {
        return data_[i0];
      }

      double &operator()(const size_t i0) {
        return data_[i0];
      }

      double operator()(const size_t i0) const {
        return data_[i0];
      }

      double* data() {
        return data_.data();
      }

      const double *data() const {
        return data_.data();
      }

      double first() const {
        return data_.front();
      }

      double& first() {
        return data_.front();
      }

      double last() const {
        return data_.back();
      }

      double& last() {
        return data_.back();
      }

      view slice(size_t start, size_t stop, size_t step = 1) {
        double *slice_start = &data_[start];
        size_t slice_length = std::ceil((double)(stop - start) / step);
        return view(slice_start, slice_length, step);
      }

      const_view slice(size_t start, size_t stop, size_t step = 1) const {
        const double *slice_start = &data_[start];
        size_t slice_length = std::ceil((double)(stop - start) / step);
        return const_view(slice_start, slice_length, step);
      }

    public:

      // ------ Self Assignment Operators -------
      // A = A + b
      void operator+=(double value) { view(*this) += value; }

      // A = A - b
      void operator-=(double value) { view(*this) -= value; }

      // A = A * b
      void operator*=(double value) { view(*this) *= value; }

      // A = A / b
      void operator/=(double value) { view(*this) /= value; }

      // A = A + B
      void operator+=(const Vector &other) { view(*this) += const_view(other); }

      // A = A - B
      void operator-=(const Vector &other) { view(*this) -= const_view(other); }

      // A = A * B (elementwise)
      void operator*=(const Vector &other) { view(*this) *= const_view(other); }

      // A = A / B (elementwise)
      void operator/=(const Vector &other) { view(*this) /= const_view(other); }

      // B = A + b
      Vector operator+(double value) const { return const_view(*this) + value; }

      // B = A - b
      Vector operator-(double value) const { return const_view(*this) - value; }

      // B = A * b
      Vector operator*(double value) const { return const_view(*this) * value; }

      // B = A / b
      Vector operator/(double value) const { return const_view(*this) / value; }

      // B =  b + A
      friend Vector operator+(double value, const Vector &x) {
        return x + value;
      }

      // B = b * A
      friend Vector operator*(double value, const Vector &x) {
        return x * value;
      }

      // B = b - A
      friend Vector operator-(double value, const Vector &x) {
        return (-1 * x) + value;
      }

      // B = b / A
      friend Vector operator/(double value, const Vector &x) {
        Vector result(x);
        for (auto &d : result) { d = value / d; }
        return result;
      }

      // ------ Vector - Vector Operations -------

      // R = A + B

      friend Vector operator+(const Vector &x, const Vector &y) {
        return const_view(x) + const_view(y);
      }

      // R = A - B
      friend Vector operator-(const Vector &x, const Vector &y) {
        return const_view(x) - const_view(y);
      }

      // R = A * B (elementwise)
      friend Vector operator*(const Vector &x, const Vector &y) {
        return const_view(x) * const_view(y);
      }

      // R = A / B (elementwise)
      friend Vector operator/(const Vector &x, const Vector &y) {
        return const_view(x) / const_view(y);
      }

      // Load and Save
      friend std::ostream& operator<<(std::ostream &out, const Vector &x) {
        out << const_view(x);
        return out;
      }

      friend std::istream& operator>>(std::istream &in, Vector &x) {
        Vector::view vx(x);
        in >> vx;
        return in;
      }

      void save(const std::string &filename){
        const_view(*this).save(filename);
      }

      void saveTxt(const std::string &filename,
                   int precision = DEFAULT_PRECISION) const {
        const_view(*this).saveTxt(filename, precision);
      }

      static Vector load(const std::string &filename){
        Vector result;
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (ifs.is_open()) {
          double dim, size;
          ifs.read(reinterpret_cast<char*>(&dim), sizeof(double));
          ASSERT_TRUE(dim == 1, "Vector::load:: Dimension mismatch.");
          ifs.read(reinterpret_cast<char*>(&size), sizeof(double));
          result.resize(size);
          ifs.read(reinterpret_cast<char*>(result.data()), sizeof(double)*size);
          ifs.close();
        }
        return result;
      }

      static Vector loadTxt(const std::string &filename) {
        Vector result;
        std::ifstream ifs(filename);
        size_t buffer;
        if (ifs.is_open()) {
          // Read dimension
          ifs >> buffer;
          ASSERT_TRUE(buffer == 1, "Vector::LoadTxt:: Dimension mismatch.");
          ifs >> buffer;
          // Allocate memory
          result.data_.resize(buffer);
          ifs >> result;
          ifs.close();
        }
        return result;
      }

    public:
      std::vector<double> data_;
  };


  //Min
  inline double min(const Vector::const_view &cv) {
    return *(std::min_element(cv.begin(), cv.end()));
  }

  // Max
  inline double max(const Vector::const_view &cv) {
    return *(std::max_element(cv.begin(), cv.end()));
  }

  // Sum
  inline double sum(const Vector::const_view &cv){
    return std::accumulate(cv.begin(), cv.end(), 0.0);
  }

  // Power
  inline Vector pow(const Vector::const_view &cv, double p = 2){
    return cv.apply(p ,std::pow);
  }

  // Dot product
  inline double dot(const Vector::const_view &x, const Vector::const_view &y) {
    return sum(x * y);
  }

  // Mean
  inline double mean(const Vector::const_view &cv){
    return sum(cv) / cv.size();
  }

  // Variance
  inline double var(const Vector::const_view &cv){
    return sum(pow(cv - mean(cv), 2)) / (cv.size() - 1);
  }

  // Standard deviation
  inline double stdev(const Vector::const_view &cv){
    return std::sqrt(var(cv));
  }

  // Exponential
  inline Vector exp(const Vector::const_view &cv){
    return cv.apply(std::exp);
  }

  // Logarithm
  inline Vector log(const Vector::const_view &cv){
    return cv.apply(std::log);
  }

  // Normalize
  inline Vector normalize(const Vector::const_view &cv) {
    return cv / sum(cv);
  }

  // Safe normalize(exp(x))
  inline Vector normalizeExp(const Vector::const_view &cv) {
    return normalize(exp(cv - max(cv)));
  }

  // Safe log(sum(exp(x)))
  inline double logSumExp(const Vector::const_view &cv) {
    double cv_max = max(cv);
    return cv_max + std::log(sum(exp(cv - cv_max)));
  }

  // Absolute value of x
  inline Vector abs(const Vector::const_view &cv){
    return cv.apply(std::fabs);
  }

  // Round to nearest integer
  inline Vector round(const Vector::const_view &cv){
    return cv.apply(std::round);
  }

  // Ceiling
  inline Vector ceil(const Vector::const_view &cv){
    return cv.apply(std::ceil);
  }

  // Floor
  inline Vector floor(const Vector::const_view &cv){
    return cv.apply(std::floor);
  }


  Vector cat(const Vector::const_view &cv1, const Vector::const_view &cv2){
    Vector result(cv1);
    result.append(cv2);
    return result;
  }

  // KL Divergence Between Vectors
  // ToDo: Maybe move to linalgebra
  inline double kl_div(const Vector::const_view &x,
                       const Vector::const_view &y) {
    ASSERT_TRUE(x.size() == y.size(), "kl_div:: Size mismatch.");
    double result = 0;
    auto it1 = x.begin();
    auto it2 = y.begin();
    for(; it1 != x.end(); ++it1, ++it2){
      if(*it1 > 0 && *it2 > 0){
        result += *it1 * (std::log(*it1) - std::log(*it2)) - *it1 + *it2;
      } else if(*it1 == 0 && *it2 >= 0){
        result += *it2;
      } else {
        result += std::numeric_limits<double>::infinity();
        break;
      }
    }
    return result;
  }

} // namespace pml

#endif