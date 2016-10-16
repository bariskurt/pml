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


  template <bool is_const = true>
  class generic_vector_iterator {

    friend class generic_vector_iterator<true>;

    typedef typename std::conditional<is_const, const double*,
            double*>::type ValuePointerType;

    typedef typename std::conditional<is_const, const double&,
            double&>::type ValueReferenceType;

    public:
      generic_vector_iterator<is_const>(ValuePointerType data_, size_t stride_)
            : data(data_), stride(stride_){}

      generic_vector_iterator<is_const>(
              const generic_vector_iterator<false>& it)
            : data(it.data), stride(it.stride){}

      bool operator== (const generic_vector_iterator& other) const {
        return data == other.data;
      }

      bool operator!= (const generic_vector_iterator& other) const {
        return data != other.data;
      }

      ValueReferenceType operator*() {
        return *data;
      }

      generic_vector_iterator& operator--(){
        data -= stride;
        return *this;
      }

      generic_vector_iterator &operator++(){
        data += stride;
        return *this;
      }

      generic_vector_iterator operator--(int){
        generic_vector_iterator temp(*this);
        data -= stride;
        return temp;
      }

      generic_vector_iterator operator++(int){
        generic_vector_iterator temp(*this);
        data += stride;
        return temp;
      }

    private:
      ValuePointerType data;
      size_t stride;
  };

  typedef generic_vector_iterator<true> const_vector_iterator;
  typedef generic_vector_iterator<false> vector_iterator;

  class Vector {

    public:
      class ConstVectorView {

        public:
          ConstVectorView(const double *data, size_t size, size_t stride = 1):
                  cdata_(data), size_(size), stride_(stride){}

          ConstVectorView(const Vector &vector)
                  : cdata_(vector.data()), size_(vector.size()), stride_(1){}

        public:
          Vector apply(double (*func)(double)) const {
            Vector result(*this);
            VectorView(result).apply(func);
            return result;
          }

          Vector apply(double value,
                       double (*func)(double, double)) const {
            Vector result(*this);
            VectorView(result).apply(value, func);
            return result;
          }

          Vector apply(const ConstVectorView &cview,
                       double (*func)(double, double)) const {
            Vector result(*this);
            VectorView(result).apply(cview, func);
            return result;
          }

        public:

          // ------ Operators -------
          Vector operator+(double value) const { return apply(value, add); }
          Vector operator-(double value) const { return apply(value, sub); }
          Vector operator*(double value) const { return apply(value, mul); }
          Vector operator/(double value) const { return apply(value, div); }

          Vector operator+(const ConstVectorView &cview) const {
            return apply(cview, add);
          }

          Vector operator-(const ConstVectorView &cview) const {
            return apply(cview, sub);
          }

          Vector operator*(const ConstVectorView &cview) const {
            return apply(cview, mul);
          }

          Vector operator/(const ConstVectorView &cview) const {
            return apply(cview, div);
          }

          // ----- Comparison -------
          bool equals(const ConstVectorView &other) const {
            if( size() != other.size() )
              return false;
            auto it2 = other.begin();
            for(auto it1 = begin(); it1 != end(); ++it1, ++it2)
              if( ! fequal(*it1, *it2) )
                return false;
            return true;
          }

          bool equals(const Vector &v) const {
            return this->equals(ConstVectorView(v));
          }

          // ----- Iterators -------
          const_vector_iterator begin() const {
            return const_vector_iterator(cdata_, stride_);
          }

          const_vector_iterator end() const {
            return const_vector_iterator(cdata_ + size_*stride_, stride_);
          }

          // ----- Getters -----
          size_t size() const {
            return size_;
          }

          size_t stride() const {
            return stride_;
          }

        protected:
          const double *cdata_;
          size_t size_;
          size_t stride_;
      };

      class VectorView : public ConstVectorView{

        public:
          VectorView(double *data, size_t size, size_t stride = 1) :
                  ConstVectorView(data, size, stride), data_(data){}

          explicit VectorView(Vector &v) : ConstVectorView(v), data_(v.data()){}

        public:
          void apply(double (*func)(double)){
            for(auto it = begin(); it < end(); ++it)
              *it = func(*it);
          }

          void apply(double value, double (*func)(double, double)){
            for(auto it = begin(); it < end(); ++it)
              *it = func(*it, value);
          }

          void apply(const ConstVectorView &view,
                     double (*func)(double, double)){
            ASSERT_TRUE(size() == view.size(),
                        "apply:: Vector view sizes mismatch.");
            auto it1 = begin();
            auto it2 = view.begin();
            for(; it1 != end(); ++it1, ++it2)
              *it1 = func(*it1, *it2);
          }

          // ----- Iterators -------
          vector_iterator begin() const {
            return vector_iterator(data, stride);
          }

          vector_iterator end() const {
            return vector_iterator(data + size*stride, stride);
          }

        public:
          // ------ Self Assignment Operators -------
          void operator+=(double value) { apply(value, add); }
          void operator-=(double value) { apply(value, sub); }
          void operator*=(double value) { apply(value, mul); }
          void operator/=(double value) { apply(value, div); }

          void operator+=(const ConstVectorView &other) { apply(other, add); }
          void operator-=(const ConstVectorView &other) { apply(other, sub); }
          void operator*=(const ConstVectorView &other) { apply(other, mul); }
          void operator/=(const ConstVectorView &other) { apply(other, div); }

          void operator=(ConstVectorView &other){
            ASSERT_TRUE(size == other.size,
                        "apply:: Vector view sizes mismatch.");
            for(size_t i=0, j=0; i < size * stride; i+=stride, j+=other.stride)
              data[i] = other.cdata[j];
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

      // Vector from range
      explicit Vector(const ConstVectorView &view) : data_(view.size){
        for(size_t i=0, j=0; i < size(); ++i, j+=view.stride){
          data_[i] = view.cdata[j];
        }
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
      void append(const Vector &v) {
        data_.insert(data_.end(), v.data_.begin(), v.data_.end());
      }

    public:
      friend bool any(const Vector &v){
        for(size_t i = 0; i < v.size(); ++i)
          if( v[i] == 1 )
            return true;
        return false;
      }

      friend bool all(const Vector &v){
        for(size_t i = 0; i < v.size(); ++i)
          if( v[i] == 0 )
            return false;
        return true;
      }

      friend Vector operator==(const Vector &x, double v) {
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], v);
        return result;
      }

      friend Vector operator==(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], y[i]);
        return result;
      }


      friend Vector operator<(const Vector &x, double d) {
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] < d;
        return result;
      }

      friend Vector operator<( double d, const Vector &x) {
        return x > d;
      }

      friend Vector operator<(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] < y[i];
        return result;
      }

      friend Vector operator>(const Vector &x, double d) {
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > d;
        return result;
      }

      friend Vector operator>( double d, const Vector &x) {
        return x < d;
      }

      friend Vector operator>(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > y[i];
        return result;
      }

      bool equals(const Vector &other){
        return ConstVectorView(*this).equals(ConstVectorView(other));
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

      // Vector slice
      /*
      Vector getSlice(size_t start, size_t stop, size_t step = 1) const {
        Vector result;
        for(size_t i = start; i < stop; i+=step){
          result.append(data_[i]);
        }
        return result;
      }
       */

      VectorView slice(size_t start, size_t stop, size_t step = 1) {
        double *slice_start = &data_[start];
        size_t slice_length = std::ceil((double)(stop - start) / step);
        return VectorView(slice_start, slice_length, step);
      }

      ConstVectorView slice(size_t start, size_t stop, size_t step = 1) const {
        const double *slice_start = &data_[start];
        size_t slice_length = std::ceil((double)(stop - start) / step);
        return ConstVectorView(slice_start, slice_length, step);
      }

    public:

      // ------ Self Assignment Operators -------

      void operator+=(double value) { VectorView(*this) += value; }

      // A = A - b
      void operator-=(double value) { VectorView(*this) -= value; }

      // A = A * b
      void operator*=(double value) { VectorView(*this) *= value; }

      // A = A / b
      void operator/=(double value) { VectorView(*this) /= value; }

      // A = A + B
      void operator+=(const Vector &other) {
        VectorView(*this) += ConstVectorView(other);
      }

      // A = A - B
      void operator-=(const Vector &other) {
        VectorView(*this) -= ConstVectorView(other);
      }

      // A = A * B (elementwise)
      void operator*=(const Vector &other) {
        VectorView(*this) *= ConstVectorView(other);
      }

      // A = A / B (elementwise)
      void operator/=(const Vector &other) {
        VectorView(*this) /= ConstVectorView(other);
      }

      // A = A + B
      void operator+=(ConstVectorView &view) { VectorView(*this) += view; }

      // A = A - B
      void operator-=(ConstVectorView &view) { VectorView(*this) -= view; }

      // A = A * B (elementwise)
      void operator*=(ConstVectorView &view) { VectorView(*this) *= view; }

      // A = A / B (elementwise)
      void operator/=(ConstVectorView &view) { VectorView(*this) /= view; }

      // ------ Vector - Double Operations -------

    public:

      // Returns A + b
      friend Vector operator+(const Vector &x, double value) {
        return ConstVectorView(x) + value;
      }

      // Returns b + A
      friend Vector operator+(double value, const Vector &x) {
        return x + value;
      }

      // Returns A * b
      friend Vector operator*(const Vector &x, double value) {
        return ConstVectorView(x) * value;
      }

      // Returns b * A
      friend Vector operator*(double value, const Vector &x) {
        return x * value;
      }

      // Returns A - b
      friend Vector operator-(const Vector &x, double value) {
        return ConstVectorView(x) - value;
      }

      // Returns b - A
      friend Vector operator-(double value, const Vector &x) {
        return (-1 * x) + value;
      }

      // returns A / b
      friend Vector operator/(const Vector &x, double value) {
        return ConstVectorView(x) / value;
      }

      // returns b / A
      friend Vector operator/(double value, const Vector &x) {
        Vector result(x);
        for (auto &d : result) { d = value / d; }
        return result;
      }

      // ------ Vector - Vector Operations -------

      // R = A + B
      friend Vector operator+(const Vector &x, const Vector &y) {
        return ConstVectorView(x) + ConstVectorView(y);
      }

      // R = A - B
      friend Vector operator-(const Vector &x, const Vector &y) {
        return ConstVectorView(x) - ConstVectorView(y);
      }

      // R = A * B (elementwise)
      friend Vector operator*(const Vector &x, const Vector &y) {
        return ConstVectorView(x) * ConstVectorView(y);
      }

      // R = A / B (elementwise)
      friend Vector operator/(const Vector &x, const Vector &y) {
        return ConstVectorView(x) / ConstVectorView(y);
      }

      // Load and Save
      friend std::ostream &operator<<(std::ostream &out,
                                      const Vector &x) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for (auto &value : x) {
          out << value << "  ";
        }
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Vector &x) {
        for (auto &value : x) {
          in >> value;
        }
        return in;
      }

      void save(const std::string &filename){
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        if (ofs.is_open()) {
          double dim = 1;
          double length = size();
          ofs.write(reinterpret_cast<char*>(&dim), sizeof(double));
          ofs.write(reinterpret_cast<char*>(&length), sizeof(double));
          ofs.write(reinterpret_cast<char*>(data()), sizeof(double)*length);
          ofs.close();
        }
      }

      void saveTxt(const std::string &filename,
                   int precision = DEFAULT_PRECISION) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 1 << std::endl;      // dimension
          ofs << size() << std::endl; // size
          ofs << std::setprecision(precision) << std::fixed;
          for (auto &value : data_) {
            ofs << value << std::endl;
          }
          ofs.close();
        }
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

  Vector cat(const Vector &v1, const Vector &v2){
    Vector result(v1);
    result.append(v2);
    return result;
  }

  Vector reverse(const Vector &v){
    Vector result = v;
    std::reverse(result.begin(), result.end());
    return result;
  }

  // Returns the set of indices i of v, such that v[i] == 1.
  Vector find(const Vector &v){
    Vector result;
    for(size_t i=0; i < v.size(); ++i){
      if(v[i] == 1)
        result.append(i);
    }
    return result;
  }

  // is_nan
  Vector is_nan(const Vector &v){
    Vector result = Vector::zeros(v.size());
    for(size_t i=0; i < v.size(); ++i){
      if(std::isnan(v[i]))
        result[i] = 1;
    }
    return result;
  }

  // is_inf
  Vector is_inf(const Vector &v){
    Vector result = Vector::zeros(v.size());
    for(size_t i=0; i < v.size(); ++i){
      if(std::isinf(v[i]))
        result[i] = 1;
    }
    return result;
  }

  // Sum
  inline void print(const Vector::ConstVectorView &w){
    for(auto v : w){
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }

  Vector add(const Vector::ConstVectorView &x,
                   const Vector::ConstVectorView &y) {
    return x + y;
  }

  inline double sum(const Vector &x){
    return std::accumulate(x.begin(), x.end(), 0.0);
  }

  // Power
  inline Vector pow(const Vector &x, double p = 2){
    return Vector::ConstVectorView(x).apply(p ,std::pow);
  }

  //Min
  inline double min(const Vector &x) {
    return *(std::min_element(x.begin(), x.end()));
  }

  // Max
  inline double max(const Vector &x) {
    return *(std::max_element(x.begin(), x.end()));
  }

  // Dot product
  inline double dot(const Vector &x, const Vector &y) {
    return sum(x * y);
  }

  // Mean
  inline double mean(const Vector &x){
    return sum(x) / x.size();
  }

  // Variance
  inline double var(const Vector &x){
    return sum(pow(x - mean(x), 2)) / (x.size() - 1);
  }

  // Standard deviation
  inline double stdev(const Vector &x){
    return std::sqrt(var(x));
  }

  // Absolute value of x
  inline Vector abs(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::fabs);
  }

  // Round to nearest integer
  inline Vector round(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::round);
  }

  // Ceiling
  inline Vector ceil(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::ceil);
  }

  // Floor
  inline Vector floor(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::floor);
  }

  // Exponential
  inline Vector exp(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::exp);
  }

  // Logarithm
  inline Vector log(const Vector &v){
    return Vector::ConstVectorView(v).apply(std::log);
  }

  // Normalize
  inline Vector normalize(const Vector &x) {
    return x / sum(x);
  }

  // Safe normalize(exp(x))
  inline Vector normalizeExp(const Vector &x) {
    return normalize(exp(x - max(x)));
  }

  // Safe log(sum(exp(x)))
  inline double logSumExp(const Vector &x) {
    double x_max = max(x);
    return x_max + std::log(sum(exp(x - x_max)));
  }

  // KL Divergence Between Vectors
  // ToDo: Maybe move to linalgebra
  inline double kl_div(const Vector &x, const Vector &y) {
    ASSERT_TRUE(x.size() == y.size(), "kl_div:: Size mismatch.");
    double result = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if(x(i) > 0 && y(i) > 0){
        result += x(i) * (std::log(x(i)) - std::log(y(i))) - x(i) + y(i);
      } else if(x(i) == 0 && y(i) >= 0){
        result += y(i);
      } else {
        result += std::numeric_limits<double>::infinity();
        break;
      }
    }
    return result;
  }

} // namespace pml

#endif