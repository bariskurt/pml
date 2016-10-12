#ifndef PML_BLOCK_H_
#define PML_BLOCK_H_

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

  class Block{

    public:
      Block(size_t size){
        size_ = size;
        capacity_ = size;
        stride_ = 1;
        owner = true;
        data = new double[capacity_];
      }

      Block(double *data, size_t size, size_t stride):
          data_(data), size_(size), stride_(stride), owner(false) { }

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

      // -------- Accessors --------
      inline double& operator[](const size_t i) {
        return data_[i * stride];
      }

      inline double operator[](const size_t i) const {
        return data_[i * stride];
      }

      // --------- Setters -------
      void append(double value){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
      }

      void resize(size_t new_size){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
      }

      void realloc(size_t new_capacity){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");

      }

      // -------- Getters --------
      double* data() {
        return data_;
      }

      const double *data() const {
        return data_;
      }

      size_t size(){
        return size_;
      }

      bool is_owner(){
        return owner_;
      }

    private:
      double *data_;
      size_t size_;
      size_t capacity_;
      size_t stride_;
      bool owner;
  };

}

#endif