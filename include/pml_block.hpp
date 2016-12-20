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

  class Block {

    public:
      template<bool is_const_iterator = true>
      class const_noconst_iterator{

        typedef typename std::conditional<is_const_iterator,
            const double&, double&>::type ValueReferenceType;

        public:
          const_noconst_iterator(double *data_){
            data = data_;
          }

          const_noconst_iterator(const const_noconst_iterator<false>& it) {
            data = it.data;
          }

          const_noconst_iterator& operator=(
              const const_noconst_iterator<false>& it){
            data = it.data;
            return *this;
          }

          bool operator==(const const_noconst_iterator& it){
            return data == it.data;
          }

          bool operator!=(const const_noconst_iterator& it){
            return data != it.data;
          }

          const_noconst_iterator& operator++(){
            ++data;
            return *this;
          }

          ValueReferenceType& operator*(){
            return *data;
          }

          friend class const_noconst_iterator<true>;

        private:
            double *data;
      };
      typedef const_noconst_iterator<false> iterator;
      typedef const_noconst_iterator<true> const_iterator;

    public:
      static const size_t INITIAL_CAPACITY = 128; // start with 1K memory
      static const double GROWTH_RATIO;

      explicit Block(size_t size = 0) : data_(nullptr), size_(size){
        realloc_data_(std::max(size, INITIAL_CAPACITY));
      }

      Block(const Block &that) : Block(that.size_){
        memcpy(data_, that.data_, sizeof(double) * that.size_);
      }

      Block(Block &&that) : data_(that.data_),
                            capacity_(that.capacity_),
                            size_(that.size_) {
        that.data_ = nullptr;
        that.size_ = 0;
        that.capacity_ = 0;
      }

      Block& operator=(const Block &that) {
        if( &that != this ){
          free_data_();
          realloc_data_(that.capacity_);
          memcpy(data_, that.data_, sizeof(double) * that.size_);
          size_ = that.size_;
        }
        return *this;
      }

      Block& operator=(Block &&that) {
        free_data_();
        // hijack the data of that block
        data_ = that.data_;
        size_ = that.size_;
        capacity_ = that.capacity_;
        // release that block's data
        that.data_ = nullptr;
        that.size_ = 0;
        that.capacity_ = 0;
        return *this;
      }

      ~Block() {
        if( data_ )
          free_data_();
      }

    public:

      double* data() {
        return data_;
      }

      const double* data() const {
        return data_;
      }

      size_t size() const {
        return size_;
      }

      size_t capacity() const {
        return capacity_;
      }

      bool empty() const {
        return size_ == 0;
      }

      void clear() {
        size_ = 0;
      }

    protected:

      void __resize__(size_t new_size) {
        if( new_size > capacity_ )
          realloc_data_(new_size);
        size_ = new_size;
      }

      void __reserve__(size_t new_capacity) {
        if (new_capacity > capacity_)
          realloc_data_(new_capacity);
      }

      void __shrink_to_fit__() {
        if (size_ < capacity_)
          realloc_data_(size_);
      }

    public:
      // -------- Iterators --------
      iterator begin() {
        return iterator(data_);
      }

      const_iterator begin() const {
        return const_iterator(data_);
      }

      const_iterator cbegin() const {
        return const_iterator(data_);
      }

      iterator end() {
        return iterator(data_ + size_);
      }

      const_iterator end() const {
        return const_iterator(data_ + size_);
      }

      const_iterator cend() const {
        return const_iterator(data_ + size_);
      }

      // -------- Accessors --------
      inline double &operator[](const size_t i) {
        return data_[i];
      }

      inline double operator[](const size_t i) const {
        return data_[i];
      }

    public:
      void push_back(double value) {
        if (size_ == capacity_) {
          realloc_data_(capacity_ * GROWTH_RATIO);
        }
        data_[size_++] = value;
      }

      void push_back(const Block &block) {
        if (size_ + block.size() > capacity_) {
          realloc_data_( (size_ + block.size_) * GROWTH_RATIO);
        }
        double *source = &block == this ? data_ : block.data_;
        memcpy(data_ + size_, source, sizeof(double) * block.size_);
        size_ += block.size_;
      }

      void pop_back(){
        if(size_ > 0)
          --size_;
      }

      void fill(double value){
        std::fill(this->begin(), this->end(), value);
      }

    private:
      void realloc_data_(size_t new_capacity) {
        data_ = (double*) realloc(data_, sizeof(double) * new_capacity);
        capacity_ = new_capacity;
      }

      void free_data_(){
        if( data_ ){
          free(data_);
          data_ = nullptr;
        }
      }

    protected:
      double *data_;
      size_t capacity_;
      size_t size_;
  };
  const double Block::GROWTH_RATIO  = 1.5;
}

#endif