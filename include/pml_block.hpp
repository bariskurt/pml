#ifndef PML_BLOCK_H_
#define PML_BLOCK_H_

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace pml {

  inline void ASSERT_TRUE(bool condition, const std::string &message) {
    if (!condition) {
      std::cout << "FATAL ERROR: " << message << std::endl;
      exit(-1);
    }
  }

  class Block {

    friend class PmlTester;

    public:
      static const size_t MINIMUM_CAPACITY;
      static const double GROWTH_RATIO;

    public:
      explicit Block(size_t size = 0) : data_(nullptr), capacity_(0), size_(0) {
        // Allocate size with exact fit in memory
        if(size > 0)
          __resize__(size, true);
      }

      Block(const Block &that) : Block(that.size_){
        memcpy(data_, that.data_, sizeof(double) * that.size_);
      }

      Block(Block &&that)
          : data_(that.data_), capacity_(that.capacity_), size_(that.size_) {
        that.data_ = nullptr;
        that.size_ = 0;
        that.capacity_ = 0;
      }

      Block& operator=(const Block &that) {
        if( &that != this ) {
          if( size_ != that.size_)
            __resize__(that.size_, true);
          memcpy(data_, that.data_, sizeof(double) * that.size_);
        }
        return *this;
      }

      Block& operator=(Block &&that) {
        __free__();
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
        __free__();
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

    public:

      // -------- Accessors --------
      double &operator[](const size_t i) {
        return data_[i];
      }

      double operator[](const size_t i) const {
        return data_[i];
      }

      double &operator()(const size_t i) {
        return data_[i];
      }

      double operator()(const size_t i) const {
        return data_[i];
      }

      void fill(double value){
        for(double *itr = data_; itr < data_ + size_; ++itr)
          *itr = value;
      }

    protected:

      void __push_back__(double value) {
        if (size_ == capacity_) {
          __realloc__(capacity_ * GROWTH_RATIO);
        }
        data_[size_++] = value;
      }

      void __push_back__(const Block &block) {
        if( this == &block ){
          // self push back
          if( size_ > 0){
            __realloc__(2 * size_ * GROWTH_RATIO);
            memcpy(data_ + size_, data_ , sizeof(double) * size_);
          }
        } else {
          if (size_ + block.size() > capacity_)
            __realloc__( (size_ + block.size_) * GROWTH_RATIO );
          memcpy(data_ + size_, block.data_, sizeof(double) * block.size_);
        }
        size_ += block.size_;
      }

      void __pop_back__() {
        if(size_ > 0)
          --size_;
      }

    protected:

      void __clear__() {
        size_ = 0;
      }

      void __resize__(size_t new_size, bool fit_in_memory = false) {
        if( fit_in_memory || new_size > capacity_ )
          __realloc__(new_size);
        size_ = new_size;
      }

      void __reserve__(size_t new_capacity) {
        if (new_capacity > capacity_)
          __realloc__(new_capacity);
      }

      void __shrink_to_fit__() {
          __realloc__(size_);
      }

    private:

      void __realloc__(size_t new_capacity) {
        new_capacity = std::max(MINIMUM_CAPACITY, new_capacity);
        if( new_capacity != capacity_) {
          data_ = (double *) realloc(data_, sizeof(double) * new_capacity);
          capacity_ = new_capacity;
        }
      }

      void __free__(){
        if( data_ ){
          free(data_);
          data_ = nullptr;
          size_ = 0;
          capacity_ = 0;
        }
      }

    protected:

      double *data_;
      size_t capacity_;
      size_t size_;
  };

  const size_t Block::MINIMUM_CAPACITY = 4;
  const double Block::GROWTH_RATIO  = 1.5;

}

#endif