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
      static const size_t INITIAL_CAPACITY;
      static const double GROWTH_RATIO;

    public:
      explicit Block(size_t size = 0) : data_(nullptr), size_(size){
        __realloc_data__(std::max(size, INITIAL_CAPACITY));
      }

      Block(const Block &that) : Block(that.size_){
        __copy_from__(that.data_, that.size_);
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
          __free_data__();
          __copy_from__(that.data_, that.size_);
        }
        return *this;
      }

      Block& operator=(Block &&that) {
        __free_data__();
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
          __free_data__();
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
      inline double &operator[](const size_t i) {
        return data_[i];
      }

      inline double operator[](const size_t i) const {
        return data_[i];
      }

      inline double &operator()(const size_t i) {
        return data_[i];
      }

      inline double operator()(const size_t i) const {
        return data_[i];
      }

      void fill(double value){
        for(double *itr = data_; itr < data_ + size_; ++itr)
          *itr = value;
      }

    protected:

      void __clear__() {
        size_ = 0;
      }

      void __resize__(size_t new_size) {
        if( new_size > capacity_ )
          __realloc_data__(new_size);
        size_ = new_size;
      }

      void __reserve__(size_t new_capacity) {
        if (new_capacity > capacity_)
          __realloc_data__(new_capacity);
      }

      void __shrink_to_fit__() {
        if (size_ < capacity_)
          __realloc_data__(size_);
      }

      void __copy_from__(const double* src, size_t size){
        if( size > capacity_ )
          __reserve__(size);
        memcpy(data_, src, sizeof(double) * size);
        size_ = size;
      }

    protected:

      void __push_back__(double value) {
        if (size_ == capacity_) {
          __realloc_data__(capacity_ * GROWTH_RATIO);
        }
        data_[size_++] = value;
      }

      void __push_back__(const Block &block) {
        if (size_ + block.size() > capacity_) {
          __realloc_data__( (size_ + block.size_) * GROWTH_RATIO);
        }
        double *source = &block == this ? data_ : block.data_;
        memcpy(data_ + size_, source, sizeof(double) * block.size_);
        size_ += block.size_;
      }

      void __pop_back__(){
        if(size_ > 0)
          --size_;
      }

    protected:

      void __realloc_data__(size_t new_capacity) {
        data_ = (double*) realloc(data_, sizeof(double) * new_capacity);
        capacity_ = new_capacity;
      }

      void __free_data__(){
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
  const double Block::GROWTH_RATIO  = 1.5;
  const size_t Block::INITIAL_CAPACITY = 128; // start with 1K memory

}

#endif