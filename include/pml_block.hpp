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
          __free_data__();
          __realloc_data__(that.capacity_);
          memcpy(data_, that.data_, sizeof(double) * that.size_);
          size_ = that.size_;
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

      void clear() {
        size_ = 0;
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

    public:
      // -------- Apply --------
      void apply(double (*func)(const double&)){
        for(size_t i=0; i < size_; ++i)
          data_[i] = func(data_[i]);
      }

      void apply(double (*func)(const double&, const double&),
                 const double value){
        for(size_t i=0; i < size_; ++i)
          data_[i] = func(data_[i], value);
      }

      void apply(double (*func)(const double&, const double&),
                 const  double* other_data_){
        for(size_t i=0; i < size_; ++i)
          data_[i] = func(data_[i], other_data_[i]);
      }

    public:
      // ---------- Any / Or ---------
      friend bool any(const Block &b){
        for(size_t i = 0; i < b.size_; ++i)
          if( b.data_[i] == 1 )
            return true;
        return false;
      }

      friend bool all(const Block &b){
        for(size_t i = 0; i < b.size_; ++i)
          if( b.data_[i] == 0 )
            return false;
        return true;
      }

  protected:

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

    private:

      void __realloc_data__(size_t new_capacity) {
        data_ = (double*) realloc(data_, sizeof(double) * new_capacity);
        capacity_ = new_capacity;
      }

      void __free_data__(){
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
  const size_t Block::INITIAL_CAPACITY = 128; // start with 1K memory

}

#endif