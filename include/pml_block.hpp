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

    class iterator{
      public:
        iterator(double *data_, size_t stride_)
            : data(data_), stride(stride_) {}

        iterator(const iterator& it)
            : data(it.data), stride(it.stride) {}

        iterator& operator=(const iterator& it){
          data = it.data;
          stride = it.stride;
          return *this;
        }

        bool operator==(const iterator& it){
          return data == it.data;
        }

        bool operator!=(const iterator& it){
          return data != it.data;
        }

        iterator& operator++(){
          data+=stride;
          return *this;
        }

        double& operator*(){
          return *data;
        }

        double operator*() const{
          return *data;
        }
      private:
        double *data;
        size_t stride;
    };

    public:
      // Allocate a new block, as the owner.
      explicit Block(size_t size = 0)
          :data_(nullptr), size_(size), stride_(1), owner_(true) {
        if( size_ > 0)
          realloc(size_);
      }

      // Create a block referencing to another memory location
      Block(double *data, size_t size, size_t stride):
          data_(data), size_(size), stride_(stride), owner_(false) { }

      // Creates a new block from another block
      Block(const Block &block): Block(block.size_) {
        copyFrom(block);
      }

      // Assigns contents from another block.
      // If Block is not the owner of its data segment, it releases it and
      // creates it's own memory. Do not use this for content assignment.
      // See: copyFrom(const Block &block)
      Block operator=(const Block &block) {
        release();
        resize(block.size_);
        copyFrom(block);
        return *this;
      }


      //  -------- Iterators--------
      iterator begin() {
        return iterator(data_, stride_);
      }

      iterator begin() const {
        return iterator(data_, stride_);
      }

      iterator end() {
        return iterator(data_ + (size_ * stride_) , stride_);
      }

      iterator end() const {
        return iterator(data_ + (size_ * stride_) , stride_);
      }

      // -------- Accessors --------
      inline double& operator[](const size_t i) {
        return data_[i * stride_];
      }

      inline double operator[](const size_t i) const {
        return data_[i * stride_];
      }

      // --------- Setters -------
      void append(double value){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
        if(size_ >= capacity_){
          realloc(capacity_ * 1.5);
        }
        data_[size_++] = value;
      }

      void resize(size_t new_size){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
        if( new_size != size_){
          realloc(new_size);
          size_ = new_size;
        }
      }

      void realloc(size_t new_capacity){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
        if( data_ != nullptr){
          double *temp = new double[new_capacity];
          memcpy(temp, data_, sizeof(double) * std::min(size_, new_capacity));
          delete[] data_;
          data_ = temp;
        } else {
          data_ = new double[new_capacity];
        }
        capacity_ = new_capacity;
      }

      void copyFrom(const Block &block){
        ASSERT_TRUE(size_ == block.size_, "Block::copyFrom() size mismatch");
        if(stride_ == 1 && block.stride_ == 1){
          memcpy(data_, block.data_, sizeof(double) * size_);
        } else {
          Block::iterator it_this = begin();
          Block::iterator it_other = block.begin();
          while(it_this != end()){
            *it_this = *it_other;
            ++it_this, ++it_other;
          }
        }
      }

      // Releases the memory that it points to.
      // If owner, deletes content
      void release(){
        if( owner_ ){
          if( data_ != nullptr){
            delete[] data_;
          }
        }
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        stride_ = 1;
        owner_ = true;
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

      size_t stride(){
        return stride_;
      }

      bool is_owner(){
        return owner_;
      }

    private:
      double *data_;
      size_t size_;
      size_t capacity_;
      size_t stride_;
      bool owner_;
  };

}

#endif