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
    friend class Vector;

    public:
      template <bool is_const>
      class block_iterator {

        // Const and non-const iterators are friends.
        template <bool is_other_const>
        friend class block_iterator;

        // Return type is "const double&" if the iterator is const.
        typedef typename std::conditional<is_const,
            const double&, double&>::type ReturnType;

        public:
          block_iterator(double *data_, size_t stride_)
              : data(data_), stride(stride_) {}

          // Can convert non-const iterator to const. But not the other way.
          block_iterator(const block_iterator<false>& it)
              : data(it.data), stride(it.stride) {}

          // Can assign non-const iterator to const. But not the other way.
          block_iterator& operator=(const block_iterator<false>&it){
            data = it.data;
            stride = it.stride;
            return *this;
          }

          template <bool is_other_const>
          bool operator==(const block_iterator<is_other_const>& it){
            return data == it.data;
          }

          template <bool is_other_const>
          bool operator!=(const block_iterator<is_other_const>& it){
            return data != it.data;
          }

          block_iterator<is_const>& operator++(){
            data+=stride;
            return *this;
          }

          block_iterator<is_const> operator++(int){
            iterator temp(*this);
            data+=stride;
            return temp;
          }

          block_iterator<is_const>& operator--(){
            data-=stride;
            return *this;
          }

          block_iterator<is_const> operator--(int){
            iterator temp(*this);
            data-=stride;
            return temp;
          }

          void operator+=(size_t step){
            data += step * stride;
          }

          void operator-=(size_t step){
            data -= step * stride;
          }

          block_iterator<is_const> operator+(size_t step){
            block_iterator<is_const> result(*this);
            result += step;
            return result;
          }

          block_iterator<is_const> operator-(size_t step){
            block_iterator<is_const> result(*this);
            result -= step;
            return result;
          }

          ReturnType operator*(){
            return *data;
          }

        private:
          double *data;
          size_t stride;
      };

      typedef block_iterator<true> const_iterator;
      typedef block_iterator<false> iterator;

    public:
      // Allocate a new block, as the owner.
      explicit Block(size_t size = 0)
          :data_(nullptr), size_(size), capacity_(0), stride_(1), owner_(true) {
        if( size_ > 0)
          realloc(size_);
      }

      explicit Block( const std::initializer_list<double> &values )
          : Block(values.size()){
        Block::iterator it = begin();
        for(double value : values){
          *it = value;
          ++it;
        }
      }

      // Create a block referencing to another memory location
      Block(double *data, size_t size, size_t stride, bool owner = false){
        if(!owner){
          data_ = data, size_ = size, stride_ = stride, owner_ = false;
        } else {
          owner_ = true;
          resize(size);
          copyFrom(data, size, stride);
        }
      }


      // Creates a new block from another block
      Block(const Block &block): Block(block.size_) {
        copyFrom(block.data(), block.size(), block.stride());
      }

      // Assigns contents from another block.
      // If Block is not the owner of its data segment, it releases it and
      // creates it's own memory. Do not use this for content assignment.
      // See: copyFrom(const Block &block)
      Block operator=(const Block &block) {
        release();
        resize(block.size_);
        copyFrom(block.data(), block.size(), block.stride());
        return *this;
      }

      // Only the owner of the data block can delete data segment.
      // ToDo: use unique pointer
      ~Block(){
        if(owner_){
          delete[] data_;
        }
      }


      //  -------- Iterators--------
      iterator begin() {
        return iterator(data_, stride_);
      }

      const_iterator begin() const {
        return const_iterator(data_, stride_);
      }

      const_iterator cbegin() const {
        return const_iterator(data_, stride_);
      }

      iterator end() {
        return iterator(data_ + (size_ * stride_) , stride_);
      }

      const_iterator end() const {
        return const_iterator(data_ + (size_ * stride_) , stride_);
      }

      const_iterator cend() const {
        return const_iterator(data_ + (size_ * stride_) , stride_);
      }

      // -------- Accessors --------
      double& operator[](const size_t i){
        return data_[i * stride_];
      }

      double operator[](const size_t i) const{
        return data_[i * stride_];
      }

      double& front() {
        return data_[0];
      }

      double front() const{
        return data_[0];
      }

      double& back() {
        return data_[(size_-1) * stride_];
      }

      double back() const {
        return data_[(size_-1) * stride_];
      }

      // --------- Setters -------
      void append(double value){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
        if(size_ >= capacity_){
          realloc(std::max(4.0, capacity_ * 1.5));
        }
        data_[size_++] = value;
      }

      void append(const Block &block){
        ASSERT_TRUE(owner_, "Block is the not owner of its data segment");
        if(size_ + block.size() > capacity_) {
          realloc(size_ + block.size());
        }
        memcpy(&data_[size_], block.data(), sizeof(double) * block.size());
        size_ += block.size();
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

      void copyFrom(const double *source, size_t size, size_t stride){
        ASSERT_TRUE(size_ == size, "Block::copyFrom() size mismatch");
        if(stride_ == 1 && stride == 1){
          memcpy(data_, source, sizeof(double) * size_);
        } else {
          size_t i = 0, j = 0;
          for(size_t iter = 0; iter < size_; ++iter){
            data_[i] = source[j];
            i += stride_;
            j += stride;
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

      size_t size() const {
        return size_;
      }

      size_t stride() const {
        return stride_;
      }

      bool is_owner() const {
        return owner_;
      }

      bool empty() const {
        return size_ == 0;
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