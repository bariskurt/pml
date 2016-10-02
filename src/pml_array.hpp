#ifndef PML_ARRAY_H_
#define PML_ARRAY_H_

#include <iostream>
#include <cstring>

namespace pml {
  // Helpers:
  inline void ASSERT_TRUE(bool condition, const std::string &message) {
    if (!condition) {
      std::cout << "FATAL ERROR: " << message << std::endl;
      exit(-1);
    }
  }

  class Block {

    public:

      class iterator {
        public:
          iterator(double *data, size_t stride) : data_(data),
                                                    stride_(stride) { }

          double &operator*() { return *data_; }

          void operator++() { data_ += stride_; }

          bool operator==(const iterator &other) const {
            return data_ == other.data_;
          }

          bool operator!=(const iterator &other) const {
            return data_ != other.data_;
          }

      private:
          double *data_;
          size_t stride_;
      };

    public:
      Block() : data_(nullptr), size_(0), stride_(1), owner_(true){
        reserve(16);
      }

      explicit Block(size_t size, double value = 0)
              : size_(size), capacity_(size), stride_(1), owner_(true) {
        data_ = new double[capacity_];
        for(size_t i=0; i < size; ++i){
          data_[i] = value;
        }
      }

      explicit Block(double *data, size_t size, size_t stride = 1)
              : data_(data), size_(size), capacity_(size),
                stride_(stride), owner_(false) { }

      ~Block() {
        if (owner_)
          delete[] data_;
      }

    public:
      //  -------- Iterators--------
      iterator begin() {
        return iterator(data_, stride_);
      }

      const iterator begin() const {
        return iterator(data_, stride_);
      }

      iterator end() {
        return iterator(data_ + size_ * stride_, stride_);
      }

      const iterator end() const {
        return iterator(data_ + size_ * stride_, stride_);
      }

    public:
      //  -------- Block - double operations --------
      void operator=(double d) {
        for (auto it = begin(); it != end(); ++it) { *it = d; }
      }

      void operator+=(double d) {
        for (auto it = begin(); it != end(); ++it) { *it += d; }
      }

      void operator-=(double d) {
        for (auto it = begin(); it != end(); ++it) { *it -= d; }
      }

      void operator*=(double d) {
        for (auto it = begin(); it != end(); ++it) { *it *= d; }
      }

      void operator/=(double d) {
        for (auto it = begin(); it != end(); ++it) { *it /= d; }
      }

      //  -------- Block - Block operations --------
      void operator+=(const Block &other) {
        ASSERT_TRUE(size_ == other.size_,
                    "Block::operator+() Block size mismatch");
        for (auto it1 = begin(), it2 = other.begin();
             it1 != end(); ++it1, ++it2) {
          *it1 += *it2;
        }
      }

      void operator-=(const Block &other) {
        ASSERT_TRUE(size_ == other.size_,
                    "Block::operator+() Block size mismatch");
        for (auto it1 = begin(), it2 = other.begin();
             it1 != end(); ++it1, ++it2) {
          *it1 -= *it2;
        }
      }

      void operator*=(const Block &other) {
        ASSERT_TRUE(size_ == other.size_,
                    "Block::operator+() Block size mismatch");
        for (auto it1 = begin(), it2 = other.begin();
             it1 != end(); ++it1, ++it2) {
          *it1 *= *it2;
        }
      }

      void operator/=(const Block &other) {
        ASSERT_TRUE(size_ == other.size_,
                    "Block::operator+() Block size mismatch");
        for (auto it1 = begin(), it2 = other.begin();
             it1 != end(); ++it1, ++it2) {
          *it1 /= *it2;
        }
      }

      Block subset(size_t offset, size_t sub_size, size_t sub_stride) {
        return Block(data_ + offset, sub_size, sub_stride);
      }

    public:
      //  -------- Push and Pop back operations --------
      void pop_back(){
        ASSERT_TRUE(owner_,
            "Block::push_back: Block can only be resized by its owner.");
        if(size_ > 0)
          --size_;
      }

      void push_back(double d){
        if(size_ == capacity_)
          reserve(capacity_ * 1.5);
        data_[size_++] = d;
      }

      void resize(size_t new_size){
        ASSERT_TRUE(owner_,
            "Block::resize: Block can only be resized by its owner.");
        if(new_size > capacity_)
          reserve(new_size);
        size_ = new_size;
      }

    private:
      void reserve(size_t new_capacity){
        ASSERT_TRUE(owner_,
            "Block::reserve: Block can only be reserved by its owner.");
        double *temp = new double[new_capacity];
        memcpy(temp, data_, sizeof(double)*size_);
        delete[] data_;
        data_ = temp;
        capacity_ = new_capacity;
      }

    public:
      // ----- get and set -------
      double* data(){
        return data_;
      }

      const double* data() const{
        return data_;
      }

      size_t size() const{
        return size_;
      }

      size_t capacity() const{
        return capacity_;
      }

      size_t stride() const{
        return stride_;
      }

      double& operator[](const size_t i) {
        return data_[i*stride_];
      }

      double operator[](const size_t i) const{
        return data_[i*stride_];
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