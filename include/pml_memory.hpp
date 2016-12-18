#ifndef PML_MEMORY_H_
#define PML_MEMORY_H_

#include <cstring>
#include <iomanip>

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
      static const size_t INITIAL_CAPACITY = 1024;
      static const double GROWTH_RATIO;

      explicit Block(size_t size = 0) : data_(nullptr), size_(size){
        reallocate(std::max(size, INITIAL_CAPACITY));
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
          deallocate();
          reallocate(that.capacity_);
          size_ = that.size_;
          memcpy(that.data_, data_, sizeof(double) * that.size_);
        }
        return *this;
      }

      Block& operator=(Block &&that) {
        deallocate();
        data_ = that.data_;
        size_ = that.size_;
        capacity_ = that.capacity_;
        that.data_ = nullptr;
        that.size_ = 0;
        that.capacity_ = 0;
        return *this;
      }

      ~Block() {
        if( data_ )
          delete[] data_;
      }

    public:

      void clear() {
        size_ = 0;
      }

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

      void resize(size_t new_size) {
        if( new_size > capacity_ )
          reallocate(new_size * GROWTH_RATIO);
        size_ = new_size;
      }

      void reserve(size_t new_capacity) {
        if (new_capacity > capacity_)
          reallocate(new_capacity);
      }

      void shrink_to_fit() {
        if (size_ < capacity_)
          reallocate(size_);
      }

    public:
      // -------- Accessors --------
      inline double &operator[](const size_t i) {
        return data_[i];
      }

      inline double operator[](const size_t i) const {
        return data_[i];
      }

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

    public:

      void fill(double value) {
        for (size_t i = 0; i < size_; ++i)
          data_[i] = value;
      }

      void push_back(double value) {
        if (size_ == capacity_) {
          reallocate(capacity_ * GROWTH_RATIO);
        }
        data_[size_++] = value;
      }

      void push_back(const Block &block) {
        if (size_ + block.size() > capacity_) {
          reallocate( (size_ + block.size_) * GROWTH_RATIO);
        }
        double *source = &block == this ? data_ : block.data_;
        memcpy(data_ + size_, source, sizeof(double) * block.size_);
        size_ += block.size_;
      }

      void pop_back(){
        if(size_ > 0)
          --size_;
      }

    private:
      void reallocate(size_t new_capacity) {
        if (!data_) {
          data_ = new double[new_capacity];
        } else {
          double *temp = new double[new_capacity];
          if (size_ > 0)
            memcpy(temp, data_, sizeof(double) * size_);
          delete[] data_;
          data_ = temp;
        }
        capacity_ = new_capacity;
      }

      void deallocate(){
        if( data_ ){
          delete[] data_;
          data_ = nullptr;
        }
      }

    private:
      double *data_;
      size_t capacity_;
      size_t size_;
  };


  const double Block::GROWTH_RATIO  = 1.5;
}

#endif