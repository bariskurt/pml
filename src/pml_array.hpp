#ifndef PML_ARRAY_H_
#define PML_ARRAY_H_

#include <iostream>

// Helpers:
inline void ASSERT_TRUE(bool condition, const std::string &message) {
  if (!condition) {
    std::cout << "FATAL ERROR: " << message << std::endl;
    exit(-1);
  }
}

class Block{

  public:

    class iterator{
      public:
        iterator(double *data_, size_t stride_): data(data_), stride(stride_) {}

        double& operator*() { return *data; }

        void operator++(){ data+=stride; }

        bool operator==(const iterator &other) const {
          return data == other.data;
        }
        bool operator!=(const iterator &other) const {
          return data != other.data;
        }

      private:
        double *data;
        size_t stride;
    };

  public:
    explicit Block(size_t size_) : size(size_), stride(1), owner(true){
      data = new double[size];
    }

    explicit Block(double *data_, size_t size_, size_t stride_=1)
            : data(data_), size(size_),  stride(stride_), owner(false){}

    ~Block(){
      if (owner)
        delete[] data;
    }

    iterator begin(){
      return iterator(data, stride);
    }

    const iterator begin() const{
      return iterator(data, stride);
    }

    iterator end(){
      return iterator(data + size * stride, stride);
    }

    const iterator end() const{
      return iterator(data, stride);
    }

    void operator=(double d){
      for(auto it=begin(); it != end(); ++it){
        *it = d;
      }
    }

    void operator+=(double d){
      for(auto it=begin(); it != end(); ++it){
        *it += d;
      }
    }

    void operator-=(double d){
      for(auto it=begin(); it != end(); ++it){
        *it -= d;
      }
    }

    void operator*=(double d){
      for(auto it=begin(); it != end(); ++it){
        *it *= d;
      }
    }

    void operator/=(double d){
      for(auto it=begin(); it != end(); ++it){
        *it /= d;
      }
    }

    void operator+=(const Block &other){
      ASSERT_TRUE(size == other.size, "Block::operator+() Block size mismatch");
      for(auto it1 = begin(), it2 = other.begin(); it1 != end(); ++it1, ++it2){
        *it1 += *it2;
      }
    }

    void operator-=(const Block &other){
      ASSERT_TRUE(size == other.size, "Block::operator+() Block size mismatch");
      for(auto it1 = begin(), it2 = other.begin(); it1 != end(); ++it1, ++it2){
        *it1 -= *it2;
      }
    }

    void operator*=(const Block &other){
      ASSERT_TRUE(size == other.size, "Block::operator+() Block size mismatch");
      for(auto it1 = begin(), it2 = other.begin(); it1 != end(); ++it1, ++it2){
        *it1 *= *it2;
      }
    }

    void operator/=(const Block &other){
      ASSERT_TRUE(size == other.size, "Block::operator+() Block size mismatch");
      for(auto it1 = begin(), it2 = other.begin(); it1 != end(); ++it1, ++it2){
        *it1 /= *it2;
      }
    }

    Block subset(size_t offset, size_t sub_size, size_t sub_stride){
      return Block(data+offset, sub_size, sub_stride);
    }

  public:
    double *data;
    size_t size;
    size_t stride;
    bool owner;
};


#endif