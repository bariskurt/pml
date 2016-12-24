#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "pml_block.hpp"

using namespace pml;
using namespace std;

std::string test_dir = "/tmp/";

inline bool fequal(double a, double b) {
  return std::fabs(a - b) < 1e-6;
}

namespace pml {

  class PmlTester {

    public:

      void assert_empty(const Block &b){
        assert(b.data_ == nullptr);
        assert(b.size_ == 0);
        assert(b.capacity_ == 0);
      }

      void test_constructors() {

        std::cout << "test_constructors...\n";

        // Block of length 10
        const size_t SIZE = 10;
        Block b(SIZE);
        // Populate with numbers [0, 9]
        for (size_t i = 0; i < SIZE; ++i)
          b.data_[i] = i;
        assert(b.size() == SIZE);
        assert(b.capacity() == SIZE);

        // Copy constructor (deep copy)
        Block b2(b);
        assert(b2.capacity() == SIZE);
        assert(b2.size() == SIZE);
        assert(b2.data() != b.data());

        // Assignment operator
        Block b3;
        b3 = b;
        assert(b3.capacity() == b.size());
        assert(b3.size() == b.size());
        assert(b3.data() != b.data());

        // Copy Initialization
        Block b4 = b;
        assert(b4.capacity() == b.size());
        assert(b4.size() == b.size());
        assert(b4.data() != b.data());

        // Empty Block operations
        {
          Block e;
          assert_empty(e);

          Block e2(e);
          assert_empty(e2);

          Block e3 = e;
          assert_empty(e3);

          Block e4; e4 = e;
          assert_empty(e4);

        }

        std::cout << "OK.\n";
      }

      void test_rvalue_references() {

        // Move constructor
        const size_t SIZE = 10;
        Block b(SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          b.data_[i] = i;
        const double *b_data_ = b.data_;

        Block b2(std::move(b));
        assert(b.data_ == nullptr);
        assert(b.size_ == 0);
        assert(b.capacity_ == 0);

        assert(b2.data_ == b_data_);
        assert(b2.size_ == SIZE);
        assert(b2.capacity_ == SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          assert(b2.data_[i] == i);

        // Move Assignment
        Block b3;
        b3 = std::move(b2);
        assert(b2.data_ == nullptr);
        assert(b2.size_ == 0);
        assert(b2.capacity_ == 0);

        assert(b3.data_ == b_data_);
        assert(b3.size_ == SIZE);
        assert(b3.capacity_ == SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          assert(b3.data_[i] == i);


      }

      void test_size() {

        std::cout << "test_size...\n";

        const size_t SIZE = 10;

        Block b(SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          b[i] = i;

        assert(b.size() == SIZE);
        assert(b.capacity() == SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          assert(fequal(b[i], i));

        // try to shrink but fail
        b.__reserve__(SIZE - 2);
        assert(b.size() == SIZE);
        assert(b.capacity() == SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          assert(fequal(b[i], i));

        // try to shrink with success
        b.__shrink_to_fit__();
        assert(b.size() == SIZE);
        assert(b.capacity() == SIZE);
        for (size_t i = 0; i < SIZE; ++i)
          assert(fequal(b[i], i));

        // try to reserve with success
        const size_t NEW_CAPACITY = 2048;
        b.__reserve__(NEW_CAPACITY);
        assert(b.size() == SIZE);
        assert(b.capacity() == NEW_CAPACITY);
        for (size_t i = 0; i < SIZE; ++i)
          assert(fequal(b[i], i));

        std::cout << "OK.\n";
      }

      void test_push_back(const size_t X_SIZE, const size_t Y_SIZE){
        Block x(X_SIZE);
        for(size_t i=0; i < x.size(); ++i)
          x[i] = i;

        Block y(Y_SIZE);
        for(size_t i=0; i < y.size(); ++i)
          y[i] = i;

        x.__push_back__(y);

        assert(x.size() == X_SIZE + Y_SIZE);
        assert(y.size() == Y_SIZE);
        if( X_SIZE + Y_SIZE > X_SIZE )
          assert(x.capacity() ==
                     (size_t)((X_SIZE + Y_SIZE) * Block::GROWTH_RATIO));

        for(size_t i=0; i < X_SIZE; ++i)
          assert(fequal(x[i], i));

        for(size_t i=0; i < Y_SIZE; ++i) {
          assert(fequal(x[X_SIZE + i], i));
          assert(fequal(y[i], i));
        }
      }

      void test_push_back_self(const size_t X_SIZE){
        Block x(X_SIZE);
        for(size_t i=0; i<X_SIZE; ++i)
          x[i] = i;

        x.__push_back__(x);

        assert(x.size() == 2 * X_SIZE);
        assert(x.capacity() == (size_t) (2 * X_SIZE * Block::GROWTH_RATIO));
        for(size_t i=0; i<X_SIZE; ++i){
          assert(fequal(x[i], i));
          assert(fequal(x[X_SIZE+i], i));
        }
      }

      void test_push_back() {

        std::cout << "test_push_back...\n";

        // PART 1: push_back double values
        {
          Block b;
          size_t SIZE = 64;
          for (size_t i = 0; i < SIZE; ++i)
            b.__push_back__(i);

          // test
          assert(b.size() == SIZE);
          for (size_t i = 0; i < SIZE; ++i)
            assert(fequal(b[i], i));
        }

        // PART 2: push_back two small vectors
        test_push_back(10, 5);

        // PART 2: push_back two large vectors
        test_push_back(1000, 500);

        // PART 3: push_back self (small)
        test_push_back_self(10);


        // PART 4: push_back self (larger)
        test_push_back_self(1000);


        //PART 5: Push back to empty vectors
        test_push_back(0, 1000);      // full to empty
        test_push_back(1000, 0);      // empty to full
        test_push_back(0, 0);         // empty to empty
        test_push_back_self(0);       // empty to itself

        std::cout << "OK.\n";
      }
  };
}

int main(){
  PmlTester blockTester;
  blockTester.test_constructors();
  blockTester.test_rvalue_references();
  blockTester.test_size();
  blockTester.test_push_back();
}