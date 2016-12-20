#ifndef PML_MEMORY_H_
#define PML_MEMORY_H_

#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace pml {

  class Memory {

    public:
      static const size_t KB = 1024;
      static const size_t MB = KB * KB;
      static const size_t GB = KB * MB;
      static const size_t DEFAULT_SIZE = 128 * MB;
      static const size_t MIN_BLOCK_SIZE = 16;

    public:

      static void init(size_t initial_size) {
        getInstance(initial_size);
      }

      static void *malloc(size_t size) {
        return getInstance().allocate(size);
      }

      static void free(void *ptr) {
        getInstance().deallocate(ptr);
      }

      static void dump(const std::string &filename = "") {
        getInstance().dumpBlock(filename);
      }

    private:

      struct FreeBlock {
          FreeBlock(size_t size, FreeBlock *next) : size_(size),
                                                    next_(next) {}

          size_t size_;
          FreeBlock *next_;
      };

      static Memory &getInstance(size_t initial_size = DEFAULT_SIZE) {
        static Memory memory(initial_size);
        return memory;
      }

      Memory(size_t mem_size) {
        size_ = mem_size;
        block_ = new char[size_];
        memset(block_, 0, size_);
        free_blocks_head = reinterpret_cast<FreeBlock *>(block_);
        free_blocks_head->size_ = size_;
        free_blocks_head->next_ = nullptr;
      }

      ~Memory() {
        delete[] block_;
      }

      void *allocate(size_t req_size) {
        // Block must be a multiple of 4bytes
        size_t reminder = req_size % 4;
        if (reminder)
          req_size += 4 - reminder;
        FreeBlock *prev_free_block = nullptr;
        FreeBlock *free_block = free_blocks_head;
        size_t total_size = req_size + sizeof(size_t);
        while (free_block) {
          if (free_block->size_ < total_size) {
            prev_free_block = free_block;
            free_block = free_block->next_;
          } else {
            // is there meaningful space after the allocation ?
            if (free_block->size_ - total_size < MIN_BLOCK_SIZE) {
              // do not divide free block, allocate all of it
              total_size = free_block->size_;
              if (prev_free_block)
                prev_free_block->next_ = free_block->next_;
              else
                free_blocks_head = free_block->next_;
            } else {
              // divide current free block
              char *block_start = reinterpret_cast<char *>(free_block);
              char *block_end = block_start + total_size;
              FreeBlock *next_block = reinterpret_cast<FreeBlock *>(block_end);
              next_block->size_ = free_block->size_ - total_size;
              next_block->next_ = free_block->next_;
              if (prev_free_block)
                prev_free_block->next_ = next_block;
              else
                free_blocks_head = next_block;
            }
            // store size at the first 8 bytes
            size_t *data_header = reinterpret_cast<size_t *>(free_block);
            char *data_start = reinterpret_cast<char *>(data_header + 1);
            *data_header = total_size;

            // clear memory for debug purposes
            memset(data_start, 0xff, req_size);

            // update global parameters
            used_memory += total_size;
            ++num_allocations;

            return data_start;
          }
        }
        return std::malloc(req_size);
      }

      void deallocate(void *ptr) {
        if (ptr < block_ || ptr >= (block_ + size_)) {
          std::free(ptr);
          return;
        }
        char *block_start = reinterpret_cast<char *>(ptr) - sizeof(size_t);
        size_t block_size = *((size_t *) block_start);
        char *block_end = block_start + block_size;

        FreeBlock *prev_free_block = nullptr;
        FreeBlock *free_block = free_blocks_head;

        while (free_block && (char *) free_block < block_end) {
          prev_free_block = free_block;
          free_block = free_block->next_;
        }

        if (!prev_free_block) {
          // Free first block
          prev_free_block = (FreeBlock *) block_start;
          prev_free_block->size_ = block_size;
          prev_free_block->next_ = free_blocks_head;
          free_blocks_head = prev_free_block;
        } else {
          if ((char *) prev_free_block + prev_free_block->size_ ==
              block_start) {
            // There's a free block at the back, extend it.
            prev_free_block->size_ += block_size;
          } else {
            // Create a new free block just after prev_free_block
            FreeBlock *temp = (FreeBlock *) block_start;
            temp->next_ = prev_free_block->next_;
            prev_free_block->next_ = temp;
            prev_free_block = temp;
          }
        }

        if (free_block && (char *) free_block == block_end) {
          prev_free_block->size_ += free_block->size_;
          prev_free_block->next_ = free_block->next_;
        }

        --num_allocations;
        used_memory -= block_size;
      }

      void dumpBlock(const std::string &filename) {
        const size_t width = 16;
        for (size_t i = 0; i < std::ceil(size_ / width); i++) {
          size_t start = i * width, stop = std::min((i + 1) * width, size_);
          printf("%p : ", &block_[start]);
          for (size_t j = start; j < stop; ++j) {
            if (j > start)
              printf(":");
            printf("%02x", (unsigned) (unsigned char) block_[j]);
          }
          printf("\n");
        }
      }

    private:
      FreeBlock *free_blocks_head;
      char *block_;
      size_t size_;

      size_t num_allocations;
      size_t used_memory;
  };

  template <typename T>
  class Allocator {

    public:

      typedef size_t size_type;
      typedef ptrdiff_t difference_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef T value_type;

    public:

      Allocator(){}

      ~Allocator(){}

      template <class U> struct rebind { typedef Allocator<U> other; };

      template <class U> Allocator(const Allocator<U>&){}

      pointer address(reference x) const {return &x;}

      const_pointer address(const_reference x) const {return &x;}

      size_type max_size() const throw() {
        return size_t(-1) / sizeof(value_type);
      }

      bool operator==(const Allocator &rhs) {
        return false;
      }

      bool operator!=(const Allocator &rhs) {
        return !operator==(rhs);
      }

    public:

      pointer allocate(size_type n, Allocator<T>::const_pointer hint = 0) {
        return static_cast<pointer>(pml::Memory::malloc(n*sizeof(T)));
      }

      void deallocate(pointer p, size_type n) {
        pml::Memory::free(p);
      }

      void construct(pointer p, const T& val) {
        new(static_cast<void*>(p)) T(val);
      }

      void construct(pointer p) {
        new(static_cast<void*>(p)) T();
      }

      void destroy(pointer p){
        p->~T();
      }
  };

}

#endif