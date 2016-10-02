#include "pml_array.hpp"

using namespace pml;

void set_even(Block &block){
  Block block1 = block.subset(0, 5, 2);
  block1 = 3;
}

void test_resize(){
  Block block(2,1);
  for(int i=0; i < 5; ++i){
    block.push_back(7);
  }
  std::cout << "size: " << block.size() << std::endl;
  std::cout << "capacity: " << block.capacity() << std::endl;
  for(Block::iterator it = block.begin(); it != block.end(); ++it){
    std::cout << *it << " ";
  }
  std::cout << std::endl;

  block.resize(4);
  for(Block::iterator it = block.begin(); it != block.end(); ++it){
    std::cout << *it << " ";
  }
  std::cout << std::endl;
  std::cout << "size: " << block.size() << std::endl;
  std::cout << "capacity: " << block.capacity() << std::endl;

  block.resize(10);
  for(Block::iterator it = block.begin(); it != block.end(); ++it){
    std::cout << *it << " ";
  }
  std::cout << std::endl;
  std::cout << "size: " << block.size() << std::endl;
  std::cout << "capacity: " << block.capacity() << std::endl;

}

int main(){
  Block block(10);
  block = 5;

  for(Block::iterator it = block.begin(); it != block.end(); ++it){
    std::cout << *it << " ";
  }
  std::cout << std::endl;

  set_even(block);

  for(Block::iterator it = block.begin(); it != block.end(); ++it){
    std::cout << *it << " ";
  }
  std::cout << std::endl;

  test_resize();

  return 0;
}