#include "pml_array.hpp"


void set_even(Block &block){
  Block block1 = block.subset(0, 5, 2);
  block1 = 3;
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

  return 0;
}