#include <cassert>

#include "pml_vector_view.hpp"
#include "pml_time.hpp"

using namespace pml;


void test_vector_view() {
  Vector v = {0, 1, 2, 3, 4, 5, 6, 7};
  VectorView vw(v);

  Vector v2 = {0, 1, 2, 3, 4, 5, 6, 7};
  VectorView vw2(v2);

  for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it)
    std::cout << *it << std::endl;

  vw += v2;

  ConstVectorView cvw(v);
  for(ConstVectorView::iterator it = cvw.begin(); it != cvw.end(); ++it)
    std::cout << *it << std::endl;

}

int main(){
  test_vector_view();
}