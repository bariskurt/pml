//
// Created by cagatay on 30.03.2016.
//


#include "pml_old.hpp"

using namespace std;
using namespace pml;

Tensor3D LinearTensor(size_t dim0, size_t dim1, size_t dim2) {
    size_t size = dim0*dim1*dim2;
    double* list = new double[size];
    for(size_t i=0; i<size; i++) {
        list[i] = i;
    }
    Tensor3D tensor(dim0,dim1,dim2,list);
    delete[] list;
    return tensor;
}

void TestBlock() {
    cout << "TestBlock" << endl;
    Tensor3D tensor = LinearTensor(2,3,4);

    for (size_t k = 0; k < tensor.dim2(); ++k) {
        for (size_t j = 0; j < tensor.dim1() ; ++j) {
            for (size_t i = 0; i < tensor.dim0(); ++i) {
                cout << tensor(i, j, k) << endl;
            }
        }
    }
}

void TestPrint() {
    Tensor3D tensor = LinearTensor(4,3,2);
    cout << "TestPrint" << endl << tensor << endl;

}

void TestDirection() {
    cout << "TestDirection" << endl;
    Tensor3D tensor = LinearTensor(5,6,3);
    Matrix m = Matrix::Ones(5,6);
    cout << tensor << endl;
    tensor.SetSlice(0,m);
    for (size_t i=0; i<3; i++) {
        cout << tensor.GetSlice(i) << endl;
    }
}

void TestAddition() {
    Tensor3D tensor = LinearTensor(2,3,4);
    Tensor3D ones = Tensor3D::Ones(2,3,4);
    cout << "tensor:"<< endl << tensor << endl << "tensor++" << endl << tensor + ones;
}

void TestMultiplication() {
    Tensor3D tensor = LinearTensor(2,3,4);
    cout << "tensor:"<< endl << tensor  << "tensor*2:\n" << tensor*2 << "2*tensor:\n" << 2*tensor << endl;

}

int main() {

    TestBlock();
    cout << "-------------\n" << endl;
    TestPrint();
    cout << "-------------\n" << endl;
    TestDirection();
    cout << "-------------\n" << endl;
    TestAddition();
    cout << "-------------\n" << endl;
    TestMultiplication();

    return 0;
}