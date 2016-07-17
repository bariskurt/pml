#include <iostream>
#include <vector>
#include <algorithm>

#include <cassert>

#include "pml_hmm.hpp"

using namespace std;
using namespace hmm;

string data_file_path = "../etc/hamlet.txt";
string hmm_dump_file_path = "../etc/hmm.dump";


string vector2str(const std::vector<double> &v) {
    string s = " ";
    for (size_t t=0; t<v.size()-1; ++t) {
        s += to_string(v[t]);
        s += ",";
    }
    s += to_string(v[v.size()-1]);
    return s;
}

// returns true if the elements in the vector are in non-decreasing order
bool checkLLDecrease(vector<double> ll) {
    for (size_t ind=1; ind<ll.size(); ind++) {
        if ( ll[ind] < ll[ind-1] ) {
            cout << "LL decreased @ time" << ind << ", " << ll[ind] - ll[ind-1] << endl;
            return true;
        }
    }
    return false;
}

// reads a file made up of an integer per line
vector<unsigned> readFile(const string &filename) {
    vector<unsigned> data;
    ifstream ifs(filename);

    for(unsigned n; ifs >> n; ) {
        data.push_back(n);
    }
    return data;
}

// inversion sampling
unsigned randgen(const Vector& v) {
    unsigned i = 0;
    double rnd = uniform::rand();
    double cumsum = 0;
    for (; i < v.size(); i++) {
        cumsum += v(i);
        if (cumsum > rnd) {
            break;
        }
    }
    return i;
}

// loads a file of nonnegative integers, all of them in a different line
vector<unsigned> loadData(const string &f_path, size_t T = 1000) {
    vector<unsigned> book = readFile(f_path);
    vector<unsigned> obs;
    copy(book.begin(),book.end(),back_inserter(obs));
    obs.resize(T);
    return obs;
}

/*
 * executes the generative model and generates data
 * returns hidden states and observations
 */
pair<vector<unsigned>,vector<unsigned>> genData() {
    size_t N = 3;       // num. of hidden states
    size_t M = 5;       // num. of distinct observation
    size_t T = 10000;   // training set size

    Vector pi = normalize(Vector::ones(N));
    Matrix A = normalize(5*Matrix::identity(N)+uniform::rand(N,N),Matrix::COLS);
    Matrix B = normalize(Matrix::identity(M)+0.1,Matrix::COLS);

    vector<unsigned> states(T);
    vector<unsigned> obs(T);

    for (size_t t=0; t<T; t++) {
        if (t== 0)
            states[t] = randgen(pi);
        else
            states[t] = randgen(A.getColumn(states[t-1]));
        obs[t] = randgen(B.getColumn(states[t]));
    }
    return make_pair(states, obs);
}

/*
 * starting the EM algorihm at the same point,
 * checks whether recursive smoother and correction smoother learn the same set of parameters
 */
void check2SSMethods() {
    size_t training_size=1000;
    unsigned epoch=20;
    vector<unsigned> train_set = loadData(data_file_path, training_size);

    HMM hmm1(hmm_dump_file_path);
    vector<double> ll1 = hmm1.learnParameters(train_set, epoch, HMM::SS_METHOD::RECURSIVE_SMOOTHER);
    assert(!checkLLDecrease(ll1));

    HMM hmm2(hmm_dump_file_path);
    vector<double> ll2 = hmm2.learnParameters(train_set, epoch, HMM::SS_METHOD::CORRECTION_SMOOTHER);
    assert(!checkLLDecrease(ll2));

    assert(hmm1.getpi() == hmm2.getpi());
    assert(hmm1.getA() == hmm2.getA());
    assert(hmm1.getB() == hmm2.getB());

}


void offlineLearningExample() {

    check2SSMethods();
    cout << "We first ensure that recursive smoother and correction smoother converge to the same point" <<
            " when EM starts at the same point." << endl;

    size_t training_size=10000;
    size_t test_size=150000;
    unsigned epoch=100;

    vector<unsigned> train_set = loadData(data_file_path, training_size);
    vector<unsigned> test_set = loadData(data_file_path, test_size);
    HMM hmm(hmm_dump_file_path);

    time_t beg = time(nullptr);
    vector<double> ll = hmm.learnParameters(train_set, epoch, HMM::SS_METHOD::RECURSIVE_SMOOTHER);

    time_t end = time(nullptr);
    cout << "A new HMM is trained with " << epoch << " epochs, within " << end-beg
            << " seconds. final ll is: " << ll[ll.size()-1] << endl;

    assert(!checkLLDecrease(ll));

    cout << "log-likelihood on the test set: "<< hmm.evaluateLogLHood(test_set) << endl << endl;

}

void onlineLearningExample() {
    unsigned T = 100000;
    unsigned n_min=100;
    long log_interval=5000;
    double learning_rate=-0.6;
    int learning_rate_update_period=1;
    double learn_rate_constant=1;

    vector<unsigned> obs = loadData(data_file_path, T);
    HMM hmm(hmm_dump_file_path);

    time_t beg = time(nullptr);
    vector<tuple<Vector,Matrix,Matrix>> resulting_params = hmm.runOnlineEM(obs, n_min, log_interval, learning_rate,
                                                                           learning_rate_update_period,
                                                                           learn_rate_constant);
    time_t end = time(nullptr);
    cout << "Online HMM training with " << T << " samples, within " << end-beg << " seconds." << endl;
    
}


int main(){

    // just for illustration
    pair<vector<unsigned>,vector<unsigned>> data = genData();

    offlineLearningExample();

    onlineLearningExample();

    return 0;
}