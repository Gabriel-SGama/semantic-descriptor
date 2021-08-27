/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
// #include <iostream>
// #include <string>
// #include <vector>
// #include <chrono>

#include "include/libUtils_basic.h"


using namespace std;


void printCharVec(const uint32_t &desc){

    cout << ((desc >> 24) & 255) << ", ";
    cout << ((desc >> 16) & 255) << ", ";
    cout << ((desc >> 8) & 255) << ", ";
    cout << (desc & 255);

}


/* Basic */
void printVec(const char text[], const std::vector<uint32_t> &vec){
    cout << text << "[";
    for(size_t i=0; i < vec.size(); i++){
        if(i != vec.size()-1){
            printCharVec(vec.at(i));
            // cout << vec.at(i) << ";";
            cout << ", ";
        }else{
            printCharVec(vec.at(i));

            // cout << vec.at(i);
        }
    }
    cout << "]" << endl;
}


template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx){
    // Starting and Ending iterators
    auto start = arr.begin() + begin_idx;
    auto end = arr.begin() + end_idx + 1;

    // To store the sliced vector
    TTypeVec result(end_idx - begin_idx + 1);

    // Copy vector using copy function()
    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}

/* Chrono */
void printElapsedTime(const char text[], Timer t1, Timer t2){
    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << text << time_elapsed.count() << " s" << endl;
}