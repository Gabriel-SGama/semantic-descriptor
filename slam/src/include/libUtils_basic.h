#ifndef LIBUTILS_BASIC_H_
#define LIBUTILS_BASIC_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

// #include "../libUtils_basic.cpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

typedef std::chrono::steady_clock::time_point Timer;

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/* Basic */
void printVec(const char text[], const std::vector<double> &vec);

template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx);

/* Chrono */
void printElapsedTime(const char text[], Timer t1, Timer t2);

#endif