#ifndef LIBUTILS_EIGEN_H_
#define LIBUTILS_EIGEN_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

// #include "../libUtils_eigen.cpp"
#include <eigen3/Eigen/Core>

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/* Eigen3/Sophus */
template <typename TTypeEigenMat>
void printMatrix(const char text[], TTypeEigenMat mat);

template <typename TTypeEigenVec>
void printVector(const char text[], TTypeEigenVec vec);

template <typename TTypeEigenQuat>
void printQuaternion(const char text[], TTypeEigenQuat quat);

double RMSE(const Eigen::Vector3d &est, const Eigen::Vector3d &gt);

Eigen::Vector2d cam2pixel(const Eigen::Vector3d &P, const Eigen::Matrix3d &K);

#endif