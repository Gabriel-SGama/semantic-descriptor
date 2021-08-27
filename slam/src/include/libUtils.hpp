#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

/* Eigen */
#include <eigen3/Eigen/Core>

/* OpenCV Libraries */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>


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



/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/* OpenCV */
int checkImage(const cv::Mat &image);

std::string type2str(int type);

void printImageInfo(const cv::Mat &image);

void printMatrix(const char text[], cv::Mat var);

void printMatrix(const char text[], cv::MatExpr var);

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);

void calcHist(const cv::Mat src, std::string img_name);
