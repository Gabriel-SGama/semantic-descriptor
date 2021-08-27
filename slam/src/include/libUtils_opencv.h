#ifndef LIBUTILS_OPENCV_H_
#define LIBUTILS_OPENCV_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

// #include "../libUtils_opencv.cpp"

/* OpenCV Libraries */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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

#endif