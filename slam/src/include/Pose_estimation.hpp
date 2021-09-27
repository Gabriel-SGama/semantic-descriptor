#pragma once

#include <opencv2/opencv.hpp>

/*defines*/
#define _OFFSET_IN_VID 20
#define _FRAMES_JUMPS 3

class Pose_estimation
{
private:

    cv::Mat R_total;
    cv::Mat map;

    std::vector<cv::Mat> translations;
    std::string textMapPath;
    std::ifstream mapFilePtr;

public:
    Pose_estimation();
    void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &keypoints1,     
    const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, cv::Mat &K);
    
    void updateMap2d(const cv::Mat &R,const cv::Mat &t);
    void updateMap2dText();

    cv::Mat vee2hat(const cv::Mat &var);
};
