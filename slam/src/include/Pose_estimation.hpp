#pragma once

#include <opencv2/opencv.hpp>


class Pose_estimation
{
private:

    cv::Mat R_total;
    cv::Mat map;

    std::vector<cv::Mat> translations;
    
public:
    Pose_estimation();
    void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &keypoints1,     
    const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, cv::Mat &K);
    
    void updateMap2d(const cv::Mat &R,const cv::Mat &t);

    cv::Mat vee2hat(const cv::Mat &var);
};
