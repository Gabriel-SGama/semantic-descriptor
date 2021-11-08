#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include "consts.h"

class Pose_estimation
{
private:

    cv::Mat R_total;
    cv::Mat map;


    std::vector<int> nFailedRansac;
    std::vector<int> nL;
    std::vector<int> ntotalFeatures;
    std::vector<int> nFeaturesUsed;
    std::vector<cv::Mat> translations;
    std::string textMapPath;
    std::string dataEvoPath;
    std::ifstream mapFilePtr;
    std::ofstream dataEvoPtr;

public:
    Pose_estimation();
    void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &keypoints1,     
    const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, cv::Mat &K);
    
    void updateMap2d(const cv::Mat &R,const cv::Mat &t);
    void updateMap2dText();
    void closeFiles();
    cv::Mat vee2hat(const cv::Mat &var);

    std::vector<cv::DMatch> ransac(const std::vector<cv::KeyPoint> &keypoints1,     
    const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, cv::Mat &K);

    void plotInfo();

};
