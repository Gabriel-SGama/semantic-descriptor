#include "include/Pose_estimation.hpp"
#include "include/libUtils.hpp"


using namespace std;
using namespace cv;

Pose_estimation::Pose_estimation()
{
    R_total = (Mat_<double>(3,3) <<
                        1, 0, 0,
                        0, 1, 0,
                        0 ,0 ,1);

    map = Mat::zeros(512, 512, CV_8UC1);
    translations.push_back(Mat::zeros(3, 1, CV_64FC1));
}


/* ==================== */
/*  OpenCV's Functions  */
/* ==================== */

void Pose_estimation::pose_estimation_2d2d(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, Mat &R, Mat &t, Mat &K){
    // Intrinsics Parameters
    Point2d principal_point(K.at<double>(0, 2), K.at<double>(1,2)); // Camera Optical center coordinates
    double focal_length = K.at<double>(1,1);                        // Camera focal length

    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)  // FIXME: Pixel Coordinates? Isto está correto?
    vector<Point2f> points1, points2;  // {(x1, x2)}_n  // FIXME: {(p1, p2)}_n?

    for (int i=0; i < (int) matches.size(); i++){  // For each matched pair {(p1, p2)}_n, do...
        // Convert pixel coordinates to camera normalized coordinates
        // cout << i << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
        points1.push_back(keypoints1[matches[i].queryIdx].pt);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1  // FIXME: Pixel Coordinates? Isto está correto?
        points2.push_back(keypoints2[matches[i].trainIdx].pt);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2  // FIXME: Pixel Coordinates? Isto está correto?
    }

    cout << endl;

    //--- Calculate the Fundamental Matrix
    // Timer t1 = chrono::steady_clock::now();
    Mat F = findFundamentalMat(points1, points2, cv::FM_8POINT);  // 8-Points Algorithm
    // Timer t2 = chrono::steady_clock::now();

    //--- Calculate the Essential Matrix
    Mat E = findEssentialMat(points1, points2, focal_length, principal_point);  // Remember: E = t^*R = K^T*F*K, Essential matrix needs intrinsics info.
    // Timer t3 = chrono::steady_clock::now();

    //--- Calculate the Homography Matrix
    //--- But the scene in this example is not flat, and then Homography matrix has little meaning.
    Mat H = findHomography(points1, points2, RANSAC, 3);
    // Timer t4 = chrono::steady_clock::now();

    //--- Restore Rotation and Translation Information from the Essential Matrix, E = t^*R
    // In this program, OpenCV will use triangulation to detect whether the detected point’s depth is positive to select the correct solution.
    // This function is only available in OpenCV3!
    recoverPose(E, points1, points2, R, t, focal_length, principal_point);
    // Timer t5 = chrono::steady_clock::now();

    /* Results */
    // printElapsedTime("Pose estimation 2D-2D: ", t1, t5);
    // printElapsedTime(" | Fundamental Matrix Calculation: ", t1, t2);
    // printElapsedTime(" |   Essential Matrix Calculation: ", t2, t3);
    // printElapsedTime(" |  Homography Matrix Calculation: ", t3, t4);
    // printElapsedTime(" |             Pose Recover(R, t): ", t4, t5);
    // cout << endl;

    // printMatrix("K:\n", K);
    // printMatrix("F:\n", F);
    // printMatrix("E:\n", E);
    // printMatrix("H:\n", H);

    // printMatrix("R:\n", R);
    // printMatrix("t:\n", t);
}

Mat Pose_estimation::vee2hat(const Mat &var){
    Mat var_hat = (Mat_<double>(3,3) <<
                         0.0, -var.at<double>(2,0),  var.at<double>(1,0),
         var.at<double>(2,0),                  0.0, -var.at<double>(0,0),
        -var.at<double>(1,0),  var.at<double>(0,0),                 0.0);  // Inline Initializer

    //printMatrix("var_hat:", var_hat);

    return var_hat;
}


void Pose_estimation::updateMap2d(const Mat &R,const Mat &t){
    int _SIZE = 512;

    map.setTo(Scalar::all(0));

    R_total = R_total*R;
    Mat t_rot = R_total*t;

    translations.push_back(t_rot + translations.back());

    double x=0, y=0, z=0;

    double max_x=100, max_y=100;
    
    for (auto &curr_t: translations){
        x = curr_t.at<double>(0,0);
        y = curr_t.at<double>(0,1);

        max_x = max(max_x, x);
        max_y = max(max_y, y);

        circle(map, Point2f((int)x*_SIZE/(3*max_x) + _SIZE/2,(int)y*_SIZE/(3*max_y) + _SIZE/2),
                1, 255, FILLED);

    }

    imshow("map", map);
    // waitKey(1);
}
