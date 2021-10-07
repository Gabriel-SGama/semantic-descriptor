/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <nmmintrin.h>
#include <bitset>
#include <fstream>

/* Custom Libraries */
#include "include/libUtils.hpp"
#include "include/ORBFeatures.hpp"
#include "include/Pose_estimation.hpp"

using std::ofstream;
using namespace std;
using namespace cv;

const int nfeatures = 500;
// const float factorPI = CV_PI/180.0;

const double matches_lower_bound = 45.0;

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    std::ifstream mapGTin;
    std::ofstream mapGTout;
    mapGTin.open("../carla-pythonAPI/pos_kitti.txt");
    mapGTout.open("../carla-pythonAPI/pos_kitti_align.txt");

    float x, y, z, r11, r12, r13, r21, r22, r23, r31 ,r32 ,r33;
    
    // int nlines = count(istreambuf_iterator<char>(mapGTin), 
    //          istreambuf_iterator<char>(), '\n');

    // mapGTin.clear();
    // mapGTin.seekg(0);
    
    cout << r11 << " | " << z << endl;

    cout << "[orb_cv] Hello!" << endl << endl;
    
    ofstream outdata;
    outdata.open("data.txt");

    ORBFeatures *orbFeatures = new ORBFeatures();
    Pose_estimation *pose_estimation = new Pose_estimation();

    Mat image1;
    Mat image2;
    Mat image_gray1;
    Mat image_gray2;

    Mat semantic1;
    Mat semantic2;
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2, sem_descriptor1, sem_descriptor2;

    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    VideoCapture cap("/home/gama/code/semantic-descriptor/carla-pythonAPI/out_camera_kitti.avi"); 
    VideoCapture cap_seg("/home/gama/code/semantic-descriptor/carla-pythonAPI/out_seg_kitti.avi"); 
    
    Mat K = (Mat_<double>(3, 3) << 256.0, 0, 256.0, 0, 256.0, 256.0, 0, 0, 1.0);
    Mat R, t, t_hat;

    if(!cap.isOpened() || !cap_seg.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int frameid = 0;
    bool read_flag = true;

    // while(frameid < _OFFSET_IN_VID){
    //     cap.read(image1);
    //     cap_seg.read(semantic1);
    //     frameid++;
    // }
    
    // read_flag = cap.read(image2);
    // cap_seg.read(semantic2);
    

    while(cap.isOpened()){
        frameid++;

        //ignores begining
        if(frameid < _OFFSET_IN_VID){
            cap.read(image1);
            cap_seg.read(semantic1);
            mapGTin >> r11 >> r12 >> r13 >> x >> r21 >> r22 >> r23 >> y >> r31 >> r32 >> r33 >> z;
            mapGTin.get();
            continue;
        }
            
        for(int k = 0; k < _FRAMES_JUMPS; k++){
            read_flag = cap.read(image2);
            cap_seg.read(semantic2);
            mapGTin >> r11 >> r12 >> r13 >> x >> r21 >> r22 >> r23 >> y >> r31 >> r32 >> r33 >> z;
            mapGTin.get();
        }


      
        if(!read_flag)
            break;

        mapGTout << fixed << setprecision(4) << r11 << " " << r12 << " " << r13 << " " << x << " "
        << r21 << " " << r22 << " " << r23 << " " << z << " "
        << r31 << " " << r32 << " " << r33 << " " << y << "\n";


        cvtColor(image1, image_gray1, COLOR_BGR2GRAY);
        cvtColor(image2, image_gray2, COLOR_BGR2GRAY);
        
        detector->detect(image_gray1, keypoints1);
        detector->detect(image_gray2, keypoints2);

        // Timer t1 = chrono::steady_clock::now();
        // descriptor->compute(image_gray1, keypoints1, sem_descriptor1);
        // descriptor->compute(image_gray2, keypoints2, sem_descriptor2);
        // Timer t2 = chrono::steady_clock::now();
        // printElapsedTime("normal desc: ", t1, t2);

        Timer t3 = chrono::steady_clock::now();
        orbFeatures->computeDescNormal(image_gray1, semantic1, keypoints1, sem_descriptor1);
        orbFeatures->computeDescNormal(image_gray2, semantic2, keypoints2, sem_descriptor2);
        Timer t4 = chrono::steady_clock::now();
        printElapsedTime("sem desc: ", t3, t4);

        vector<DMatch> matches;
        // matcher->match(sem_descriptor1, sem_descriptor2, matches);
        orbFeatures->matchDescNormal(sem_descriptor1, sem_descriptor2, matches);
        
        auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
            return m1.distance < m2.distance;
        });

        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;

        vector<DMatch> goodMatches;

        for (int i=0; i<sem_descriptor1.rows; i++){
            if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
                goodMatches.push_back(matches[i]);
            }
        }

        //--- Step 5: Visualize the Matching result
        Mat outImage1, outImage2;
        Mat outImage_gray1, outImage_gray2;
        Mat image_matches;
        Mat image_goodMatches;

        drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        
        drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches,
            Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches,
            Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // /* Results */
        cout << "-- Number of matches: " << matches.size() << endl << endl;
        
        cout << "-- Min dist: " << min_dist << endl;
        cout << "-- Max dist: " << max_dist << endl;
        cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
        cout << "In total, we get " << goodMatches.size() << "/" << matches.size() << " good pairs of feature points." << endl << endl;

        // R_old = R.clone();
        // t_old = t.clone();
        
        pose_estimation->pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t, K);
        
        // if(goodMatches.size() > MIN_FEATURES && min_dist < 40){
        //     read_flag = cap.read(image2);
        //     cap_seg.read(semantic2);
        // }else{
        //     image1 = image2.clone();
        //     semantic1 = semantic2.clone();

        //     read_flag = cap.read(image2);
        //     cap.read(semantic2);

        //     pose_estimation->updateMap2d(R_old,t_old);
        // }
        
        pose_estimation->updateMap2d(R,t);

        string flag;
        t_hat = pose_estimation->vee2hat(t);
        int counter = 0;

        for(DMatch m : goodMatches){  // For each matched pair {(p1, p2)}_n, do...
            // Pixel Coordinates to Normalized Coordinates, {(p1, p2)}_n to {(x1, x2)}_n
            Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
            Point2f x2 = pixel2cam(keypoints2[m.trainIdx].pt, K);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2

            // Convert to Homogeneous Coordinates
            Mat xh1 = (Mat_<double>(3,1) << x1.x, x1.y, 1);
            Mat xh2 = (Mat_<double>(3,1) << x2.x, x2.y, 1);

            // Calculate Epipolar Constraint
            double res = ((cv::Mat)(xh2.t()*t_hat*R*xh1)).at<double>(0);

            if(res > -1e-2 && res < 1e-2){
                flag = "Ok!";
                counter++;
            }else
                flag = "Failed!";

            // printf("x2^T*E*x1 = % 01.19f\t%s\n", res, flag.c_str());
        }

        // cout << "\nFinal Result: " << counter << "/" << goodMatches.size() << " Features Pairs respected the Epipolar Constraint!"<< endl << endl;

        /* Display */
        //input
        // imshow("image1", image1);
        // imshow("image2", image2);

        // imshow("semantic1", semantic1);
        // imshow("semantic2", semantic2);

        //output
        // imshow("outImage1", outImage1);
        // imshow("outImage2", outImage2);

        // imshow("outImage_gray1", outImage_gray1);
        // imshow("outImage_gray2", outImage_gray2);
        // imshow("image_matches", image_matches);
        imshow("image_goodMatches", image_goodMatches);
        waitKey(100);

        image1 = image2.clone();
        semantic1 = semantic2.clone();
    }

    pose_estimation->closeFiles();
    waitKey(0);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    
    
    return 0;
}
