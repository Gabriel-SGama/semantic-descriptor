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

// const float factorPI = CV_PI/180.0;



vector<String> readFile(string filePath, vector<Mat>imageSeq)
{
    vector<String> filenames;

    glob(filePath, filenames);

    Mat myImage;

    imageSeq.reserve(filenames.size());
    cout << "reading " << filenames.size() << " files..." << endl;
    // for (size_t i = 0; i < filenames.size(); ++i) {
    //     imageSeq.push_back(imread(filenames[i]));
    //     if(!(i%(filenames.size()/8)))
    //         cout << "arquivo " << i << endl;
    // }
    return filenames;
}


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
    

    // vector<Mat> imageSeq;
    // vector<String> filenames = readFile("/home/gama/Documents/Datasets/data_odometry_gray/dataset/sequences/00/image_0/*.png", imageSeq);
    // int nlines = count(istreambuf_iterator<char>(mapGTin), 
    //          istreambuf_iterator<char>(), '\n');

    // mapGTin.clear();
    // mapGTin.seekg(0);
    
    cout << "[orb_cv] Hello!" << endl << endl;
    
    ofstream outdata;
    outdata.open("data.txt");

    ORBFeatures *orbFeatures = new ORBFeatures(nfeatures, nrBrief, nSemrBrief, patch_size, half_patch_size);
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
    
    // sem_descriptor1 = Mat::zeros(nfeatures, nrBrief/32, CV_32SC1);
    // sem_descriptor2 = Mat::zeros(nfeatures, nrBrief/32, CV_32SC1);
    // sem_descriptor1 = Mat::zeros(nfeatures, nrBrief/32 + nSemrBrief/6, CV_32SC1);
    // sem_descriptor2 = Mat::zeros(nfeatures, nrBrief/32 + nSemrBrief/6, CV_32SC1);

    while(cap.isOpened()){
    // while(frameid < filenames.size()){
        frameid++;

        //ignores begining
        if(frameid < _OFFSET_IN_VID){
            cap.read(image1);
            cap_seg.read(semantic1);
            // image1 = imread(filenames[frameid-1]);
            mapGTin >> r11 >> r12 >> r13 >> x >> r21 >> r22 >> r23 >> y >> r31 >> r32 >> r33 >> z;
            mapGTin.get();
            continue;
        }
        
        for(int k = 0; k < _FRAMES_JUMPS; k++){
            read_flag = cap.read(image2);
            cap_seg.read(semantic2);
            // image2 = imread(filenames[frameid]);

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

        Timer t1 = chrono::steady_clock::now();
        orbFeatures->computeDesc(image_gray1, semantic1, keypoints1, sem_descriptor1, _BRIEF_DESC);
        orbFeatures->computeDesc(image_gray2, semantic2, keypoints2, sem_descriptor2, _BRIEF_DESC);
        Timer t2 = chrono::steady_clock::now();
        printElapsedTime("desc time: ", t1, t2);

        vector<DMatch> matches;
        Timer t3 = chrono::steady_clock::now();
        // matcher->match(sem_descriptor1, sem_descriptor2, matches);
        orbFeatures->matchDescNormal(sem_descriptor1, sem_descriptor2, matches);
        Timer t4 = chrono::steady_clock::now();
        printElapsedTime("match time: ", t3, t4);

        auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
            return m1.distance < m2.distance;
        });

        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;

        vector<DMatch> goodMatches;

        for (int i=0; i< (int)keypoints1.size(); i++){
            if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
                goodMatches.push_back(matches[i]);
            }
        }

        //--- Step 5: Visualize the Matching result
        Mat outImage1, outImage2;
        Mat outImage_gray1, outImage_gray2;
        Mat image_matches;
        Mat image_goodMatches, image_goodMatchesRansac;

        // drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        cout << keypoints1.size() << endl;
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
        
        Timer t5 = chrono::steady_clock::now();
        pose_estimation->pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t, K);
        Timer t6 = chrono::steady_clock::now();
        printElapsedTime("pose estimation time: ", t5, t6);
        // vector<DMatch> goodMatchesRansac = pose_estimation->ransac(keypoints1, keypoints2, goodMatches, R, t, K);
        // drawMatches(image1, keypoints1, image2, keypoints2, goodMatchesRansac, image_goodMatchesRansac,
        //     Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        pose_estimation->updateMap2d(R,t);

        string flag;
        t_hat = pose_estimation->vee2hat(t);

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
        imshow("Good Matches", image_goodMatches);
        // imshow("Matches Ransac", image_goodMatchesRansac);
        waitKey(1);

        image1 = image2.clone();
        semantic1 = semantic2.clone();
    }

    pose_estimation->closeFiles();
    // pose_estimation->plotInfo();
    waitKey(0);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    
    
    return 0;
}
