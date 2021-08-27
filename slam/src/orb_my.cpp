/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <nmmintrin.h>
#include <bitset>

/* OpenCV Libraries */
// #include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "include/libUtils.hpp"
#include "include/ORBFeatures.hpp"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../carlaData/image/id00044.png";
string image2_filepath = "../carlaData/image/id00045.png";

string semantic1_filepath = "../carlaData/semantic/id00044.png";
string semantic1color_filepath = "../carlaData/semantic/idcolor00044.png";
string semantic2_filepath = "../carlaData/semantic/id00045.png";


const int nfeatures = 500;
const int nrBrief = 256;
const int nSemrBrief = 48;
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 24;
// const float factorPI = CV_PI/180.0;

double matches_lower_bound = 60.0;

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl << endl;

    ORBFeatures *orbFeatures = new ORBFeatures();

    /* Load the images */
    Mat image1 = imread(image1_filepath, IMREAD_COLOR);
    Mat image2 = imread(image2_filepath, IMREAD_COLOR);
    Mat image_gray1(image1.size().height, image1.size().width, CV_8UC1);
    Mat image_gray2(image2.size().height, image2.size().width, CV_8UC1);
    cvtColor(image1, image_gray1, COLOR_BGR2GRAY);
    cvtColor(image2, image_gray2, COLOR_BGR2GRAY);

    Mat semantic1 = imread(semantic1_filepath, IMREAD_COLOR);
    // Mat semantic1color = imread(semantic1color_filepath, IMREAD_COLOR);
    
    Mat semantic2 = imread(semantic2_filepath, IMREAD_COLOR);
    // assert(image1.data != nullptr && image2.data != nullptr);  // FIXME: I think this its not working!

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Mat sem_descriptor1, sem_descriptor2;

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //--- Step 1: Detect the position of the Oriented FAST keypoints (Corner Points)
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image_gray1, keypoints1);
    detector->detect(image_gray2, keypoints2);
    Timer t2 = chrono::steady_clock::now();

    //--- Step 2: Calculate the BRIEF descriptors based on the position of Oriented FAST keypoints
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    
    // orbFeatures->computeDesc(image_gray1, keypoints1, descriptors1);
    // orbFeatures->computeDesc(image_gray2, keypoints2, descriptors2);
    
    orbFeatures->convertDesc(descriptors1, sem_descriptor1, semantic1);
    orbFeatures->convertDesc(descriptors2, sem_descriptor2, semantic2);

    orbFeatures->computeSemanticDesc(semantic1, keypoints1, sem_descriptor1);
    orbFeatures->computeSemanticDesc(semantic2, keypoints2, sem_descriptor2);

    Timer t3 = chrono::steady_clock::now();

    // cout << keypoints1.size() << endl;
    // cout << descriptors1.row(0) << endl;
    // cout << sem_descriptor1.row(0) << endl;
    // cout << "--------------------------------" << endl;
    
    /* ------------------- */
    /*  Features Matching  */
    /* ------------------- */
    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    Timer t4 = chrono::steady_clock::now();
    orbFeatures->matchDesc(sem_descriptor1, sem_descriptor2, matches);
    // matcher->match(descriptors1, descriptors2, matches);
    Timer t5 = chrono::steady_clock::now();

    /* -------------------- */
    /*  Features Filtering  */
    /* -------------------- */
    //--- Step 4: Correct matching selection
    /* Calculate the min & max distances */
    /** Parameters: 
     * @param[in] __first – Start of range.
    /* @param[in] __last – End of range.
    /* @param[in] __comp – Comparison functor.
    /* @param[out] make_pair(m,M) Return a pair of iterators pointing to the minimum and maximum elements in a range.
     */
    Timer t6 = chrono::steady_clock::now();
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
        //cout << m1.distance << " " << m2.distance << endl;
        return m1.distance < m2.distance;
    });  

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    /* Perform Filtering */
    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching
    // as wrong. But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    vector<DMatch> goodMatches;

    Timer t7 = chrono::steady_clock::now();
    for (int i=0; i<sem_descriptor1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }
    Timer t8 = chrono::steady_clock::now();

    //--- Step 5: Visualize the Matching result
    Mat outImage1, outImage2;
    Mat outImage_gray1, outImage_gray2;
    Mat image_matches;
    Mat image_goodMatches;

    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    // drawKeypoints(image_gray1, keypoints_gray1, outImage_gray1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // drawKeypoints(image_gray2, keypoints_gray2, outImage_gray2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    // /* Results */
    printElapsedTime("ORB Features Extraction: ", t1, t3);
    printElapsedTime(" | Oriented FAST Keypoints detection: ", t1, t2);
    printElapsedTime(" | BRIEF descriptors calculation: ", t2, t3);
    cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

    printElapsedTime("ORB Features Matching: ", t4, t5);
    cout << "-- Number of matches: " << matches.size() << endl << endl;
    
    printElapsedTime("ORB Features Filtering: ", t6, t8);
    printElapsedTime(" | Min & Max Distances Calculation: ", t6, t7);
    printElapsedTime(" | Filtering by Hamming Distance: ", t7, t8);
    cout << "-- Min dist: " << min_dist << endl;
    cout << "-- Max dist: " << max_dist << endl;
    cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
    cout << "In total, we get " << goodMatches.size() << "/" << matches.size() << " good pairs of feature points." << endl << endl;

    /* Display */
    //input
    // imshow("image1", image1);
    // imshow("image2", image2);

    imshow("semantic1", semantic1);
    // imshow("semantic1color", semantic1color);
    // imshow("semantic2", semantic2);
    //output
    imshow("outImage1", outImage1);
    // imshow("outImage2", outImage2);

    // imshow("outImage_gray1", outImage_gray1);
    // imshow("outImage_gray2", outImage_gray2);
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches);
    // cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    /* Save */
    // imwrite("../../orb_features/src/results/orb_cv_goodMatches.png", image_goodMatches);

    cout << "Done." << endl;

    return 0;
}
