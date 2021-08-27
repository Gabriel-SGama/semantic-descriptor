
#include "include/libUtils.hpp"

using namespace std;
using namespace Eigen;
using namespace cv;


void printCharVec(const uint32_t &desc){

    cout << ((desc >> 24) & 255) << ", ";
    cout << ((desc >> 16) & 255) << ", ";
    cout << ((desc >> 8) & 255) << ", ";
    cout << (desc & 255);

}


/* Basic */
void printVec(const char text[], const std::vector<uint32_t> &vec){
    cout << text << "[";
    for(size_t i=0; i < vec.size(); i++){
        if(i != vec.size()-1){
            printCharVec(vec.at(i));
            // cout << vec.at(i) << ";";
            cout << ", ";
        }else{
            printCharVec(vec.at(i));

            // cout << vec.at(i);
        }
    }
    cout << "]" << endl;
}


template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx){
    // Starting and Ending iterators
    auto start = arr.begin() + begin_idx;
    auto end = arr.begin() + end_idx + 1;

    // To store the sliced vector
    TTypeVec result(end_idx - begin_idx + 1);

    // Copy vector using copy function()
    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}

/* Chrono */
void printElapsedTime(const char text[], Timer t1, Timer t2){
    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << text << time_elapsed.count() << " s" << endl;
}


/* ========================== */
/*  Eigen3/Sophus' Functions  */
/* ========================== */
template <typename TTypeEigenMat>
void printMatrix(const char text[], TTypeEigenMat mat){
    cout << text << endl;
    cout << mat << "\n" << "(" << mat.rows() << ", " << mat.cols() << ")" << endl << endl;
}

template <typename TTypeEigenVec>
void printVector(const char text[], TTypeEigenVec vec){
    cout << text << endl;
    cout << vec << "\n" << "(" << vec.size() << ",)" << endl << endl;
}

template <typename TTypeEigenQuat>
void printQuaternion(const char text[], TTypeEigenQuat quat){
    cout << text << quat.coeffs().transpose() << endl << endl;
}

double RMSE(const Vector3d &est, const Vector3d &gt){
    double sum = 0.0;
    int N = est.size();

    for(int i=0; i<N; i++){
        sum += pow(est[i]-gt[i], 2.0);
    }

    return sqrt(sum/(double)N);
}

/**
 * @brief Convert Normalized Coordinates to Pixel Coordinates (Image Plane, f=1)
 *
 * @param x Point2f in Normalized Coordinates, x=(x,y)=(X/Z, Y/Z)
 * @param K Intrinsic Parameters Matrix
 * @return Point2f in Pixel Coordinates Coordinates, p=(u,v)
 */
Vector2d cam2pixel(const Vector3d &P, const Matrix3d &K) {
    return Vector2d(
        K(0, 0)*P[0]/P[2] + K(0, 2),  // u = fx*x + cx
        K(1, 1)*P[1]/P[2] + K(1, 2)   // v = fy*y + cy
    );
}



/* ==================== */
/*  OpenCV's Functions  */
/* ==================== */
int checkImage(const cv::Mat &image){
    // Check if the data is correctly loaded
    if (image.data == nullptr) {
        cerr << "File doesn't exist." << endl;
        return 0;
    } else{
        cout << "Successful." << endl;
    }

    // Check image type
    if (image.type()!= CV_8UC1 && image.type() != CV_8UC3){
        // We need grayscale image or RGB image
        cout << "Image type incorrect!" << endl;
        return 0;
    }

    return 1;
}

string type2str(int type){
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch(depth){
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void printImageInfo(const char var[], const cv::Mat &image){
/*  +--------+----+----+----+----+----+----+----+----+
    |        | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
    +--------+----+----+----+----+----+----+----+----+
    | CV_8U  |  0 |  8 | 16 | 24 | 32 | 40 | 48 | 56 |
    | CV_8S  |  1 |  9 | 17 | 25 | 33 | 41 | 49 | 57 |
    | CV_16U |  2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
    | CV_16S |  3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
    | CV_32S |  4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
    | CV_32F |  5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
    | CV_64F |  6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
    +--------+----+----+----+----+----+----+----+----+ */

    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);

    cout << var << ":" << endl;
    cout << "(" << image.rows << "," << image.cols << "," << image.channels() << ")";  // (Height, Width, Channels)
    cout << ", " << type2str(image.type()) << endl;
    cout << "min: " << minVal << ", max: " << maxVal << endl << endl;
}

void printMatrix(const char text[], cv::Mat var){
    cout << text << var << "\n(" << var.rows << ", " << var.cols << ")" << endl << endl;
}

void printMatrix(const char text[], cv::MatExpr var){
    cout << text << var << "\n(" << var.size().height << ", " << var.size().width << ")" << endl << endl;
}

/**
 * @brief Convert Pixel Coordinates to Normalized Coordinates (Image Plane, f=1)
 *
 * @param p Point2f in Pixel Coordinates, p=(u,v)
 * @param K Intrinsic Parameters Matrix
 * @return Point2f in Normalized Coordinates, x=(x,y)=(X/Z, Y/Z)
 */
Point2f pixel2cam(const Point2f &p, const Mat &K) {
    return Point2f(
        (p.x - K.at<double>(0, 2))/K.at<double>(0, 0),  // x = (u-cx)/fx
        (p.y - K.at<double>(1, 2))/K.at<double>(1, 1)   // y = (v-cy)/fy
    );
}

void calcHist(const Mat src_raw, string img_name){
    Mat src;
    if(src_raw.channels()==1){
        cvtColor(src_raw, src, COLOR_GRAY2BGR);
    }else{
        src_raw.copyTo(src);
    }

    //! [Separate the image in 3 places ( B, G and R )]
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    //! [Separate the image in 3 places ( B, G and R )]

    //! [Establish the number of bins]
    int histSize = 256;
    //! [Establish the number of bins]

    //! [Set the ranges ( for B,G,R) )]
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    //! [Set the ranges ( for B,G,R) )]

    //! [Set histogram param]
    bool uniform = true, accumulate = false;
    //! [Set histogram param]

    //! [Compute the histograms]
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    //! [Compute the histograms]

    //! [Draw the histograms for B, G and R]
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    //! [Draw the histograms for B, G and R]

    //! [Normalize the result to ( 0, histImage.rows )]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    //! [Normalize the result to ( 0, histImage.rows )]

    //! [Draw for each channel]
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    //! [Draw for each channel]

    //! [Display]
    imshow(img_name, src);
    img_name += ", hist";
    imshow(img_name, histImage);
}
