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
   
    dataEvoPath = "../carla-pythonAPI/mapGenRansac.txt";
    dataEvoPtr.open(dataEvoPath);
    
    /*
    textMapPath = "../carla-pythonAPI/pos2.txt";
    mapFilePtr.open(textMapPath);

    float x, ini_x, max_x = 100;
    float y, ini_y, max_y = 100;
    float z;
    float row;
    float pitch;
    float yaw;
    
    int nlines = count(istreambuf_iterator<char>(mapFilePtr), 
             istreambuf_iterator<char>(), '\n');


    float (*pos)[6] = new float[nlines][6];

    mapFilePtr.clear();
    mapFilePtr.seekg(0);
    mapFilePtr >> x >> y >> z >> row >> pitch >> yaw;

    for(int idx = 0; idx < nlines; idx++){
        

        pos[idx][0] = x;
        pos[idx][1] = y;
        pos[idx][2] = z;

        cout  << idx << " | " << x << " | " << pos[idx][0] << endl;

        max_x = max(max_x, abs(x-pos[0][0]));
        max_y = max(max_y, abs(y-pos[0][1]));

        mapFilePtr >> x >> y >> z >> row >> pitch >> yaw;
        mapFilePtr.get();
    }

    mapFilePtr.close();
    
    ini_x = pos[0][0];
    ini_y = pos[0][1];

    for (int i = 0; i < nlines; i++){
        x = pos[i][0];
        y = pos[i][1];
        // cout << x << endl;
        // cout << y << endl;

        circle(map, Point2i((int((x-ini_x)*(512/(2.5*max_x)) + 512/2)), int((y-ini_y)*(512/(2.5*max_y))+ 512/2)), 1, 255, FILLED);
    }
    
    imshow("map", map);
    cv::waitKey(0);
    */
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
    // Mat F = findFundamentalMat(points1, points2, cv::FM_8POINT);  // 8-Points Algorithm
    // Timer t2 = chrono::steady_clock::now();

    //--- Calculate the Essential Matrix
    Mat E = findEssentialMat(points1, points2, focal_length, principal_point);  // Remember: E = t^*R = K^T*F*K, Essential matrix needs intrinsics info.
    // Timer t3 = chrono::steady_clock::now();

    //--- Calculate the Homography Matrix
    //--- But the scene in this example is not flat, and then Homography matrix has little meaning.
    // Mat H = findHomography(points1, points2, RANSAC, 3);
    // Timer t4 = chrono::steady_clock::now();

    //--- Restore Rotation and Translation Information from the Essential Matrix, E = t^*R
    // In this program, OpenCV will use triangulation to detect whether the detected point’s depth is positive to select the correct solution.
    // This function is only available in OpenCV3!
    cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
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


vector<DMatch> Pose_estimation::ransac(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, Mat &R, Mat &t, Mat &K){
    
    static int nrMatchesLow = 0;
    // Intrinsics Parameters
    Point2d principal_point(K.at<double>(0, 2), K.at<double>(1,2)); // Camera Optical center coordinates
    double focal_length = K.at<double>(1,1);                        // Camera focal length

    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)  // FIXME: Pixel Coordinates? Isto está correto?
    vector<Point2f> points1, points2;  // {(x1, x2)}_n  // FIXME: {(p1, p2)}_n?
    vector<Point2f> points1_ransc, points2_ransc;

    for (int i=0; i < (int) matches.size(); i++){  // For each matched pair {(p1, p2)}_n, do...
        // Convert pixel coordinates to camera normalized coordinates
        // cout << i << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
        points1.push_back(keypoints1[matches[i].queryIdx].pt);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1  // FIXME: Pixel Coordinates? Isto está correto?
        points2.push_back(keypoints2[matches[i].trainIdx].pt);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2  // FIXME: Pixel Coordinates? Isto está correto?
    }


    
    vector<DMatch> goodMatchesRansac;
    int L = 0;
    // int n_ransac_min = min(_N_RANSAC, (int) (matches.size()*0.8));
    int n_ransac_min = (int) (matches.size()*0.92);
    
    Mat t_hat;
    
    while ((int) goodMatchesRansac.size() < n_ransac_min && L < _L_MAX) {
        points1_ransc.clear();
        points2_ransc.clear();

        goodMatchesRansac.clear();
        
        //6 randon points
        for(int i=0; i < 6; i++){
            int pos = rand() % matches.size();
            points1_ransc.push_back(points1.at(pos));
            points2_ransc.push_back(points2.at(pos));
        }
        //--- Calculate the Essential Matrix
        Mat E = findEssentialMat(points1_ransc, points2_ransc, focal_length, principal_point);  // Remember: E = t^*R = K^T*F*K, Essential matrix needs intrinsics info.

        cv::recoverPose(E, points1_ransc, points2_ransc, R, t, focal_length, principal_point);
        t_hat = this->vee2hat(t);

        for(DMatch m : matches){
    
            Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
            Point2f x2 = pixel2cam(keypoints2[m.trainIdx].pt, K);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2

            // Convert to Homogeneous Coordinates
            Mat xh1 = (Mat_<double>(3,1) << x1.x, x1.y, 1);
            Mat xh2 = (Mat_<double>(3,1) << x2.x, x2.y, 1);
            double res = ((cv::Mat)(xh2.t()*t_hat*R*xh1)).at<double>(0);

            if(res > -0.01 && res < 0.01)
                goodMatchesRansac.push_back(m);
        }

        L++;
    }
    
    cout << "ransac matches min: " << n_ransac_min << endl;
    cout << "ransac matches nr: " << goodMatchesRansac.size() << endl;
    cout << "L: " << L << endl;


    if((int) goodMatchesRansac.size() >= n_ransac_min){
        points1_ransc.clear();
        points2_ransc.clear();
        for (int i=0; i < (int) goodMatchesRansac.size(); i++){  // For each matched pair {(p1, p2)}_n, do...
            points1_ransc.push_back(keypoints1[goodMatchesRansac[i].queryIdx].pt);
            points2_ransc.push_back(keypoints2[goodMatchesRansac[i].trainIdx].pt);
        }

        nFeaturesUsed.push_back(goodMatchesRansac.size());

        Mat E = findEssentialMat(points1_ransc, points2_ransc, focal_length, principal_point);
        cv::recoverPose(E, points1_ransc, points2_ransc, R, t, focal_length, principal_point);
    
    }else{
        cout << "numero minimo de matches nao atingido" << endl;
        
        nFeaturesUsed.push_back(matches.size());
        nrMatchesLow++;
        
        Mat E = findEssentialMat(points1, points2, focal_length, principal_point);
        cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
    }

    cout << "total de matches que falharam: " << nrMatchesLow << endl;

    nL.push_back(L);
    ntotalFeatures.push_back(matches.size());
    nFailedRansac.push_back(nrMatchesLow);

    return goodMatchesRansac;
}

Mat Pose_estimation::vee2hat(const Mat &var){
    Mat var_hat = (Mat_<double>(3,3) <<
                         0.0, -var.at<double>(2,0),  var.at<double>(1,0),
         var.at<double>(2,0),                  0.0, -var.at<double>(0,0),
        -var.at<double>(1,0),  var.at<double>(0,0),                 0.0);  // Inline Initializer

    //printMatrix("var_hat:", var_hat);

    return var_hat;
}

void Pose_estimation::plotInfo(){
    int min = -1;

    cv::namedWindow("graph");
    cv::moveWindow("graph", 0, 1000);
    
    const int _GRAPH_HEIGHT = 600, _GRAPH_WIDTH = 1000;

    Mat graph = cv::Mat(_GRAPH_HEIGHT, _GRAPH_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
   

    //verifica se e uma curva por produto vetorial
    const int pointsToConsiderate = 8;
    Point2d vecTemp;
    vector<Point2d> vec;
    vector<bool> curveDetect;
    double vecProd;


    cout << "making vectors" << endl;
    for (int i = 1; i < (int)translations.size(); i++) {
        vecTemp.x = translations.at(i).at<double>(0,0) - translations.at(i-1).at<double>(0,0);
        vecTemp.y = translations.at(i).at<double>(0,2) - translations.at(i-1).at<double>(0,2);
        vec.push_back(vecTemp);
    }
    

    for (int i = 0; i < pointsToConsiderate/2; i++) {
        curveDetect.push_back(false);
    }
    

    cout << "making vectors prod" << endl;
    for (int i = pointsToConsiderate/2; i < (int)translations.size() - pointsToConsiderate/2 - 1; i++){
        vecProd = 0;
        
        for (int k = -pointsToConsiderate/2; k < pointsToConsiderate/2; k++){
            vecProd += abs(vec.at(i + k).x*vec.at(i + k + 1).y - vec.at(i + k).y*vec.at(i + k + 1).x);
        }
        cout << vecProd << endl;
        if(vecProd > 1.8)
            curveDetect.push_back(true);
        else
            curveDetect.push_back(false);
    }
    
    for (int i = 0; i < pointsToConsiderate/2 + 1; i++) {
        curveDetect.push_back(false);
    }

    cout << "making graph" << endl;

    int maxValue = *max_element(begin(nFailedRansac), end(nFailedRansac)); 

    const float dif = (maxValue - min)*1.2;
    const float scaleY = _GRAPH_HEIGHT / dif;
    const float scaleX = (float)_GRAPH_WIDTH / (nFailedRansac.size() - 1);
    const float offset = -min*1.2;

    Scalar color;

    for (int i = 1; i < (int)nFailedRansac.size() - 1; i++) {
        //----------------N FAILED RANSAC----------------
        color = Scalar(255,255,255);
        if(curveDetect.at(i))
            color = Scalar(0,0,255);
        cv::line(graph, cv::Point((i - 1) * scaleX, _GRAPH_HEIGHT - (nFailedRansac[i - 1] + offset) * scaleY),
                        cv::Point(i*scaleX, _GRAPH_HEIGHT - (nFailedRansac[i] + offset) * scaleY),
                        color);
    }

    imshow("graph", graph);

    int _SIZE = 512;
    map.setTo(Scalar::all(0));


    double x=0, y=0;

    double max_x=10, max_y=10;
    
    double x_ini = 0, y_ini = 0;

    for (auto &curr_t: translations){
        x = curr_t.at<double>(0,0);
        y = curr_t.at<double>(0,2);

        max_x = max(max_x, abs(x));
        max_y = max(max_y, abs(y));
    }
    int colorT;
    int i=0;
    for (auto &curr_t: translations){
        colorT = 255;
        if(curveDetect.at(i))
            colorT = 125;
        
        x = curr_t.at<double>(0,0);
        y = curr_t.at<double>(0,2);

        line(map,Point2f((int)x_ini*_SIZE/(3*max_x) + _SIZE/2,(int)y_ini*_SIZE/(3*max_y) + _SIZE/2), Point2f((int)x*_SIZE/(3*max_x) + _SIZE/2,(int)y*_SIZE/(3*max_y) + _SIZE/2), colorT);
        // circle(map, Point2f((int)x*_SIZE/(3*max_x) + _SIZE/2,(int)y*_SIZE/(3*max_y) + _SIZE/2),
                // 1, 255, FILLED);

        x_ini = x;
        y_ini = y;
        i++;
    }

    imshow("map", map);

}



void Pose_estimation::updateMap2d(const Mat &R,const Mat &t){
    int _SIZE = 512;

    map.setTo(Scalar::all(0));

    R_total = R_total*R;
    Mat t_rot = R_total*t;

    double* ptr = &R_total.at<double>(0,0);

    
    Mat t_result = t_rot + translations.back();

    dataEvoPtr << fixed << setprecision(4) << (float)ptr[0] << " " << 
                (float)ptr[1] << " " <<
                (float)ptr[2] << " " <<
                (float)t_result.at<double>(0,0) << " " <<
                (float)ptr[3] << " " <<
                (float)ptr[4] << " " <<
                (float)ptr[5] << " " <<
                (float)t_result.at<double>(0,1) << " " <<
                (float)ptr[6] << " " <<
                (float)ptr[7] << " " <<
                (float)ptr[8] << " " <<
                (float)t_result.at<double>(0,2) << "\n";


    translations.push_back(t_result);

    double x=0, y=0;

    static double max_x=10, max_y=10;
    
    double x_ini = 0, y_ini = 0;

    for (auto &curr_t: translations){
        x = curr_t.at<double>(0,0);
        y = curr_t.at<double>(0,2);

        max_x = max(max_x, abs(x));
        max_y = max(max_y, abs(y));

        line(map,Point2f((int)x_ini*_SIZE/(3*max_x) + _SIZE/2,(int)y_ini*_SIZE/(3*max_y) + _SIZE/2), Point2f((int)x*_SIZE/(3*max_x) + _SIZE/2,(int)y*_SIZE/(3*max_y) + _SIZE/2), 255);
        // circle(map, Point2f((int)x*_SIZE/(3*max_x) + _SIZE/2,(int)y*_SIZE/(3*max_y) + _SIZE/2),
                // 1, 255, FILLED);

        x_ini = x;
        y_ini = y;

    }

    imshow("map", map);
    // waitKey(1);
}


void Pose_estimation::closeFiles(){
    dataEvoPtr.close();
}