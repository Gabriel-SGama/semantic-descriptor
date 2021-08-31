#include <nmmintrin.h>
#include <math.h>

#include "include/ORBFeatures.hpp"

using namespace std;
using namespace cv;

ORBFeatures::ORBFeatures(int maxFeatures, int nrBrief, int nSemrBrief, int patch_size, int half_patch_size, float matches_lower_bound){

    this->maxFeatures = maxFeatures;
    this->nrBrief = nrBrief;
    this->nSemrBrief = nSemrBrief;
    this->patch_size = patch_size;
    this->half_patch_size = half_patch_size;
    this->matches_lower_bound = matches_lower_bound;

    scaleFactor = 1.2;
    nLevels = 8;
    imagePyramid.resize(nLevels);

    imagePyramidScale.resize(nLevels);
    imagePyramidScale[0]=1.0f;

    for(int i=1; i<nLevels; i++)
    {
        imagePyramidScale[i]=imagePyramidScale[i-1]/scaleFactor;
    }
    
    // for(int i = 0; i < nLevels; i++){
    //     imagePyramidScale[i] = (float)pow(scaleFactor, i);
    // }
    
}


void ORBFeatures::createPyramid(const Mat img){
    const int EDGE_THRESHOLD = 31;

    for (int level = 0; level < nLevels; ++level){

        float scale = imagePyramidScale[level];
        Size sz(cvRound((float)img.cols*scale), cvRound((float)img.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, img.type()), masktemp;
        imagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(imagePyramid[level-1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(imagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            copyMakeBorder(img, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
            // imagePyramid[0] = img.clone();            
        }
        
    }

}



void ORBFeatures::computeDesc(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor){
    
    createPyramid(img);
    
    for(int i = 0; i < nLevels; i++)
        GaussianBlur(imagePyramid[i], imagePyramid[i], Size(7, 7), 2, 2, BORDER_REFLECT_101);
    // GaussianBlur(img, img, Size(7, 7), 2, 2, BORDER_REFLECT_101);
    

    descriptor = Mat::zeros(keypoints.size(), nrBrief/8 /*+ nSemrBrief/6*/, CV_8UC1);
    // descriptor.create(500, 32, CV_8UC1);
    
    
    int count_kp = 0;


    for(auto &kp: keypoints){
        // cout << kp.octave << " | " <<kp.pt << " | " << kp.size << endl;
          // float m01=0.0, m10=0.0;

        // for(int y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; y++){
        //     for(int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; x++){
        //         m01 += y*img.at<uchar>(kp.pt.y+y, kp.pt.y+x);
        //         m10 += x*img.at<uchar>(kp.pt.y+y, kp.pt.y+x);
        //     }
        // }
        
        // kp.angle = atan2(m01, m10);
        // float angle = cvRound((kp.angle*factorPI)/(CV_PI/15.0));
        
        // angle *= CV_PI/15.0;

        // float sin_kp = sin(angle);
        // float cos_kp = cos(angle);

        float sin_kp = sin(kp.angle*factorPI);
        float cos_kp = cos(kp.angle*factorPI);


        // const u_char* center = &imagePyramid[kp.octave].at<uchar>(cvRound(kp.pt.y*imagePyramidScale[kp.octave]), cvRound(kp.pt.x*imagePyramidScale[kp.octave]));
        // const int step = (int)imagePyramid[kp.octave].step;
        // float a = cos_kp;
        // float b = sin_kp;
        
        float scale = imagePyramidScale[kp.octave];
        // #define GET_VALUE(idx_p) /*\//
        //     center[cvRound(scale*cvRound(ORB_pattern[idx_p]*sin_kp + ORB_pattern[idx_p+1]*cos_kp)*step + \
        //            cvRound(ORB_pattern[idx_p]*cos_kp - ORB_pattern[idx_p+1]*sin_kp))]
        
        
        for(int i = 0; i < 32; i++){
            u_char desc = 0;
            int idx = i*8*4;
      
            for(int pt = 0; pt < 8*4; pt+=4){
            // for(int pt = 0; pt < 8*4; pt+=4){
                
                Point2f p(ORB_pattern[idx + pt], ORB_pattern[idx + pt + 1]);
                Point2f q(ORB_pattern[idx + pt + 2],ORB_pattern[idx + pt + 3]);
                
                Point2f pp = Point2f(cvRound((cos_kp*p.x - sin_kp*p.y + scale*kp.pt.x)), cvRound((sin_kp*p.x + cos_kp*p.y + scale*kp.pt.y)));
                Point2f qq = Point2f(cvRound((cos_kp*q.x - sin_kp*q.y + scale*kp.pt.x)), cvRound((sin_kp*q.x + cos_kp*q.y + scale*kp.pt.y)));

                // Point2f pp = Point2f((cos_kp*p.x - sin_kp*p.y), (sin_kp*p.x + cos_kp*p.y)) + kp.pt;
                // Point2f qq = Point2f((cos_kp*q.x - sin_kp*q.y), (sin_kp*q.x + cos_kp*q.y)) + kp.pt;

                // desc |= (GET_VALUE(idx + pt) < GET_VALUE(idx + pt + 2)) << cvRound(pt*0.25);
                if(imagePyramid[kp.octave].at<uchar>(pp.y, pp.x) < imagePyramid[kp.octave].at<uchar>(qq.y, qq.x))
                    desc |= 1 << cvRound(pt*0.25);

            }
            descriptor.at<uchar>(count_kp, i) = desc;
        }
        count_kp++;
    
        // #undef GET_VALUE
    }

}


void ORBFeatures::computeSemanticDesc(const Mat &sem_img, const vector<KeyPoint> &keypoints, Mat &descriptor){
    // descriptor.create(maxFeatures, nrBrief/32 + nSemrBrief/6, CV_32SC1);

    int count_kp = 0;

    for(auto &kp: keypoints){
        float sin_kp = sin(kp.angle*factorPI);
        float cos_kp = cos(kp.angle*factorPI);
    
        for(int i = 0; i < nSemrBrief/6; i++){
            int32_t desc = 0;
            int idx = i*6*2;
            for(int pt = 0; pt < 6*2; pt+=2){
                Point2f p(ORB_pattern[idx + pt], ORB_pattern[idx + pt + 1]);

                Point2f pp = Point2f(round(cos_kp*p.x - sin_kp*p.y + kp.pt.x), round(sin_kp*p.x + cos_kp*p.y + kp.pt.y));
                desc |= sem_img.at<uchar>(pp.y, pp.x*3 + 2) << int(pt*3); /*int(pt*0.5*6)*/
            }
            descriptor.at<u_int32_t>(count_kp, nrBrief/32 + i) = desc;
        }
        count_kp++;
    }
}


void ORBFeatures::convertDesc(Mat &descriptor, Mat &sem_descriptor, Mat semantic_img){

    int n, cols, idxSemDescCol, idxSemBitOffset;

    sem_descriptor = Mat::zeros(maxFeatures,nrBrief/32 + nSemrBrief/6, CV_32SC1);

    for(n = 0; n < descriptor.rows; n++){
        idxSemDescCol = -1;
        // cout << "linha: " << n << endl;
        for(cols = 0; cols < descriptor.cols; cols++, idxSemBitOffset++){
            // cout << "coluna: " << n;
            // cout << (int)semantic_img.at<uchar>(n,cols*3 + 2) << endl;
            if(!(cols % 4)){
                idxSemDescCol++;
                sem_descriptor.at<int32_t>(n,idxSemDescCol) = 0;
    
                if(!(cols % 8))
                    idxSemBitOffset = 0;
            }
            
            sem_descriptor.at<int32_t>(n,idxSemDescCol) |= descriptor.at<uchar>(n, cols) << (7-idxSemBitOffset)*8;
            // sem_descriptor.at<int32_t>(n,idxSemDescCol + nrBrief/32)
        }    
    }
}


void ORBFeatures::matchDesc(Mat &descriptor1, Mat &descriptor2, vector<DMatch> &matches){

    // const int d_max = 40;

    for (int i1 = 0; i1 < maxFeatures; ++i1) {
        cv::DMatch m{i1, 0, 0, 256};

        for (int i2 = 0; i2 < maxFeatures; ++i2) {

            int distance = 0;

            for (int k = 0; k < 8; k++) {
                distance += _mm_popcnt_u32((uint32_t)(descriptor1.at<int32_t>(i1,k) ^ descriptor2.at<int32_t>(i2,k)));
            }

            for (int j = nrBrief/32; j < nrBrief/32 + nSemrBrief/6; j++) {
                for (int k = 0; k < 6; k++) {
                    // cout << ((int)((descriptor1.at<int32_t>(i1,j) >> k*6 ^ descriptor1.at<int32_t>(i2,j) >> k*6) & 63) ? 8 : 0) << endl;
                    distance += ((descriptor1.at<int32_t>(i1,j) >> k*6 ^ descriptor2.at<int32_t>(i2,j) >> k*6) & 63) ? 6 : 0;
                }
            }

            // if (distance < d_max && distance < m.distance) {
            if (distance < m.distance) {
                m.distance = distance;
                m.trainIdx = i2;
            }
        }

        // if (m.distance < d_max) {
        matches.push_back(m);
        // }
    }
}
