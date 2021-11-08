#pragma once

//ORB
const int nfeatures = 2000;
const double matches_lower_bound = 50.0;
const int nrBrief = 256;
const int nSemrBrief = 24;
const int patch_size = 31;
const int half_patch_size = 15;

const float semError = 2;


//RANSAC
#define _OFFSET_IN_VID 30
#define _FRAMES_JUMPS 3

const int _N_RANSAC = 55;
const int _L_MAX = 6;
const float _RANSAC_MIN_PCTG = 0.85;

//NORMAL STUFF
const float factorPI = CV_PI/180.0;
