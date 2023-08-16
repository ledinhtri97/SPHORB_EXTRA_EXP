/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}
*/

#ifndef _DETECTOR_H
#define _DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

typedef Point xy;
typedef unsigned char detectbyte;

// K 表示 patternSize 的一半，用于判断是否特征显著的点过半
// N = 3*K+1 ，用于 patternSize 的绕一圈的列表循环
const int patternSize = 18;
const int K = patternSize/2, N = 3*K+1;

void makeOffsets(int pixel[], int xstride, int pixel_size);

void sfast(Mat img, Mat imgMask, Mat geoMask, int threshold, bool nonmax_suppression,
           const int partIndex, vector<KeyPoint>& keypoints);

//xy* sfast_corner_detect(const detectbyte* im, const detectbyte* mask, int xsize, int xstride, int ysize, int barrier, int* num);
xy* sfast_corner_detect(const detectbyte* im, const detectbyte* im_custom_mask, const detectbyte* mask, int xsize, int xstride, int ysize, int barrier, int* num);

int sfast_corner_score(const detectbyte* im, const int pixel[], int threshold);

int* sfastScore(const unsigned char* i, int stride, xy* corners, int num_corners, int b);

void sfastNonmaxSuppression(const xy* corners, const int* scores, int num_corners, vector<KeyPoint>& kps, int partIndex);

#endif