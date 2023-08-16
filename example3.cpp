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

#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "utility.h"
#include "nlohmann_json.hpp"

using namespace std;
using namespace cv;
using json = nlohmann::json;

int main(int argc, char * argv[])
{
	json config;
	try {
		std::ifstream f("config.json");
		config = json::parse(f);
	}
	catch (...) {
		cout << "Error to parse config.json!" << endl;
		return 1;
	}
	
	string maskPath = config["main"]["mask"];
	string fimage = config["main"]["1st-image"];
	string simage = config["main"]["2nd-image"];
	string omatches = config["main"]["output-image"];
	string ojson = config["main"]["output-json"];

	if (fimage.empty() || simage.empty()) {
		cout<<"Input images are empty! "<<endl;
	}
	cout << "Input 1st image: " << fimage << endl;
	cout << "Input 2nd image: " << simage << endl;

	int nfeatures = config["sphorb"]["nfeatures"];
	int nlevels = config["sphorb"]["nlevels"];
	int barrier = config["sphorb"]["barrier"];
	float ratio = config["sphorb"]["ratio-match"];

	std::vector<float> offsets = {
		config["frame"]["offset-top"], 
		config["frame"]["offset-left"], 
		config["frame"]["offset-right"], 
		config["frame"]["offset-bottom"]
	};
	
	Mat img1 = imread(fimage);
	Mat img2 = imread(simage);
	resize(img1, img1, Size(1280, 640), 0, 0, INTER_AREA);
	resize(img2, img2, Size(1280, 640), 0, 0, INTER_AREA);

	Mat imgMask = Mat();
	if (!maskPath.empty()) {
		imgMask = imread(maskPath, IMREAD_GRAYSCALE);
	}

	Mat descriptors1;
	Mat descriptors2;
	vector<KeyPoint> kPoint1;
	vector<KeyPoint> kPoint2;

	SPHORB sorb(nfeatures, nlevels, barrier);
	sorb(img1, imgMask, kPoint1, descriptors1);
	sorb(img2, imgMask, kPoint2, descriptors2); 

	cout<<"Keypoint1: "<<kPoint1.size()<<", Keypoint2: "<<kPoint2.size()<<endl;

	BFMatcher matcher(NORM_HAMMING, false);
	Matches matches;
	
	vector<Matches> dupMatches;
	matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
	ratioTest(dupMatches, ratio, matches);
	cout<<"Matches: "<<matches.size()<<endl;
	
	vector<int> roiImage1 = {
		static_cast<int>(offsets[1] * img1.cols),
		static_cast<int>(offsets[0] * img1.rows),
		static_cast<int>((1-offsets[2]) * img1.cols),
		static_cast<int>((1-offsets[3]) * img1.rows)
	};
	vector<int> roiImage2 = {
		static_cast<int>(offsets[1] * img2.cols),
		static_cast<int>(offsets[0] * img2.rows),
		static_cast<int>((1-offsets[2]) * img2.cols),
		static_cast<int>((1-offsets[3]) * img2.rows)
	};
	cout << "Roi 1: (" << roiImage1[0] << ", " << roiImage1[1] << ") (" << roiImage1[2] << ", " << roiImage1[3] << ")" << endl;
	cout << "Roi 2: (" << roiImage2[0] << ", " << roiImage2[1] << ") (" << roiImage2[2] << ", " << roiImage2[3] << ")" << endl;

	Rect rect1(roiImage1[0], roiImage1[1], roiImage1[2] - roiImage1[0], roiImage1[3] - roiImage1[1]);
	Rect rect2(roiImage2[0], roiImage2[1], roiImage2[2] - roiImage2[0], roiImage2[3] - roiImage2[1]);
	rectangle(img1, rect1, Scalar(0, 0, 255), 2);
	rectangle(img2, rect2, Scalar(0, 0, 255), 2);
	for (auto m = matches.begin(); m != matches.end(); ) {
		int i1 = m->queryIdx;
		int i2 = m->trainIdx;
		const KeyPoint &kp1 = kPoint1[i1], &kp2 = kPoint2[i2];
		if (kp1.pt.x < roiImage1[0] || kp1.pt.y < roiImage1[1] || kp1.pt.x > roiImage1[2] || kp1.pt.y > roiImage1[3]
		|| kp2.pt.x < roiImage2[0] || kp2.pt.y < roiImage2[1] || kp2.pt.x > roiImage2[2] || kp2.pt.y > roiImage2[3]) {
			m = matches.erase(m);
		} else {
			++m;
		}
	}

	Mat imgMatches;
	::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS, true);

	if (!omatches.empty()) {
		std::cout << "output Image: " << omatches << std::endl;
		imwrite(omatches, imgMatches);
	}
	if (!ojson.empty()) {
		std::cout << "output Json: " << ojson << std::endl;

		try {
			json jsonArray;
			for( size_t m = 0; m < matches.size(); m++ )
			{
				int i1 = matches[m].queryIdx;
				int i2 = matches[m].trainIdx;
				const KeyPoint &kp1 = kPoint1[i1], &kp2 = kPoint2[i2];
				json obj;
				obj["x1"] = kp1.pt.x;
				obj["y1"] = kp1.pt.y;
				obj["x2"] = kp2.pt.x;
				obj["y2"] = kp2.pt.y;
				jsonArray.push_back(obj);
			}
			config["matches"] = jsonArray;
			std::ofstream outputFile(ojson);
			outputFile << std::setw(4) << std::setfill(' ') << config;
			outputFile.close();
		}
		catch (...) {
			cout << "Error to save:" << ojson << endl;
			return 1;
		}
	}

	return 0;
}