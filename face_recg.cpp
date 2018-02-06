#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <sstream>
#include <fstream>
//#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace face;

//#define READ_CSV

#ifdef READ_CSV
void read_csv(string& fileName,vector<Mat>& images,vector<int>& labels,char separator = ';')
{
	ifstream file(fileName.c_str(), ifstream::in);
	if (!file) {
		cout << "No valid input file was given, please check the given filename." << endl;
		return;
	}

	string line, path, label;

	while (getline(file, line)) {
		stringstream lines(line);
		getline(lines, path, separator);
		getline(lines, label);

		if (!path.empty() && !label.empty()) {
			images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			labels.push_back(atoi(label.c_str()));
		}
	}
}
#endif

int main()
{
	CascadeClassifier cas("./haarcascade_frontalface_default.xml");		//load Cascade
	Ptr<FaceRecognizer> fc = createFisherFaceRecognizer();

#ifdef READ_CSV
	string csvPath = "./train_data.txt";

	vector<Mat> images;    
	vector<int> labels;
	read_csv(csvPath, images, labels);

	fc->train(images, labels);              //train
	fc->save("./Face_Model.xml");         //save data after trained.
#else
	fc->load("Face_Model.xml");
#endif

	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	Mat image;
	vector<Rect> recs;
	Mat test(200, 200, CV_8UC1);
	Mat gray;
	int x = 0, y = 0;

	while(1) {
		//image = imread("0_02.jpg", CV_LOAD_IMAGE_COLOR);
		bool ret = cap.read(image);
		if (ret == false) {
			cout << "Can't get data from cap" << endl;
			break;
		}

		cas.detectMultiScale(image, recs,1.2,6,0,Size(50,50));    //detect the face
		for (int i = 0; i < recs.size();i++) {
			rectangle(image, recs[i], Scalar(0, 0, 255));
			x = recs[i].x + recs[i].width / 2;
			y = recs[i].y + recs[i].height / 2;

			Mat roi = image(recs[i]);
			cvtColor(roi, gray, CV_BGR2GRAY);
			resize(gray, test, Size(200, 200));    //since the  training sample is 200x200, so need resize here.

			int result = fc->predict(test);
			switch (result) {
				case 0:
					putText(image, "Liu Tao", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
				break;

				case 1:
					putText(image, "Ren Junze", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
				break;

				case 2:
					putText(image, "Tian Heming", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
				break;

				case 3:
					putText(image, "Tian Minjie", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
				break;

				default:
					putText(image, "Hi, New Friend", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
				break;
			}
		}

		imshow("Detected", image);
		if (waitKey(10) == 27)
			break;
	}

	return 0;
}
