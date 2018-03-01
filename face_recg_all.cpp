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

static CascadeClassifier face_cascade("./haarcascade_frontalface_default.xml");		//load Cascade
static CascadeClassifier eyes_cascade("./haarcascade_eye_tree_eyeglasses.xml");
static CascadeClassifier nose_cascade("./haarcascade_mcs_nose.xml");
static CascadeClassifier mouth_cascade("./haarcascade_mcs_mouth.xml");

static string names[] = {"Hobo", "Liss", "Liu Tao", "Luo Chuanyou", "Ma Wen",
				 "Ren Junze", "Tian Heming", "Tian Minjie", "Zhao Cancan", 
				 "Zhao Cong", "Zheng Fanglei"};

static void detect_features(Mat& img, Rect& face);
int main()
{
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
	int predict_label = 0;
	double predict_confidence = 0.0;

	while(1) {
		//image = imread("0_02.jpg", CV_LOAD_IMAGE_COLOR);
		bool ret = cap.read(image);
		if (ret == false) {
			cout << "Can't get data from cap" << endl;
			break;
		}

		face_cascade.detectMultiScale(image, recs,1.2,6,0,Size(50,50));    //detect the face
		for (int i = 0; i < recs.size();i++) {
			rectangle(image, recs[i], Scalar(0, 0, 255));
			x = recs[i].x + recs[i].width / 2;
			y = recs[i].y + recs[i].height / 2;

			Mat roi = image(recs[i]);
			cvtColor(roi, gray, CV_BGR2GRAY);
			resize(gray, test, Size(200, 200));    //the training sample is 200x200, resize the pic here.

			fc->predict(test, predict_label, predict_confidence);                                                       
			if (predict_confidence < 800) {                                                                             
				putText(image, names[predict_label], Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
			} else {
				putText(image, "Hi, New friends!", Point(recs[i].x, recs[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);  
			}
			detect_features(image, recs[i]);
		}

		imshow("Detected", image);
		if (waitKey(10) == 27)
			break;
	}

	return 0;
}

static void detect_features(Mat& img, Rect& face)
{
	bool is_full_detection = true;
//	rectangle(img, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
//			Scalar(255, 0, 0), 1, 4);

	// Eyes, nose and mouth will be detected inside the face (region of interest)
	Mat ROI = img(Rect(face.x, face.y, face.width, face.height));

	if (!eyes_cascade.empty())
	{
		//detect eyes
		vector<Rect_<int> > eyes;
		eyes_cascade.detectMultiScale(ROI, eyes, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		// Mark points corresponding to the centre of the eyes
		for(unsigned int j = 0; j < eyes.size(); ++j)
		{
			Rect e = eyes[j];
			circle(ROI, Point(e.x+e.width/2, e.y+e.height/2), 3, Scalar(0, 255, 0), -1, 8);
			/* rectangle(ROI, Point(e.x, e.y), Point(e.x+e.width, e.y+e.height),
			Scalar(0, 255, 0), 1, 4); */
		}
	}

	// Detect nose if classifier provided by the user
	double nose_center_height = 0.0;
	if(!nose_cascade.empty())
	{
		vector<Rect_<int> > nose;
		nose_cascade.detectMultiScale(ROI, nose, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		// Mark points corresponding to the centre (tip) of the nose
		for(unsigned int j = 0; j < nose.size(); ++j)
		{
			Rect n = nose[j];
			circle(ROI, Point(n.x+n.width/2, n.y+n.height/2), 3, Scalar(0, 255, 0), -1, 8);
			nose_center_height = (n.y + n.height/2);
		}
	}

	// Detect mouth if classifier provided by the user
	double mouth_center_height = 0.0;
	if(!mouth_cascade.empty())
	{
		vector<Rect_<int> > mouth;
		mouth_cascade.detectMultiScale(ROI, mouth, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		for(unsigned int j = 0; j < mouth.size(); ++j)
		{
			Rect m = mouth[j];
			mouth_center_height = (m.y + m.height/2);

			// The mouth should lie below the nose
			if( (is_full_detection) && (mouth_center_height > nose_center_height) )
			{
				rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
			}
			else if( (is_full_detection) && (mouth_center_height <= nose_center_height) )
				continue;
			else
			rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
		}
	}
}