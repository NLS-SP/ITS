#include "DBoW3/DBoW3.h"
#include<Eigen/Core>
#include<Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#define MATCHES_DIST_THRESHOLD 25
#define PI 3.1415926
#define SIGMA 100
#define MarginRight 330
#define MarginLeft  30
using namespace cv;
using namespace std;
using namespace Eigen;
void similarityCal(vector<Mat> descriptors) {
	//read the images and database
	DBoW3::Vocabulary vocab("./vocabulary_dataset_1Floor.yml.gz");
	if (vocab.empty())
	{
		cerr << "Vocabulary does not exist." << endl;
	}
	DBoW3::Database db(vocab, false, 0);
	for (size_t i = 0;i < descriptors.size();i++) {
		db.add(descriptors[i]);
	}
	for (int i = 0;i < descriptors.size();i++) {
		DBoW3::QueryResults ret;
		db.query(descriptors[i], ret, 4);
		cout << "searching for image " << i << "  returns " << ret << endl << endl;
	}
}
Point get_vp(vector<Point> vec) {
	double x = 0;double y = 0;
	double x_mean = 0;double y_mean = 0;
	double var_x_total = 0;double var_y_total = 0;
	double var_x;double var_y;
	vector<double> vec_x; vector<double> vec_y;
	for (size_t i = 0;i < vec.size();i++) {
		x += vec[i].x;
		y += vec[i].y;
		vec_x.push_back(vec[i].x);
		vec_y.push_back(vec[i].y);
	}
	x_mean = x / vec.size();
	y_mean = y / vec.size();
	std::sort(vec_x.begin(),vec_x.end());
	std::sort(vec_y.begin(), vec_y.end());
	for (size_t i = 0;i < vec.size();i++) {
		var_x_total += (vec[i].x - x_mean)*(vec[i].x - x_mean);
		var_y_total += (vec[i].y - y_mean)*(vec[i].y - y_mean);
	}
	var_x = var_x_total / vec.size();
	var_y = var_y_total / vec.size();
	cout << "var:" << var_x << "     " << var_y << endl;
	Point p;
	if (var_x < SIGMA && var_y < SIGMA)
	{
		p.x = x_mean;p.y = y_mean;
	}
	else
	{
		p.x = vec_x[vec.size() / 2];
		p.y = vec_y[vec.size() / 2];
	}
	return p;
}
void find_feature_matches(const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1,
	std::vector<KeyPoint>& keypoints_2,
	std::vector< DMatch >& matches)
{

	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	vector<DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);

	double min_dist = 10000, max_dist = 0;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (match[i].distance <= max(2 * min_dist, 30.0))
		{
			matches.push_back(match[i]);
		}
	}
}
void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
	std::vector<KeyPoint> keypoints_2,
	std::vector< DMatch > matches,
	Mat& R, Mat& t)
{
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

	vector<Point2f> points1;
	vector<Point2f> points2;

	for (int i = 0; i < (int)matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	Mat fundamental_matrix;
	fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
	cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

	Point2d principal_point(325.1, 249.7);
	double focal_length = 521;
	Mat essential_matrix;
	essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
	cout << "essential_matrix is " << endl << essential_matrix << endl;

	Mat homography_matrix;
	homography_matrix = findHomography(points1, points2, RANSAC, 3);
	cout << "homography_matrix is " << endl << homography_matrix << endl;

	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	cout << "R is " << endl << R << endl;
	cout << "t is " << endl << t << endl;

}
Point vanish_point_detection(Mat & image, Mat & cdst) {
	Mat dst;
	vector<Vec2f> left_lines;
	vector<Point> Intersection;
	vector<Vec2f> right_lines;
	//hough_line_detection
	Canny(image, dst, 30, 70, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		if (5 * PI / 180 < theta && theta < 45 * PI / 180) {
			left_lines.push_back(lines[i]);
		}
		else if (140 * PI / 180 < theta && theta < 170 * PI / 180) {
			right_lines.push_back(lines[i]);
		}
	}
	size_t i = 0, j = 0;
	long sum_x = 0;
	long sum_y = 0;
	double x = 0;
	double y = 0;
	//HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
	for (i = 0; i < left_lines.size(); ++i) {
		for (j = 0; j < right_lines.size(); ++j) {
			float rho_l = left_lines[i][0], theta_l = left_lines[i][1];
			float rho_r = right_lines[j][0], theta_r = right_lines[j][1];
			double denom = (sin(theta_l) * cos(theta_r) - cos(theta_l) * sin(theta_r));
			x = (rho_r * sin(theta_l) - rho_l * sin(theta_r)) / denom;
			y = (rho_l * cos(theta_r) - rho_r * cos(theta_l)) / denom;

			Point pt(x, y);
			Intersection.push_back(pt);
			circle(image, pt, 5, Scalar(0, 0, 0), 5);
			cout << "(" << pt.x << "," << pt.y << ")" << endl;
		}
	}
	return get_vp(Intersection);
}
void LBD_draw() {
	//Create LBD class pointer
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	Mat img_1 = imread("C:\\Users\\Lucky\\Desktop\\Journal Paper\\dataset\\experments\\0829����ʵ��\\23.jpg");
	cv::Mat mask = Mat::ones(img_1.size(), CV_8UC1);
	vector<line_descriptor::KeyLine>keylines1;
	lbd->detect(img_1, keylines1, mask);
	Mat line_descriptors1;
	lbd->compute(img_1, keylines1, line_descriptors1);
	Mat outputImage;
	std::cout << keylines1.size() << endl;
	std::cout << line_descriptors1.rows << endl;
	std::cout << line_descriptors1.cols << endl;
	std::cout << line_descriptors1.size() << endl;
	//line_descriptor::drawKeylines(img_1, keylines1, outputImage);
	outputImage = img_1.clone();
	for (size_t i = 0; i < keylines1.size(); i++)
	{
		/* decide lines' color  */
		Scalar lineColor;
			int R = (rand() % (int)(255 + 1));
			int G = (rand() % (int)(255 + 1));
			int B = (rand() % (int)(255 + 1));
			lineColor = Scalar(R, G, B);		
		/* get line */
		line_descriptor::KeyLine k = keylines1[i];
		/* draw line */
		//if (k.lineLength > 0 && k.angle>=-0.1 && k.angle<=0.1) {
		//	std::cout << k.lineLength << "      " << k.angle << endl;
			cv::line(outputImage, Point2f(k.startPointX, k.startPointY), Point2f(k.endPointX, k.endPointY), lineColor, 3);
		//}
	}
	cv::imshow("result", outputImage);
	cv::waitKey(600000);
}
void LBD_and_VP_draw() {
	//Create LBD class pointer
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	Mat img_1 = imread("/home/lucky/FOEY/2018-10-30-14-31-21/IMG/35882917127567.jpg");
	Mat img_2;
	//cvtColor(img_1, img_2, CV_RGB2GRAY);
	//imwrite("C:\\Users\\Lucky\\Desktop\\Journal Paper\\dataset\\experments\\0829����ʵ��\\14_gray.jpg",img_2);
	cv::Mat mask = Mat::ones(img_1.size(), CV_8UC1);
	vector<line_descriptor::KeyLine>keylines1;
	lbd->detect(img_1, keylines1, mask);
	Mat line_descriptors1;
	lbd->compute(img_1, keylines1, line_descriptors1);
	Mat outputImage;
	std::cout << keylines1.size() << endl;
	std::cout << line_descriptors1.rows << endl;
	std::cout << line_descriptors1.cols << endl;
	std::cout << line_descriptors1.size() << endl;
	//line_descriptor::drawKeylines(img_1, keylines1, outputImage);
	outputImage = img_1.clone();
	//vanishing point part***************
	Mat dst, cdst;
	vector<Vec2f> left_lines;
	vector<Point> Intersection;
	vector<Vec2f> right_lines;
	//hough_line_detection
	Canny(img_1, dst, 30, 70, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		if (10 * PI / 180 < theta && theta < 80 * PI / 180) {
			left_lines.push_back(lines[i]);
		}
		else if (100 * PI / 180 < theta && theta < 170 * PI / 180) {
			right_lines.push_back(lines[i]);
		}
	}
	size_t i = 0, j = 0;
	long sum_x = 0;
	long sum_y = 0;
	double x = 0;
	double y = 0;
	Canny(img_1, dst, 30, 70, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
	//// draw lines
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		if (10 * PI / 180 < theta && theta < 80 * PI / 180) {
			line(img_1, pt1, pt2, Scalar(0, 255, 0), 3, CV_AA);
		}
		else if (100 * PI / 180 < theta && theta < 170 * PI / 180) {
			line(img_1, pt1, pt2, Scalar(255, 0, 0), 3, CV_AA);
		}
		else {
			line(img_1, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
	}
	for (i = 0; i < left_lines.size(); ++i) {
		for (j = 0; j < right_lines.size(); ++j) {
			float rho_l = left_lines[i][0], theta_l = left_lines[i][1];
			float rho_r = right_lines[j][0], theta_r = right_lines[j][1];
			double denom = (sin(theta_l) * cos(theta_r) - cos(theta_l) * sin(theta_r));
			x = (rho_r * sin(theta_l) - rho_l * sin(theta_r)) / denom;
			y = (rho_l * cos(theta_r) - rho_r * cos(theta_l)) / denom;

			Point pt(x, y);
			Intersection.push_back(pt);
			circle(img_1, pt, 5, Scalar(0, 0, 0), 5);
		}
	}
	//***********************************
	for (size_t i = 0; i < keylines1.size(); i++)
	{
		/* decide lines' color  */
		Scalar lineColor;
		int R = (rand() % (int)(255 + 1));
		int G = (rand() % (int)(255 + 1));
		int B = (rand() % (int)(255 + 1));
		lineColor = Scalar(R, G, B);
		/* get line */
		line_descriptor::KeyLine k = keylines1[i];
		/* draw line */
		//if (k.lineLength > 0 && k.angle>=-0.1 && k.angle<=0.1) {
		//	std::cout << k.lineLength << "      " << k.angle << endl;
		cv::line(img_1, Point2f(k.startPointX, k.startPointY), Point2f(k.endPointX, k.endPointY), lineColor, 3);
		//}
	}
	cv::imshow("result", img_1);
	imwrite("/home/lucky/dataset/ITS/experments/14_gray_result.jpg",img_1);
	cv::waitKey(600000);
}
void LBD_match() {
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	//Ptr<line_descriptor::LSDDetector> bd = line_descriptor::LSDDetector::createLSDDetector();
	Mat img_1 = imread("/home/lucky/dataset/ITS/experments/1.jpg");
	Mat img_2 = imread("/home/lucky/dataset/ITS/experments/2.jpg");
	cv::Mat mask = Mat::ones(img_1.size(), CV_8UC1);
	vector<line_descriptor::KeyLine>keylines1, keylines2;
	lbd->detect(img_1, keylines1, mask);
	lbd->detect(img_2, keylines2, mask);
	Mat line_descriptors1;
	Mat line_descriptors2;
	lbd->compute(img_1, keylines1, line_descriptors1);
	lbd->compute(img_2, keylines2, line_descriptors2);

	std::vector<line_descriptor::KeyLine> lbd_octave1, lbd_octave2;
	Mat left_lbd, right_lbd;
	for (int i = 0; i < (int)keylines1.size(); i++)
	{
		if (keylines1[i].octave == 0)
		{
			lbd_octave1.push_back(keylines1[i]);
			left_lbd.push_back(line_descriptors1.row(i));
		}
	}

	for (int j = 0; j < (int)keylines2.size(); j++)
	{
		if (keylines2[j].octave == 0)
		{
			lbd_octave2.push_back(keylines2[j]);
			right_lbd.push_back(line_descriptors2.row(j));
		}
	}

	//Ptr<line_descriptor::BinaryDescriptorMatcher> lbd_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
	line_descriptor::BinaryDescriptorMatcher lbd_matcher;
	vector<DMatch> matches;
	lbd_matcher.match(line_descriptors1, line_descriptors2, matches);
	std::vector<DMatch> good_matches;
	for (int i = 0; i < (int)matches.size(); i++)
	{
		if (matches[i].distance < MATCHES_DIST_THRESHOLD)
			good_matches.push_back(matches[i]);
	}
	std::cout << "The size of line descriptors in image is:" << endl;
	std::cout << line_descriptors1.size() << endl;
	std::cout << "The number of good matches is:" << endl;
	std::cout << good_matches.size() << endl;
	Mat outputImage;
	///--------------------------------------------------------------------------------------------------///////////////////////
	if (img_1.type() != img_2.type())
	{
		std::cout << "Input images have different types" << std::endl;
	}

	//check how many rows are necessary for output matrix 
	int totalRows = img_1.rows >= img_2.rows ? img_1.rows : img_2.rows;
	//initialize output matrix 
	outputImage = Mat::zeros(totalRows, img_1.cols + img_2.cols, img_1.type());
	Scalar singleLineColorRGB;
	int R = (rand() % (int)(255 + 1));
	int G = (rand() % (int)(255 + 1));
	int B = (rand() % (int)(255 + 1));
	singleLineColorRGB = Scalar(R, G, B);
	// copy input images to output images 
	Mat roi_left(outputImage, Rect(0, 0, img_1.cols, img_1.rows));
	Mat roi_right(outputImage, Rect(img_1.cols, 0, img_2.cols, img_2.rows));
	img_1.copyTo(roi_left);
	img_2.copyTo(roi_right);
	// get columns offset 
	int offset = img_1.cols;
	std::cout << "good matches:" << good_matches.size() << endl;
	for (size_t counter = 0; counter < good_matches.size(); counter++)
	{

		DMatch dm = good_matches[counter];
		line_descriptor::KeyLine left = keylines1[dm.queryIdx];
		line_descriptor::KeyLine right = keylines2[dm.trainIdx];
		std::cout << "left.angle:" << left.angle <<"      "<<"right.angle:"<<right.angle<< endl;

		Scalar matchColorRGB;

		int R = (rand() % (int)(255 + 1));
		int G = (rand() % (int)(255 + 1));
		int B = (rand() % (int)(255 + 1));
		matchColorRGB = Scalar(R, G, B);
		singleLineColorRGB = matchColorRGB;
		if ((left.angle >= -PI/8&& left.angle <= PI/8) || (left.angle>=15*PI/16) ||(left.angle<=-15*PI/16)) 
		{
			if (left.startPointX <= MarginRight && left.startPointX >= MarginLeft && left.endPointX <= MarginRight && left.endPointX >= MarginLeft) {
				std::cout << "coordinate: " << "(" << left.startPointX << "," << left.startPointY << ")";
				std::cout << "coordinate: " << "(" << left.endPointX << "," << left.endPointY << ")" << endl;
				line(outputImage, Point2f(left.sPointInOctaveX, left.sPointInOctaveY), Point2f(left.ePointInOctaveX, left.ePointInOctaveY), singleLineColorRGB, 2);
				line(outputImage, Point2f(right.sPointInOctaveX + offset, right.sPointInOctaveY), Point2f(right.ePointInOctaveX + offset, right.ePointInOctaveY), singleLineColorRGB,
					2);
				//link correspondent lines 
				line(outputImage, Point2f(left.sPointInOctaveX, left.sPointInOctaveY), Point2f(right.sPointInOctaveX + offset, right.sPointInOctaveY), matchColorRGB, 1);
			}
		}
	}

	//-----------------------------------------------------------------------------------------------------///////////////////////
	//Mat img_matches;
	//drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	imshow("result", outputImage);
	waitKey(600000);
}
void DBoW3_points() {
	//-------------------2018/0312----------------------------------------------------------------------//
	// read the image 
	cout << "reading images... " << endl;
	vector<Mat> images;
	for (int i = 0; i<10; i++)
	{
	string path = "/home/lucky/dataset/ITS/experments/" + to_string(i + 1) + ".png";
	images.push_back(imread(path));
	}
	for (Mat& image : images)
	{
		cout << image.rows << "   " << image.cols << endl;
	}
	cout << "detecting ORB features ... " << endl;
	Ptr< Feature2D > detector = ORB::create();
	vector<Mat> descriptors;
	for (Mat& image : images)
	{
	vector<KeyPoint> keypoints;
	Mat descriptor;
	detector->detectAndCompute(image, Mat(), keypoints, descriptor);
	cout << " descriptor's size is : " << descriptor.size() << endl;
	cout << descriptor.rows << endl;
	cout << descriptor.cols << endl;
	descriptors.push_back(descriptor);
	}
	cout << "vector<mat>'s descriptors's size is : " << descriptors.size() << endl;
	// create vocabulary
	cout << "creating vocabulary ... " << endl;
	DBoW3::Vocabulary vocab;
	//  DBoW3::Vocabulary vocab;
	vocab.create(descriptors);
	cout << "vocabulary info: " << vocab << endl;
	vocab.save("vocabulary.yml.gz");
	cout << "done" << endl;
	
}
void DBow3_lines() {
	// read the image 
	cout << "reading images... " << endl;
	vector<Mat> images;
	for (int i = 1; i<=906; i++)
	{
	string path = "/home/lucky/dataset/ITS/experments/" + to_string(i) + ".png";
	Mat image = imread(path);
	if (!image.data)
		continue;
	else
	    images.push_back(image);
	}
	// detect Line features
	cout << "detecting line features ... " << endl;
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	vector<Mat> line_descriptors;
	for (Mat& image : images)
	{
	vector<line_descriptor::KeyLine> keylines;
	Mat descriptor;
	lbd->detect(image, keylines);
	lbd->compute(image, keylines, descriptor);
	//if(descriptor.rows !=0 && descriptor.cols !=0)
	line_descriptors.push_back(descriptor);

	}
	// create vocabulary
	cout << "creating vocabulary ... " << endl;
	DBoW3::Vocabulary vocab;
	//  DBoW3::Vocabulary vocab;
	vocab.create(line_descriptors);
	cout << "vocabulary info: " << vocab << endl;
	vocab.save("vocabulary_dataset_1Floor.yml.gz");
	cout << "done" << endl;
	
}
void DBow3_detect() {
	cout << "reading images... " << endl;
	vector<Mat> images;
	for (int i = 0; i<905; i++)
	{
		string path = "/home/lucky/dataset/ITS/experments/" + to_string(i + 1) + ".png";
		Mat image = imread(path);
		if (!image.data)
			continue;
		else
		images.push_back(image);
	}
	// detect Line features
	cout << "detecting line features ... " << endl;
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	vector<Mat> line_descriptors;
	for (Mat& image : images)
	{
		vector<line_descriptor::KeyLine> keylines;
		Mat descriptor;
		lbd->detect(image, keylines);
		lbd->compute(image, keylines, descriptor);
		//if(descriptor.rows !=0 && descriptor.cols !=0)
		line_descriptors.push_back(descriptor);

	}
	similarityCal(line_descriptors);
	//------------*********************************************************************************************************-----
}
void PoseEstimation_lines() {
	Mat img_1 = imread("/home/lucky/dataset/ITS/experments/13_gray.jpg");
	Mat img_2 = imread("/home/lucky/dataset/ITS/experments/14_gray.jpg");
	Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	cv::Mat mask = Mat::ones(img_1.size(), CV_8UC1);
	vector<line_descriptor::KeyLine>keylines1, keylines2;
	double start1 = double(getTickCount());
	lbd->detect(img_1, keylines1, mask);
	lbd->detect(img_2, keylines2, mask);
	Mat line_descriptors1;
	Mat line_descriptors2;
	lbd->compute(img_1, keylines1, line_descriptors1);
	lbd->compute(img_2, keylines2, line_descriptors2);
	std::vector<line_descriptor::KeyLine> lbd_octave1, lbd_octave2;
	
	Mat left_lbd, right_lbd;
	for (size_t i = 0; i < keylines1.size(); i++)
	{
		if (keylines1[i].octave == 0)
		{
			lbd_octave1.push_back(keylines1[i]);
			left_lbd.push_back(line_descriptors1.row(i));
		}
	}

	for (size_t j = 0; j < keylines2.size(); j++)
	{
		if (keylines2[j].octave == 0)
		{
			lbd_octave2.push_back(keylines2[j]);
			right_lbd.push_back(line_descriptors2.row(j));
		}
	}
	
	line_descriptor::BinaryDescriptorMatcher lbd_matcher;
	vector<DMatch> matches;
	lbd_matcher.match(line_descriptors1, line_descriptors2, matches);
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < MATCHES_DIST_THRESHOLD)
			good_matches.push_back(matches[i]);
	}
	vector<line_descriptor::KeyLine> HorizenLinesLeft, HorizenLinesRight;
	vector<line_descriptor::KeyLine> VerticalLinesLeft, VerticalLinesRight;
	
	for (size_t counter = 0; counter < good_matches.size(); counter++)
	{
		DMatch dm = good_matches[counter];
		line_descriptor::KeyLine left = keylines1[dm.queryIdx];
		line_descriptor::KeyLine right = keylines2[dm.trainIdx];
		//cout << "left.angle:" << left.angle << "      " << "right.angle:" << right.angle << endl;
		if ((left.angle >= -PI / 8 && left.angle <= PI / 8) || (left.angle >= 15 * PI / 16) || (left.angle <= -15 * PI / 16)) {
			if (left.startPointX <= MarginRight && left.startPointX >= MarginLeft && left.endPointX <= MarginRight && left.endPointX >= MarginLeft)
			{
				HorizenLinesLeft.push_back(left);
				HorizenLinesRight.push_back(right);
			}
		}
		if ((left.angle >= -1.58&& left.angle <= -1.56) || (left.angle >= 1.56 && left.angle <= 1.58)) {
			VerticalLinesLeft.push_back(left);
			VerticalLinesLeft.push_back(right);
		}

	}
	
	float roll = HorizenLinesLeft[0].angle - HorizenLinesRight[0].angle;
	cout << "sdsds" << endl;
	std::cout << HorizenLinesLeft.size() << endl;
	std::cout << "roll_1:" << HorizenLinesLeft[0].angle << "   " << "roll_2:" << HorizenLinesRight[0].angle << endl;
	std::cout << "Roll: " << roll*180/PI << endl;
	double dur1 = (double(getTickCount()) - start1) * 1000 / getTickFrequency();
	cout << "roll took " << dur1 << "ms" << endl;
	//************************************************************************************************
	Mat cdst,cdst2;
	double start = double(getTickCount());
	Point current_point=vanish_point_detection(img_1,cdst);
	Point current_point2 = vanish_point_detection(img_2, cdst2);
	//imshow("image1",img_1);
	//imshow("image2", img_2);
	waitKey(0);
	std::cout << "current_point:" << current_point.x << "   " << current_point.y << endl;
	std::cout << "current_point2:" << current_point2.x << "   " << current_point2.y << endl;

	float focalLenX = 487.5180;
	float focalLenY = 486.9239;
	float principleX = 174.4413;
	float principleY = 325.5396;
	imshow("result",img_1);
	imshow("result2", img_2);
	double yaw_1 = asin((current_point.x - principleX) / focalLenX);
	double yaw_2 = asin((current_point2.x - principleX) / focalLenX);
	double pitch_1 = asin((current_point.y - principleY) / (focalLenY*cos(yaw_1)));
	double pitch_2 = asin((current_point2.y - principleY) / (focalLenY*cos(yaw_2)));
	std::cout << "Yaw: " << (yaw_1-yaw_2) * 180 / PI << endl;
	std::cout << "Pitch: " << (pitch_1-pitch_2) * 180 / PI << endl;
	double dur = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	cout << "vanishing point took " << dur << "ms" << endl;
	waitKey(0);

}
void PoseEstimation_points() {
	double start = double(getTickCount());
	Mat img_1 = imread("/home/lucky/dataset/ITS/experments/11_gray.jpg");
	Mat img_2 = imread("/home/lucky/dataset/ITS/experments/12_gray.jpg");
	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cv::Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
	//cout << R << endl;
	Eigen::Matrix3d R_;
	R_ <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
	
	Eigen::Vector3d euler_angles = R_.eulerAngles(2, 1, 0);
	cout << euler_angles.transpose() << endl;
	double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	cout << "It took " << duration_ms << " ms." << endl;
}
int main()
{
	//LBD_draw();
	LBD_and_VP_draw();
	//LBD_match();
	//PoseEstimation_lines();
	//DBow3_lines();
	//DBoW3_points(); 
	//DBow3_detect();
    //PoseEstimation_points();
	//Mat img_1 = imread("C:\\Users\\Lucky\\Desktop\\Journal Paper\\dataset\\experments\\0829����ʵ��\\22.jpg");
	//Mat cdst;
	//Point d = vanish_point_detection(img_1, cdst);
	//imshow("vp",img_1);
	//waitKey(0);
	
return 0;

}
