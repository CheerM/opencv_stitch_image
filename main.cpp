#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>  
#include <opencv2/opencv.hpp>  
#include <cv.h>
#include <highgui.h>
#include <iostream>  
#include <stdio.h>  
 
using namespace cv;
using namespace std;


int main( int argc, char** argv) {
	
	//读入图像
	Mat image1 = imread( argv[1], 1 ); 
	Mat image2 = imread( argv[2], 1 );
	imshow("image1", image1);
	imshow("image2", image2);

	//detect descriptors
	int minHessian = 400; //阈值
    	SurfFeatureDetector detector( minHessian );
    	vector<KeyPoint> kp1, kp2;
    	detector.detect(image1, kp1);
    	detector.detect(image2, kp2);
    	cout <<"the num of descriptor of image1:"<< kp1.size() << " the num of descriptor of image2:" << kp2.size()  << endl;
    	
    	//calculate descriptors
    	SurfDescriptorExtractor extractor;
    	Mat descriptor1;
    	Mat descriptor2;
    	extractor.compute(image1, kp1, descriptor1);
    	extractor.compute(image2, kp2, descriptor2);
    	cout<<"the size of vector matrix of image1"<<descriptor1.size()  
		<<"，num of feature vector："<<descriptor1.rows<<"，dimension："<<descriptor1.cols<<endl;  
	cout<<"the size of vector matrix of image1"<<descriptor2.size()  
		<<"，num of feature vector："<<descriptor2.rows<<"，dimension："<<descriptor2.cols<<endl; 

	//draw keypoints
    	Mat kp1_image;
    	Mat kp2_image;
    	drawKeypoints( image1, kp1, kp1_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    	drawKeypoints(  image2, kp2, kp2_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    	imwrite( "images/keypoint_image1.png", kp1_image );  
    	imwrite( "images/keypoint_image2.png", kp2_image );  

	//match descriptors
    	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);
	Mat img_match;
	drawMatches(image1,kp1,image2,kp2,matches, img_match, Scalar::all(-1), CV_RGB(0,255,0), Mat(), 2);  
	//imshow("match", img_match);
	imwrite( "images/first_match.png", img_match );  

	double max_dist = 0;
	double min_dist = 200;
	for (int i = 0; i < matches.size(); i ++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "number of matches" << "=" << matches.size() << endl;
	cout << "max_dist = " << max_dist << "; " << "min_dist = " << min_dist << endl;

	vector<DMatch> better_matches;
	for (int i = 0; i < matches.size(); i ++) {
		if (matches[i].distance < 0.2 * max_dist)
			better_matches.push_back(matches[i]);
	}
	Mat img_better_match;
	drawMatches(image1,kp1,image2,kp2,better_matches, img_better_match, Scalar::all(-1), CV_RGB(0,255,0), Mat(), 2);  
	
	imwrite( "images/better_match.png", img_better_match);  

	//RANSAC
	vector<DMatch>  m_Matches = better_matches;
	int ptCount = (int)m_Matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i = 0; i < ptCount; i ++) {
		pt = kp1[m_Matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = kp2[m_Matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
	int InlinerCount = 0;
	for (int i=0; i<ptCount; i++) {
		if (m_RANSACStatus[i] == 1)InlinerCount++;
	}
	cout<<"内点：" << InlinerCount << endl;

	vector<Point2f> image1_inlier;
	vector<Point2f> image2_inlier;
	vector<DMatch> inlier_matches;

	inlier_matches.resize(InlinerCount);
	image1_inlier.resize(InlinerCount);
	image2_inlier.resize(InlinerCount);
	InlinerCount=0;
	float inlier_minRx=image1.cols;
	for (int i=0; i<ptCount; i++) {
		if (m_RANSACStatus[i] != 0) {
			image1_inlier[InlinerCount].x = p1.at<float>(i, 0);
			image1_inlier[InlinerCount].y = p1.at<float>(i, 1);
			image2_inlier[InlinerCount].x = p2.at<float>(i, 0);
			image2_inlier[InlinerCount].y = p2.at<float>(i, 1);
			inlier_matches[InlinerCount].queryIdx = InlinerCount;
			inlier_matches[InlinerCount].trainIdx = InlinerCount;

			if(image2_inlier[InlinerCount].x<inlier_minRx) inlier_minRx=image2_inlier[InlinerCount].x;
			InlinerCount++;
		}
	}

	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(image1_inlier, key1);
	KeyPoint::convert(image2_inlier, key2);

	Mat OutImage;
	drawMatches(image1, key1, image2, key2, inlier_matches, OutImage);
	//namedWindow( "Match features", 1);
	//imshow("Match features", OutImage);
	imwrite( "images/feature_match.png", OutImage);  

	//透视变换
	Mat H = findHomography( image1_inlier, image2_inlier, RANSAC );
	vector<Point2f> image1_corners(4);
	vector<Point2f> image2_corners(4);
	image1_corners[0] = Point(0, 0);
	image1_corners[1] = Point( image1.cols, 0 );
	image1_corners[2] = Point( image1.cols, image1.rows );
	image1_corners[3] = Point( 0, image1.rows );
	perspectiveTransform( image1_corners, image2_corners, H);

	Point2f offset( (float)image1.cols, 0); 
	line( OutImage, image2_corners[0]+offset, image2_corners[1]+offset, Scalar( 255, 0, 0), 4 );
	line( OutImage, image2_corners[1]+offset, image2_corners[2]+offset, Scalar( 255, 0, 0), 4 );
	line( OutImage, image2_corners[2]+offset, image2_corners[3]+offset, Scalar( 255, 0, 0), 4 );
	line( OutImage, image2_corners[3]+offset, image2_corners[0]+offset, Scalar( 255, 0, 0), 4 );
	//imshow( "Good Matches & Object detection", OutImage );
	imwrite( "images/better_match&image1_detection.png", OutImage);  

	int drift = image2_corners[1].x;

	int width = int(max(abs(image2_corners[1].x), abs(image2_corners[2].x)));
  	int height= image1.rows;
  	float origin_x=0,origin_y=0;

  	if(image2_corners[0].x<0) {
  		if (image2_corners[3].x<0) origin_x+=min(image2_corners[0].x,image2_corners[3].x);
    		else origin_x+=image2_corners[0].x;
    	}
    	width-=int(origin_x);
  	if(image2_corners[0].y<0) {
    		if (image2_corners[1].y) origin_y+=min(image2_corners[0].y,image2_corners[1].y);
    		else origin_y+=image2_corners[0].y;
  	}
  	height-=int(origin_y);

  	Mat imageturn=Mat::zeros(width,height,image1.type());
  	for (int i=0;i<4;i++) {
    		image2_corners[i].x -= origin_x; 
  	}
  	Mat H1=getPerspectiveTransform(image1_corners, image2_corners);
  	warpPerspective(image1,imageturn,H1,Size(width,height));
	//imshow("image_Perspective", imageturn);
	imwrite( "images/perspective_transform.png", imageturn);  

	//图像融合
	int width_ol=width-int(inlier_minRx-origin_x);
	int start_x=int(inlier_minRx-origin_x);
	cout<<"width: "<<width<<endl;
	cout<<"image1.width: "<<image1.cols<<endl;
	cout<<"start_x: "<<start_x<<endl;
	cout<<"width_ol: "<<width_ol<<endl;
 
	uchar* ptr=imageturn.data;
	double alpha=0, beta=1;
	for (int row=0;row<height;row++) {
		ptr=imageturn.data+row*imageturn.step+(start_x)*imageturn.elemSize();
		for(int col=0;col<width_ol;col++) {
			uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
			uchar* ptr2=image2.data+row*image2.step+(col+int(inlier_minRx))*image2.elemSize();
			uchar* ptr2_c1=ptr2+image2.elemSize1();  uchar* ptr2_c2=ptr2_c1+image2.elemSize1();

			alpha=double(col)/double(width_ol); beta=1-alpha;
			if (*ptr==0&&*ptr_c1==0&&*ptr_c2==0) {
				*ptr=(*ptr2);
				*ptr_c1=(*ptr2_c1);
				*ptr_c2=(*ptr2_c2);
			}
			*ptr=(*ptr)*beta+(*ptr2)*alpha;
			*ptr_c1=(*ptr_c1)*beta+(*ptr2_c1)*alpha;
			*ptr_c2=(*ptr_c2)*beta+(*ptr2_c2)*alpha; 
			ptr+=imageturn.elemSize();
		}
	}
 
	Mat img_result=Mat::zeros(height,width+image2.cols-drift,image1.type());
	uchar* ptr_r=imageturn.data;
 
	for (int row=0;row<height;row++) {
		ptr_r=img_result.data+row*img_result.step;
		for(int col=0;col<imageturn.cols;col++) { 
			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
			uchar* ptr=imageturn.data+row*imageturn.step+col*imageturn.elemSize();
			uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
			*ptr_r=*ptr;
			*ptr_rc1=*ptr_c1;
			*ptr_rc2=*ptr_c2;
			ptr_r+=img_result.elemSize();
		}
		ptr_r=img_result.data+row*img_result.step+imageturn.cols*img_result.elemSize();
		for(int col=imageturn.cols;col<img_result.cols;col++) {
			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
			uchar* ptr2=image2.data+row*image2.step+(col-imageturn.cols+drift)*image2.elemSize();
			uchar* ptr2_c1=ptr2+image2.elemSize1();  uchar* ptr2_c2=ptr2_c1+image2.elemSize1();
			*ptr_r=*ptr2;
			*ptr_rc1=*ptr2_c1;
			*ptr_rc2=*ptr2_c2;
			ptr_r+=img_result.elemSize();
		}
	}
 
	imshow("image_result", img_result);
	imwrite( "images/result.png", img_result);  
	
	waitKey(0);
	return 0;
}
