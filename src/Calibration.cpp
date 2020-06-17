#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

class Calibration{
    IplImage* image;
    IplImage* color_image;
public:
    Calibration(string folder_name){
        image = new IplImage();
        color_image = new IplImage();
        std::vector<std::vector<cv::Vec3f> > object_points;
        std::vector<std::vector<cv::Vec2f> > image_points;
        for(int i = 0; i < 15; i++){
            string index_str;
            if (i < 10){
                index_str = "0"+ to_string(i);
            } else {
                index_str = to_string(i);
            }
            string file_name = folder_name + index_str + ".png";
            image = cvLoadImage(file_name.c_str(), 0);
            color_image = cvCreateImage(cvGetSize(image), 8, 3);
            cvCvtColor(image, color_image, CV_GRAY2BGR);

            CvSize pattern_size;
            pattern_size.height = 6;
            pattern_size.width = 8;
            int win_size = 10;

            CvPoint2D32f* corners = new CvPoint2D32f[pattern_size.height * pattern_size.width];
            int corner_count;

            cvFindChessboardCorners(image, pattern_size, corners, &corner_count, 0);

            for (int j=0; j < corner_count; j++){
                cvLine(color_image, 
                cvPoint(cvRound(corners[j].x-7), cvRound(corners[i].y)), 
                cvPoint(cvRound(corners[j].x+7), cvRound(corners[j].y)),
                CV_RGB(255,0,0));
                
                cvLine(color_image, 
                cvPoint(cvRound(corners[j].x), cvRound(corners[i].y-7)), 
                cvPoint(cvRound(corners[j].x), cvRound(corners[j].y+7)),
                CV_RGB(255,0,0));
            }

            cvFindCornerSubPix(image, 
            corners, 
            corner_count, 
            cvSize(win_size, win_size),
            cvSize(-1, -1), 
            cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03));
            for (int j=0; j < corner_count; j++){
                cvLine(color_image, 
                cvPoint(cvRound(corners[j].x-5), cvRound(corners[i].y)), 
                cvPoint(cvRound(corners[j].x+5), cvRound(corners[j].y)),
                CV_RGB(255,0,0));
                
                cvLine(color_image, 
                cvPoint(cvRound(corners[j].x), cvRound(corners[i].y-5)), 
                cvPoint(cvRound(corners[j].x), cvRound(corners[j].y+5)),
                CV_RGB(255,0,0));
            }
            cout << corner_count << endl;
            cvShowImage("detect", color_image);
            cvWaitKey();
        }
    };
};
