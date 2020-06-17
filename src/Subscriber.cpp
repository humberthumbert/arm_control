#include <ros/ros.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    
    typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubscriber;
    ImageSubscriber image_sub_;
    ImageSubscriber depth_sub_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync;

public:
    ImageConverter() 
        : it_(nh_),
        image_sub_( nh_, "/camera/color/image_raw", 1),
        depth_sub_( nh_, "/camera/depth/image_rect_raw", 1),
        sync(SyncPolicy(10), image_sub_, depth_sub_) {

        sync.registerCallback(boost::bind( &ImageConverter::callback, this, _1, _2));
        cv::namedWindow(OPENCV_WINDOW);
    }

    ~ImageConverter(){
        cv::destroyWindow(OPENCV_WINDOW);
    }

    void callback(
        const sensor_msgs::ImageConstPtr& image_, 
        const sensor_msgs::ImageConstPtr& depth_){

        cv_bridge::CvImagePtr image_ptr;
        try {
            image_ptr = cv_bridge::toCvCopy(image_, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge image exceptions: %s", e.what());
            return;
        }

        cv_bridge::CvImagePtr depth_ptr;
        try{
            depth_ptr = cv_bridge::toCvCopy(depth_, sensor_msgs::image_encodings::MONO16);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge depth exceptions: %s", e.what());
            return;
        }

        // Update GUI Window
        cv::imshow(OPENCV_WINDOW, image_ptr->image);
        cv::waitKey(3);
    }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "image_converter");
    ImageConverter ic;
    while(ros::ok()){
        ros::spin();
    }
    return 0;
}
