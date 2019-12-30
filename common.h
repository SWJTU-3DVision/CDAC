// write by recdcp 2018.3.15
#include <fstream>
#include "hog.h"
#include <cv.h>
#include <io.h>
#include <direct.h>  
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>  
#include <opencv2/nonfree/nonfree.hpp>
#include "facedetect-dll.h"
#define fl at<float>
using namespace cv;

//landmark(x1,x2,x3,...,x_n,y1,y2,y3,...,y_n), size(landmark) = [1,(2 * n)]

//image affine transformation matrix(2*3) with local affine transformation and landmarks
Mat getsrc_roi2(Mat x0, Mat dst);

//image affine transformation matrix(2*3) with cv::estimateRigidTransform and landmarks
Mat getsrc_roi(Mat x0, Mat dst);

//landmarks affine transformation matrix(3*3)
Mat get_new_roi(Mat roi);

//landmarks affine transformation
Mat calc_MatMul(Mat x, Mat roi);

//Extraction of hog descriptors
Mat getHog(Mat parameters, Mat image, VlHogVariant vlhog_variant, int num_cells, int cell_size, int num_bins);

//file_copy
void mycopy(const char* src, const char* dst);

//get file's name by filepath
string get_name_by_path(string path);

//face rects (x_1 y_1 w_1 y_1 '\n' x_2 y_2 w_2 h_2 '\n' .... x_n y_n w_n y_n)
vector<Rect> readRects(string rect_path);

//Euclidean distance of points a and b
double dis(Point2f a, Point2f b);

double getErr_68pt_Normalized_by_boundingbox(Mat gt, Mat x0);

double getErr_68pt_Normalized_by_Pupil_distance(Mat gt, Mat x0);

vector<string>read_files_list(string path);


Mat read_pts_landmarks(string filename);
bool write_Pts_landmarks(string filename, cv::Mat& landmark);
void draw_landmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 251.0, 168.0));
vector<Mat> load_model(string filename);

Rect getBoxRect(Mat gt);

//face rect convert to mat
Mat rect2mat(Rect rect);
Rect mat2rect(Mat x);
Mat get_roi_by_rects(Rect rect, Rect rect_std);


// Get all the specified type files under the folder
bool GetFilenameUnderPath(string folder, std::vector<string>& files, string type);

bool isFacebox(Rect rect, float x, float y);


Mat zoom_landmarks(Mat gt, float zoom);

vector<Rect> Facedetect(Mat frame);


Mat align_mean(Mat mean, cv::Rect facebox, float scaling_x = 1.0f, float scaling_y = 1.0f, float translation_x = 0.0f, float translation_y = 0.0f);