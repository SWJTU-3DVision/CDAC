#include "common.h"

Mat getsrc_roi2(Mat x0, Mat dst)
{
	int size = dst.cols / 2;
	Mat A = Mat::zeros(size * 2, 4, dst.type());
	Mat B = Mat::zeros(size * 2, 1, dst.type());

	//[ x1 -y1 1 0] [a]       [x_1]
	//[ y1  x1 0 1] [b]   =   [y_1]
	//[ x2 -y2 1 0] [c]       [x_2]
	//[ y2  x2 0 1] [d]       [y_2]	

	for (int i = 0; i < size; i++)
	{
		A.at<float>(i << 1, 0) = x0.fl(i);// roi_dst[i].x;
		A.at<float>(i << 1, 1) = -x0.fl(i + size);
		A.at<float>(i << 1, 2) = 1;
		A.at<float>(i << 1, 3) = 0;
		A.at<float>(i << 1 | 1, 0) = x0.fl(i + size);
		A.at<float>(i << 1 | 1, 1) = x0.fl(i);
		A.at<float>(i << 1 | 1, 2) = 0;
		A.at<float>(i << 1 | 1, 3) = 1;

		B.at<float>(i << 1) = dst.fl(i);
		B.at<float>(i << 1 | 1) = dst.fl(i + size);
	}

	Mat roi = Mat::zeros(2, 3, A.type());
	cv::Mat AT = A.t();
	cv::Mat ATA = A.t() * A;
	Mat R = ATA.inv() * AT * B;

	//roi = [a -b c;b a d ];

	roi.at<float>(0, 0) = R.at<float>(0, 0);
	roi.at<float>(0, 1) = -R.at<float>(1, 0);
	roi.at<float>(0, 2) = R.at<float>(2, 0);
	roi.at<float>(1, 0) = R.at<float>(1, 0);
	roi.at<float>(1, 1) = R.at<float>(0, 0);
	roi.at<float>(1, 2) = R.at<float>(3, 0);
	return roi;

}

Mat getsrc_roi(Mat x0, Mat dst)
{
	int size = dst.cols / 2;
	std::vector<cv::Point2f>roi_src(size);
	std::vector<cv::Point2f>roi_dst(size);
	for (int z = 0; z < size; z++)
	{
		roi_src[z].x = x0.fl(z);
		roi_src[z].y = x0.fl(z + size);
		roi_dst[z].x = dst.fl(z);
		roi_dst[z].x = dst.fl(z + size);
	}
	Mat roi = cv::estimateRigidTransform(roi_src, roi_dst, 0);//findHomography(roi_src, roi_dst);//
	if (roi.data == NULL) return getsrc_roi2(x0, dst);
	roi.convertTo(roi, x0.type());
	return roi;
}

Mat get_new_roi(Mat roi)
{
	roi = roi.t();
	Mat roi_tmp(3, 3, roi.type());
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++)
			roi_tmp.at<float>(i, j) = roi.at<float>(i, j);
	}
	roi_tmp.at<float>(0, 2) = 0.0;
	roi_tmp.at<float>(1, 2) = 0.0;
	roi_tmp.at<float>(2, 2) = 1.0;
	return roi_tmp;
}

Mat calc_MatMul(Mat x, Mat roi)
{
	int n_points = x.cols / 2;
	Mat y(1, x.cols, x.type());
	for (int j = 0; j < n_points; j++)
	{
		y.at<float>(j) = x.at<float>(j)*roi.at<float>(0, 0) + x.at<float>(j + n_points) * roi.at<float>(1, 0) + roi.at<float>(2, 0);
		y.at<float>(j + n_points) = x.at<float>(j)*roi.at<float>(0, 1) + x.at<float>(j + n_points)*roi.at<float>(1, 1) + roi.at<float>(2, 1);
	}
	return y;
}

Mat getHog(Mat parameters, Mat image, VlHogVariant vlhog_variant, int num_cells, int cell_size, int num_bins)
{
	assert(parameters.rows == 1);
	using cv::Mat;

	Mat gray_image;
	if (image.channels() == 3) {
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	}
	else {
		gray_image = image;
	}

	// Note: We could use the 'regressorLevel' to choose the window size (and
	// other parameters adaptively). We omit this for the sake of a short example.

	int patch_width_half = num_cells * (cell_size / 2);

	Mat hog_descriptors; // We'll get the dimensions later from vl_hog_get_*

	int num_landmarks = parameters.cols / 2;
	for (int i = 0; i < num_landmarks; ++i) {
		int x = cvRound(parameters.fl(i));
		int y = cvRound(parameters.fl(i + num_landmarks));
		x = x < 0 ? 0 : x;
		y = y < 0 ? 0 : y;
		x = x > gray_image.cols ? gray_image.cols - 1 : x;
		y = y > gray_image.rows ? gray_image.rows - 1 : y;
		Mat roi_img;
		if (x - patch_width_half < 0 || y - patch_width_half < 0 || x + patch_width_half >= gray_image.cols || y + patch_width_half >= gray_image.rows) {
			// The feature extraction location is too far near a border. We extend the
			// image (add a black canvas) and then extract from this larger image.
			int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0; // x and y are patch-centers
			int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
			int borderRight = (x + patch_width_half) >= gray_image.cols ? std::abs(gray_image.cols - (x + patch_width_half)) : 0;
			int borderBottom = (y + patch_width_half) >= gray_image.rows ? std::abs(gray_image.rows - (y + patch_width_half)) : 0;
			Mat extendedImage = gray_image.clone();
			cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
			cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2); // Rect: x y w h. x and y are top-left corner.
			roi_img = extendedImage(roi).clone(); // clone because we need a continuous memory block
		}
		else {
			cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
			roi_img = gray_image(roi).clone(); // clone because we need a continuous memory block
		}
		roi_img.convertTo(roi_img, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
		VlHog* hog = vl_hog_new(vlhog_variant, num_bins, false); // transposed (=col-major) = false
		vl_hog_put_image(hog, (float*)roi_img.data, roi_img.cols, roi_img.rows, 1, cell_size); // (the '1' is numChannels)
		int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
		int hh = static_cast<int>(vl_hog_get_height(hog));
		int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
		Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
		vl_hog_extract(hog, hogArray.ptr<float>(0));
		vl_hog_delete(hog);
		Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
		// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector:
		for (int j = 0; j < dd; ++j) {
			Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
			hogFeatures = hogFeatures.t(); // necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
			hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
			Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
			hogFeatures.copyTo(currentDimSubMat);
		}
		hogDescriptor = hogDescriptor.t(); // now a row-vector
		hog_descriptors.push_back(hogDescriptor);
	}
	// concatenate all the descriptors for this sample vertically (into a row-vector):
	hog_descriptors = hog_descriptors.reshape(0, hog_descriptors.cols * num_landmarks).t();
	return hog_descriptors;
};

void mycopy(const char* src, const char* dst)
{
	using namespace std;
	ifstream in(src, ios::binary);
	ofstream out(dst, ios::binary);
	if (!in.is_open()) {
		cout << "error open file " << src << endl;
		exit(EXIT_FAILURE);
	}
	if (!out.is_open()) {
		cout << "error open file " << dst << endl;
		exit(EXIT_FAILURE);
	}
	if (src == dst) {
		cout << "the src file can't be same with dst file" << endl;
		exit(EXIT_FAILURE);
	}
	char buf[2048];
	long long totalBytes = 0;
	while (in)
	{
		//read从in流中读取2048字节，放入buf数组中，同时文件指针向后移动2048字节
		//若不足2048字节遇到文件结尾，则以实际提取字节读取。
		in.read(buf, 2048);
		//gcount()用来提取读取的字节数，write将buf中的内容写入out流。
		out.write(buf, in.gcount());
		totalBytes += in.gcount();
	}
	in.close();
	out.close();
}

string get_name_by_path(string path)
{
	int pos = path.find_last_of('\\');
	if (pos == 0)pos = path.find_last_of('/');
	string name = path.substr(pos + 1);
	name = name.substr(0, name.length() - 4);
	return name;
}

vector<Rect> readRects(string rect_path)
{
	vector<Rect>rects;
	int rx = -1, ry = -1, rw = -1, rh = -1;
	FILE *fw = fopen(rect_path.c_str(), "r");
	if (fw == NULL)	return rects;
	while (fscanf(fw, "%d %d %d %d", &rx, &ry, &rw, &rh) != EOF) {
		rects.push_back(Rect(rx, ry, rw, rh));
	}
	fclose(fw);
	return rects;
}

double dis(Point2f a, Point2f b)
{
	double dlet = (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
	return sqrt(dlet);
}

double getErr_68pt_Normalized_by_boundingbox(Mat gt, Mat x0)
{
	if (!gt.data) return 0;
	double sum = 0;
	int num_points = x0.cols / 2;
	float maxx = gt.at<float>(0);
	float maxy = gt.at<float>(num_points);
	float minx = gt.at<float>(0);
	float miny = gt.at<float>(num_points);
	for (int i = 0; i < num_points; i++)
	{
		Point2f p_gt, p_x0;
		p_gt.x = gt.at<float>(i);
		p_x0.x = x0.at<float>(i);
		p_gt.y = gt.at<float>(i + num_points);
		p_x0.y = x0.at<float>(i + num_points);
		sum += dis(p_gt, p_x0);

		maxx = max(maxx, p_gt.x);
		maxy = max(maxy, p_gt.y);
		minx = min(minx, p_gt.x);
		miny = min(miny, p_gt.y);
	}
	double d = sqrt((maxx - minx)*(maxy - miny));
	return sum / (num_points * d);
}

double getErr_68pt_Normalized_by_Pupil_distance(Mat gt, Mat x0)
{
	if (!gt.data) return 1;
	double sum = 0;
	Point2f le, re;
	le.x = 0; re.x = 0;
	le.y = 0; re.y = 0;
	int num_points = x0.cols / 2;
	for (int i = 0; i < num_points; i++)
	{
		Point2f p_gt, p_x0;
		p_gt.x = gt.at<float>(i);
		p_x0.x = x0.at<float>(i);
		p_gt.y = gt.at<float>(i + num_points);
		p_x0.y = x0.at<float>(i + num_points);
		if (i>35 && i < 42)
		{
			le.x += p_gt.x;
			le.y += p_gt.y;
		}
		if (i>41 && i < 48)
		{
			re.x += p_gt.x;
			re.y += p_gt.y;
		}
		//if (i == 30 || i == 48 || i == 54)
		sum += dis(p_gt, p_x0);
	}
	le.x /= 6;
	le.y /= 6;
	re.x /= 6;
	re.y /= 6;
	//sum += dis(le, lle) + dis(re, rre);
	return sum / (num_points * dis(le, re));
	//return sum / (num_points);
}

vector<string>read_files_list(string path)
{
	FILE *fp = fopen(path.c_str(), "r");
	vector<string>files;
	char str[150];
	while (fscanf(fp, "%s", str) != EOF)
		files.push_back(str);
	fclose(fp);
	return files;
}

void draw_landmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color)
{
	auto num_landmarks = landmarks.cols / 2;
	for (int i = 0; i < num_landmarks; ++i) {
		if (landmarks.at<float>(i)<0 || landmarks.at<float>(i + num_landmarks) < 0)continue;
		cv::circle(image, cv::Point2f(landmarks.at<float>(i), landmarks.at<float>(i + num_landmarks)), 3, color);
	}
	cvNamedWindow("Lena");
	imshow("Lena", image);
	cvWaitKey(1000);
}

Mat read_pts_landmarks(std::string filename)
{

	FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) return Mat();

	char str[100];
	fgets(str, 100, fp);  // 'version: 1'
	fgets(str, 100, fp);  // // 'n_points : num'
	int len = strlen(str) - 1;
	int n_points = 0;
	while (!(str[len] >= '0' && str[len] <= '9'))len--;
	for (int i = len; i >= 0; i--)
	{
		if (str[i] >= '0' && str[i] <= '9')
			n_points += pow(10, len - i) * (str[i] - '0');
		else break;
	}
	fgets(str, 100, fp);//"{"
	Mat landmarks(1, n_points << 1, CV_32F);
	for (int i = 0; i < n_points; i++)
	{
		float x, y;
		fscanf(fp, "%f %f", &x, &y);
		landmarks.fl(i) = x;
		landmarks.fl(i + n_points) = y;
	}
	fclose(fp);
	return landmarks;
};

bool write_Pts_landmarks(string filename, cv::Mat& landmark)
{

	FILE *fp = fopen(filename.c_str(), "w+");

	if (fp == NULL) return false;

	// 写入数据  
	fprintf(fp, "version : 1\n");
	fprintf(fp, "n_points:  %d\n", landmark.cols / 2);
	fprintf(fp, "{\n");
	for (int i = 0; i < landmark.cols / 2; i++)
	{
		float x = landmark.fl(i);  //读取数据，at<type> - type 是矩阵元素的具体数据格式 
		float y = landmark.fl(i + landmark.cols / 2);
		fprintf(fp, "%.5f %.5f\n", x, y);
	}
	fprintf(fp, "}\n");
	fclose(fp);

	return true;
}

vector<Mat> load_model(string filename)
{
	vector<Mat>model;
	FILE *fp = fopen(filename.c_str(), "rb");
	if (fp == NULL) return model;

	int cols, rows;
	while (fp != NULL && fread(&rows, sizeof(int), 1, fp))
	{
		fread(&cols, sizeof(int), 1, fp);
		//	printf("%d * %d\n", rows, cols);
		Mat R(rows, cols, CV_32FC1);

		//fread(&R, sizeof(float), rows*cols, fp);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				float data;
				fread(&data, sizeof(float), 1, fp);
				R.at<float>(i, j) = data;
			}
		}

		model.push_back(R);
	}
	fclose(fp);
	return model;
};

Rect getBoxRect(Mat gt)
{
	const int MAXV = 99999;
	float minx = MAXV, miny = MAXV, maxx = -MAXV, maxy = -MAXV, zero = 0.0;
	for (int i = 0; i < gt.cols / 2; i++)
	{
		//if (gt.at<float>(i) < 0 || gt.at<float>(i + gt.cols / 2) < 0)continue;
		minx = min(minx, gt.at<float>(i));
		miny = min(miny, gt.at<float>(i + gt.cols / 2));
		maxx = max(maxx, gt.at<float>(i));
		maxy = max(maxy, gt.at<float>(i + gt.cols / 2));
	}
	return Rect(minx, miny, maxx - minx, maxy - miny);
}

Mat rect2mat(Rect rect)
{
	Mat x(1, 8, CV_32F);
	x.at<float>(0) = rect.x;
	x.at<float>(4) = rect.y;
	x.at<float>(1) = rect.x + rect.width;
	x.at<float>(5) = rect.y;
	x.at<float>(2) = rect.x;
	x.at<float>(6) = rect.y + rect.height;
	x.at<float>(3) = rect.x + rect.width;
	x.at<float>(7) = rect.y + rect.height;
	return x;


}

Rect mat2rect(Mat x)
{
	Rect rect;
	rect.x = x.at<float>(0);
	rect.y = x.at<float>(4);
	rect.width = x.at<float>(3) - rect.x;
	rect.height = x.at<float>(7) - rect.y;
	return rect;


}

Mat get_roi_by_rects(Rect rect, Rect rect_std)
{
	Mat x0 = rect2mat(rect);
	Mat std = rect2mat(rect_std);
	Mat roi = getsrc_roi(x0, std);
	return roi;
}

bool isFacebox(Rect rect, float x, float y)
{
	return (x > rect.x && y >rect.y && x < rect.x + rect.width && y < rect.y + rect.height);
}

bool GetFilenameUnderPath(string folder, std::vector<string>& files, string type)
{
	intptr_t   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(folder).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					GetFilenameUnderPath(p.assign(folder).append("\\").append(fileinfo.name), files, type);
			}
			else
			{
				char *ext = strrchr(fileinfo.name, '.');
				if (ext) {
					ext++;
					if (_stricmp(ext, type.c_str()) == 0)
						files.push_back(p.assign(folder).append("\\").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return true;
}

Mat zoom_landmarks(Mat gt, float zoom)
{
	Mat x0 = gt.clone();
	int num = x0.cols / 2;
	int nose_id = (num == 5) ? 2 : 30;
	double xx = x0.at<float>(nose_id);
	double yy = x0.at<float>(nose_id + num);
	if (nose_id == 30) { xx = 200; yy = 200; }
	for (int i = 0; i < x0.cols; i++)
		x0.at<float>(i) = x0.at<float>(i) * zoom;
	double x = xx - x0.at<float>(nose_id);
	double y = yy - x0.at<float>(nose_id + num);
	for (int i = 0; i < num; i++)
	{
		x0.at<float>(i) = x0.at<float>(i) +x;
		x0.at<float>(i + num) = x0.at<float>(i + num) + y;
	}
	return x0;
}

vector<Rect> Facedetect(Mat frame)
{
	vector<Rect> rects;
	if (!frame.data)return rects;
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY);
	int * pResults = NULL;
	pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step, 1.25f, 5, 24); //1.25f, 5, 241.28f, 2, 24
	int peopleNUM = (pResults ? *pResults : 0);
	int maxSize = 0; //寻找最大人脸框
	for (int i = 0; i < peopleNUM; i++)//代表有几张人脸(pResults ? *pResults : 0)
	{
		short * p = ((short*)(pResults + 1)) + 6 * i;
		Rect rect = Rect(p[0], p[1], p[2], p[3]);
		//寻找最大人脸框
		//int size = p[2] * p[3];
		//if (size > maxSize)
		//{
		//	rects.push_back(rect);
		//	maxSize = size;
		//}
		rects.push_back(rect);
	}
	return rects;
}


Mat align_mean(Mat mean, cv::Rect facebox, float scaling_x, float scaling_y, float translation_x, float translation_y)
{
	// Initial estimate x_0: Center the mean face at the [0, 1] x [0,1] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [0, 1] x [0, 1]), and just place it in the face-box as
	// if the box is [0, 1] x [0, 1]. (i.e. the mean coordinates get upscaled)
	Mat aligned_mean = mean.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2);
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);
	aligned_mean_x = (aligned_mean_x*scaling_x + translation_x) * facebox.width + facebox.x;
	aligned_mean_y = (aligned_mean_y*scaling_y + translation_y) * facebox.height + facebox.y;
	return aligned_mean;
}