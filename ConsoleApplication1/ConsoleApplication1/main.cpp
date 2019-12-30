#include<iostream>
#include<string>
#include "common.h"
using namespace std;

const int batch_size = 64;
const int level = 4;
const int num_landmarks = 68;
const int feat_dim = 144;
const int epoch = 30;

ofstream out_zhang("out_1.0-aaa.txt");

int saveR(string filename, Mat R)
{
	int retVal = 0;
	FILE *fp = fopen(filename.c_str(), "ab+");
	if (fp == NULL) return -1;
	int cols = R.cols;
	int rows = R.rows;
	
	fwrite(&rows, sizeof(int), 1, fp);
	fwrite(&cols, sizeof(int), 1, fp);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			float data = R.at<float>(i, j);
			fwrite(&data, sizeof(float), 1, fp);
		}
	}
	fclose(fp);
	return retVal;

}

Mat roi_landmarks(Mat landmarks)
{
	double PI = acos(-1);
	int n_points = landmarks.cols / 2;
	double angle = ((rand() % 41) - 20) * PI / 180;
	int nose_id = (n_points == 5) ? 2 : 30;
	float xx = landmarks.at<float>(nose_id);
	float yy = landmarks.at<float>(nose_id + n_points);
	for (int i = 0; i < n_points; i++)
	{
		float x = landmarks.at<float>(i);
		float y = landmarks.at<float>(i + n_points);
		landmarks.at<float>(i) = (x - xx) * cos(angle) - (y - yy) * sin(angle) + xx;
		landmarks.at<float>(i + n_points) = (y - yy) * cos(angle) + (x - xx) * sin(angle) + yy;
	}
	return landmarks;
}

Rect zoomRect(Rect rect, float zoom, float lx, float ly)
{
	float half_w = rect.width / 2;
	float half_h = rect.height / 2;
	float c_x = rect.x + half_w;
	float c_y = rect.y + half_h;
	int x = c_x - half_w * zoom + lx;
	int y = c_y - half_h * zoom + ly;
	int w = rect.width * zoom;
	int h = rect.height * zoom;
	return Rect(max(x, 0), max(y, 0), w, h);
}


Mat land_move(Mat landmarks, float x, float y)
{
	int cols = landmarks.cols / 2;
	for (int i = 0; i < cols; i++)
	{
		landmarks.at<float>(i) = landmarks.at<float>(i) +x;
		landmarks.at<float>(i + cols) = landmarks.at<float>(i + cols) + y;
	}
	return landmarks;
}

Mat getSVD(Mat mdata)
{
	SVD svd(mdata);
	Mat pinvA = svd.vt.t() * Mat::diag(1. / svd.w) * svd.u.t();
	return pinvA;
}


double test_300W_2D(vector<Mat>model, vector<string>test_files)
{
	double sum = 0;
	int cnt = 0;
	Mat std_landmarks_68 = read_pts_landmarks("./model/std_2D_landmarks_68pt.pts");



	Rect rect_std = getBoxRect(std_landmarks_68.clone());
	for (int i = 0; i < test_files.size(); i++)
	{
		string img_path = test_files[i];
		Mat image = imread(img_path);
		Mat img = image.clone();
		string pts_2d_path = img_path.substr(0, img_path.length() - 3) + "pts";

		Mat t_gt_2d_landmarks = read_pts_landmarks(pts_2d_path);

		Rect rect = getBoxRect(t_gt_2d_landmarks.clone());
		cnt++;

		Mat ans_x2d = Mat::zeros(t_gt_2d_landmarks.size(), t_gt_2d_landmarks.type());

		Mat x2d = std_landmarks_68.clone();
		Rect t_rect = zoomRect(rect, 1, 0, 0);
		Mat roi = get_roi_by_rects(t_rect, rect_std);// getsrc_roi(x2d_5pt, std_landmarks_5.clone());//
		cv::warpAffine(img, image, roi, Size(400, 400));
		roi = get_new_roi(roi);


		for (int k = 0; k < model.size(); k++)
		{
			Mat hog = getHog(x2d, image, VlHogVariant::VlHogVariantUoctti, 3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/);
			Mat update = hog * model[k];
			x2d = x2d + update;
		}
		x2d = calc_MatMul(x2d, roi.inv());
		ans_x2d = ans_x2d + x2d;
		double s = getErr_68pt_Normalized_by_Pupil_distance(t_gt_2d_landmarks, ans_x2d);
		sum += s;


	}
	return sum / cnt;
}



double gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	X = X * 9 + 10;
	return X;
}


void  face_2d_RIL_box_train(vector<string>files, vector<string>test_files)
{
	std::cout << "Training, pls don't close. Thanks." << endl;
	std::cout << "train: " << files.size() << endl;

	std::cout << "test:" << test_files.size() << endl;

	string pre_model_path = "";
	vector<Mat>model = load_model(pre_model_path);


	for (int i = 0; model.size() < level; i++)
		model.push_back(Mat::zeros(feat_dim * num_landmarks, num_landmarks * 2, CV_32F));

	double loss2d = 0;

	if (model.size())

		loss2d = test_300W_2D(model, test_files);

	int batch = batch_size;
	int dataSize = files.size();
	float base_lr = 0.01;
	int iter = 0;
	////////////////////////////////////////////////////////
	vector<string>files_random;

	random_shuffle(files.begin(), files.end());

	for (int i = 0; i < 328; i++)
	{
		files_random.push_back(files[i].substr(0, files[i].length() - 3));
	}

	vector<double>random_bias;
	for (int i = 0; i < 328; i++)
	{
		double bias = pow(-1, rand() % 2) * gaussrand();
		random_bias.push_back(bias);
	}


	////////////////////////////////////////////////////////

	std::cout << "beging:" << endl;

	Mat std_landmarks_68 = read_pts_landmarks("./model/std_2D_landmarks_68pt.pts");
	Rect rect_std = getBoxRect(std_landmarks_68.clone());
	for (int e = 1; e <= epoch; e++)
	{
		random_shuffle(files.begin(), files.end());


		Mat batch_gt2d, batch_x2d;
		vector<Mat>batch_imgs;
		vector<Mat>rois;
		vector<string>imgs;
		for (int i = 0; i < dataSize; i++)
		{

			string img_path = files[i];
			Mat image = imread(img_path);
			if (!image.data) continue;
			string pts_2d_path = img_path.substr(0, img_path.length() - 3) + "pts";

			string name = get_name_by_path(img_path);
			if (_access(pts_2d_path.c_str(), 0) == -1)	continue;
			Mat t_gt_2d_landmarks = read_pts_landmarks(pts_2d_path);
			///////////////////////////////////////////////////////
			//int nRet = std::count(files_random.begin(), files_random.end(), img_path.substr(0, img_path.length() - 3));

			vector <string>::iterator iElement = find(files_random.begin(), files_random.end(), img_path.substr(0, img_path.length() - 3));
			Mat radom_1;
			if (iElement != files_random.end())
			{
				int nPosition = distance(files_random.begin(), iElement);
				radom_1 = t_gt_2d_landmarks - random_bias[nPosition];
				//cout << "  find in the vector at position: " << nPosition << endl;
			}
			else
			{
				radom_1 = t_gt_2d_landmarks;
			}

			///////////////////////////////////////////////////////
			Rect rect = getBoxRect(radom_1.clone());

			Mat roi = get_roi_by_rects(rect, rect_std); // or getsrc_roi(t_gt_2d_landmarks, std_landmarks_68)
			roi = get_new_roi(roi);
			Rect rect_x0 = mat2rect(calc_MatMul(rect2mat(rect), roi));


			float zoom = 0.9 + 0.1 * (rand() % 3);
			Mat x0 = zoom_landmarks(std_landmarks_68.clone(), zoom);
			float lxx = rand() % 41 - 20;
			float lyy = rand() % 41 - 20;
			x0 = land_move(x0, lxx, lyy);



			float zoom1 = 0.9 + 0.1 * (rand() % 3);
			float lx = rand() % 61 - 30;
			float ly = rand() % 61 - 30;
			rect_x0 = zoomRect(rect_x0, zoom1, lx, ly);

			rect = mat2rect(calc_MatMul(rect2mat(rect_x0), roi.inv()));
			roi = get_roi_by_rects(rect, rect_std);

			cv::warpAffine(image, image, roi, Size(400, 400));
			roi = get_new_roi(roi);

			batch_imgs.push_back(image.clone());
			imgs.push_back(img_path);
			batch_x2d.push_back(x0);
			batch_gt2d.push_back(calc_MatMul(t_gt_2d_landmarks.clone(), roi));
			if (batch_imgs.size() == batch || i == files.size() - 1)
			{
				if (batch_imgs.size() < batch / 3)continue;
				iter++;
				float lr = base_lr;
				for (int k = 0; k < model.size(); k++)
				{
					Mat hogs;
					for (int j = 0; j < batch_imgs.size(); j++)
					{
						Mat img = batch_imgs[j].clone();

						Mat hog = getHog(batch_x2d.row(j), img, VlHogVariant::VlHogVariantUoctti, 3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/);
						hogs.push_back(hog);
					}
					Mat t_update_3dx = hogs * model[k];
					Mat dx = batch_gt2d - batch_x2d - t_update_3dx;// +0.05 * cv::norm(model[k], cv::NORM_L2);
					Mat dR = getSVD(hogs) * dx;
					model[k] = model[k] + lr * dR;
					Mat dx2d = hogs * model[k];
					batch_x2d = batch_x2d + dx2d;
				}
				double error = cv::norm(batch_x2d, batch_gt2d, cv::NORM_L2) / cv::norm(batch_gt2d, cv::NORM_L2);
				std::cout << "Epoch " << e << " ,Iter " << iter << " ,loss2D = " << error << endl;
				out_zhang << "Epoch " << e << " ,Iter " << iter << " ,loss2D = " << error << endl;
				imgs.clear();
				batch_gt2d.release();
				batch_x2d.release();
				batch_imgs.clear();

				if (error > 1) // error should be < 1
				{
					model.clear();
					model = load_model(pre_model_path);
				}
				else
				{
					if (iter % 40 == 0)
					{
						double e_2d = test_300W_2D(model, test_files);
						std::cout << "Epoch " << e << " ,Iter " << iter << " ,           test_loss2D = " << e_2d << endl;
						out_zhang << "Epoch " << e << " ,Iter " << iter << " ,           test_loss2D = " << e_2d << endl;
						if (e_2d < loss2d)
						{
							loss2d = e_2d;
							string path = "./model/gx/model_roi_zoom_gtbox_iter_", id = "";
							char str[33];
							sprintf(str, "%d_%.6f.bin", iter, e_2d);
							id.append(str);
							path = path + id;
							for (int k = 0; k < model.size(); k++)
								saveR(path, model[k]);
							pre_model_path = path;
						}
					}
				}
			}
		}
	}
}





void train()
{
	string folder;
	vector<string>files, test_files;
	while (cin >> folder) {
		if (folder == "end")break;
		GetFilenameUnderPath(folder, files, "jpg");
		GetFilenameUnderPath(folder, files, "png");
	}
	while (cin >> folder) {
		if (folder == "end")break;
		GetFilenameUnderPath(folder, test_files, "jpg");
		GetFilenameUnderPath(folder, test_files, "png");
	}

	face_2d_RIL_box_train(files, test_files);

}





int main()
{
	train();
	return 0;
}
