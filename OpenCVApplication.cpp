#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>;
#include <cstdlib>;
#include <ctime>
using namespace std;
wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
bool isInside(int i, int j, int height, int width) {
	return i >= 0 && i < height && j >= 0 && j < width;
}
//2. Facem conversie BGR -> LUV si calculam deviatia standard care va fi salvata in variabilele cv_1, ch_2 si ch_3
void convertToLuvAndComputeStd(const Mat& filtered_image, Mat& luv_image,double& ch1_std,double& ch2_std)
{
	//Mat luv_image;
	cvtColor(filtered_image, luv_image, COLOR_BGR2Luv);
	//-------------------------------LUV

	//-------------------------------Deviatie standard
	int height = luv_image.rows;
	int width = luv_image.cols;
	int dim = height * width;

	Mat_<uchar> l(height, width);
	Mat_<uchar> u(height, width);
	Mat_<uchar> v(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b pixel = luv_image.at<Vec3b>(i, j);
			uchar L = pixel[0];
			uchar U = pixel[1];
			uchar V = pixel[2];

			l(i, j) = L;
			u(i, j) = U;
			v(i, j) = V;

		}
	}

	//deviatia standard pentru u*
	double sum_u = 0, media_u = 0, dev_u = 0;
	int hist_u[256] = { 0 };
	double fdp_u[256] = { 0 };

	//hist. originala
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar val = u(i, j);
			hist_u[val]++;// de cate ori apare i in canalul u*
		}
	}

	for (int i = 0; i < 256; i++) {
		sum_u += i * hist_u[i]; //contributia fiecarei valori - suma ponderata
		fdp_u[i] = (double)hist_u[i] / dim; //frecventa normalizata ~ probabilitatea de a apartine
	}

	media_u = sum_u / dim; //media canalului u
	for (int i = 0; i < 256; i++) {
		//fdp = frecventa pixelilor cu valoarea i
		//media canalului i
		dev_u += (i - media_u) * (i - media_u) * fdp_u[i]; //formula deviatia standard
	}
	 ch1_std = sqrt(dev_u);

	//deviatia standard pentru v*
	double sum_v = 0, media_v = 0, dev_v = 0;
	int hist_v[256] = { 0 };
	double fdp_v[256] = { 0 };

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar val = v(i, j);
			hist_v[val]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		sum_v += i * hist_v[i];
		fdp_v[i] = (double)hist_v[i] / dim;
	}
	media_v = sum_v / dim;
	for (int i = 0; i < 256; i++) {
		dev_v += (i - media_v) * (i - media_v) * fdp_v[i];
	}
	 ch2_std = sqrt(dev_v);

	printf("Standard deviation u* (ch1_std) = %.4f\n", ch1_std); //cat de mult variaza culoarea in imagine pe canalul u*
	printf("Standard deviation v* (ch2_std) = %.4f\n", ch2_std); //cat de mult variaza culoarea in imagine pe canalul v*

}
//3+4. Algoritm region-growing
//primeste imaginea in luv, deviatiile standard , factor de scalare => matricea de labels si numarul maxim de labels(regiuni) detectate
void regionGrowing(const Mat& luv_image, double ch1_std, double ch2_std, double value,
	Mat_<int>& labels_out, int& max_label, vector<Vec3d>& region_means_out) {

	int height = luv_image.rows;
	int width = luv_image.cols;

	region_means_out.resize(1); //index 0 e ignorat deoarece etichetele incep de la 1

	Mat_<float> channel_l(height, width); // u*
	Mat_<float> channel_u(height, width); // u*
	Mat_<float> channel_v(height, width); // v*

	double sum_ch0 = 0;
	double sum_ch1 = 0;
	double sum_ch2 = 0;
	int count = 0;
	int w = 3;

	//Extract u* si v* din imaginea Luv
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = luv_image.at<Vec3b>(i, j);
			//extragem canalele u si v si salvam in matrici separate
			channel_l(i, j) = pixel[0];
			channel_u(i, j) = pixel[1];
			channel_v(i, j) = pixel[2];
		}
	}

	//initializare etichete cu 0
	Mat_<int> labels(height, width, 0);
	int k = 1; //eticheta curecnta

	//folosim deviatia standard pentru a seta un prag adaptat la variatia culorii
	double T = value * sqrt(ch1_std * ch1_std + ch2_std * ch2_std);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (labels(i, j) != 0) continue;  //sarim peste pixelii etichetati

			//initializare FIFO cu pixelul seed si marcare cu eticheta k
			queue<Point> q;
			q.push(Point(j, i));
			labels(i, j) = k;

			sum_ch0 = 0;
			sum_ch1 = 0;
			sum_ch2 = 0;
			count = 0;

			//initializam medie si nr pixeli
			for (int di = -(w / 2); di <= (w / 2); di++) {
				for (int dj = -(w / 2); dj <= (w / 2); dj++) {
					int ni = i + di;
					int nj = j + dj;

					if (isInside(ni, nj, height, width) && labels(ni, nj) == 0) { //daca pixelul e valid si neetichetat
						sum_ch0 += channel_l(ni, nj);
						sum_ch1 += channel_u(ni, nj);
						sum_ch2 += channel_v(ni, nj);
						count++;
					}
				}
			}
			double ch0_avg = (count > 0) ? sum_ch0 / count : channel_l(i, j);
			double ch1_avg = (count > 0) ? sum_ch1 / count : channel_u(i, j);
			double ch2_avg = (count > 0) ? sum_ch2 / count : channel_v(i, j);

			int N = 1;

			// Region growing
			while (!q.empty()) {
				Point p = q.front();
				q.pop();

				for (int dx = -1; dx <= 1; dx++) { //pt toti vecinii pixelului din coada
					for (int dy = -1; dy <= 1; dy++) {

						if (dx == 0 && dy == 0) continue;

						int ni = p.y + dy;
						int nj = p.x + dx;

						//daca un vecin este in interiorul imaginii, nu e etichetat
						//si distanta din spatiul (u, v) < T
						if (!isInside(ni, nj, height, width)) continue;
						if (labels(ni, nj) != 0) continue;

						//distanta Euclidiană în spațiul (u*, v*)
						double d1 = channel_u(ni, nj) - ch1_avg; //distanta euclidiana(d.p.d.v al culorii) dintre pixelul vecin si media regiunii
						double d2 = channel_v(ni, nj) - ch2_avg;
						double dist = sqrt(d1 * d1 + d2 * d2);

						if (dist < T) {
							q.push(Point(nj, ni));
							labels(ni, nj) = k;

							//actualiam medii
							ch0_avg = (N * ch0_avg + channel_l(ni, nj)) / (N + 1);
							ch1_avg = (N * ch1_avg + channel_u(ni, nj)) / (N + 1);
							ch2_avg = (N * ch2_avg + channel_v(ni, nj)) / (N + 1);
							N++;
						}
					}
				}
			}
			region_means_out.push_back(Vec3d(ch0_avg, ch1_avg, ch2_avg));
			k++; // trecem la următoarea regiune
		}
	}

	labels_out = labels.clone();
	max_label = k - 1;
}
//reconstruiesc imaginea color etichetata
Mat reconstructFromLabelsLuvToRgb(const Mat_<int>& labels, vector<Vec3d>& region_means) {

	int height = labels.rows;
	int width = labels.cols;

	Mat result_luv(height, width, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int label = labels(i, j);
			if (label == 0) continue;

			Vec3d avg = region_means[label];
			result_luv.at<Vec3b>(i, j) = Vec3b((uchar)avg[0], (uchar)avg[1], (uchar)avg[2]);
		}
	}

	// Convertim din Luv în RGB pentru afișare
	Mat result_rgb;
	cvtColor(result_luv, result_rgb, COLOR_Luv2BGR);
	return result_rgb;
}


//5.Postprocesare - eroziune si dilatare
void erodeLabels(Mat_<int>& labels) {
	Mat_<int> new_labels = labels.clone();
	int height = labels.rows;
	int width = labels.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels(i, j) == 0) continue; //doar pixelii etichetati

			int currentLabel = labels(i, j);
			bool differentNeighbor = false;

			//8 vecini
			for (int dx = -1; dx <= 1; dx++) {
				for (int dy = -1; dy <= 1; dy++) {
					if (dx == 0 && dy == 0) continue;

					int ni = i + dx;
					int nj = j + dy;
					if (isInside(ni, nj, height, width)) {
						if (labels(ni, nj) != currentLabel) { //daca avem vecini cu etichete liferite
							differentNeighbor = true;
							break;
						}
					}
				}
				if (differentNeighbor) break;
			}

			if (differentNeighbor) //pixelii de la margine devin neetichetati
				new_labels(i, j) = 0;
		}
	}

	labels = new_labels.clone(); //supracriem noua imagine
}
void dilateLabels(Mat_<int>& labels) {
	Mat_<int> new_labels = labels.clone();
	int height = labels.rows, width = labels.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels(i, j) != 0) {
				// Eticheteazs vecinii neetichetati
				for (int dx = -1; dx <= 1; dx++) {
					for (int dy = -1; dy <= 1; dy++) {
						if (dx == 0 && dy == 0) continue;

						int ni = i + dx;
						int nj = j + dy; //exitinde regiunile in zonele neetichetate din apropiere
						if (isInside(ni, nj, height, width) && labels(ni, nj) == 0) {
							new_labels(ni, nj) = labels(i, j); //eticheta lui se propaga catre vecini
						}
					}
				}
			}
		}
	}

	labels = new_labels.clone();
}

void postprocessLabels(Mat_<int>& labels) {
	for (int i = 0; i < 3; i++) {
		erodeLabels(labels);
	}

	for (int i = 0; i < 10; i++) {
		dilateLabels(labels);
	}
}

//Pct. 1 - incarcam imaginea si facem filtrarea cu filtru gaussian trece jos de dimensiune 5
void gaussian_filtered() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat rgb_image = imread(fname, IMREAD_COLOR);

		if (!rgb_image.data) //check for invalid input
		{
			printf("Could not open or find the image\n");
			return;
		}

		int height = rgb_image.rows;
		int width = rgb_image.cols;

		Mat filtered_image(height, width, CV_8UC3);
		Mat luv_image;
		double ch1_std, ch2_std;
		Mat_<uchar> blue(height, width);
		Mat_<uchar> green(height, width);
		Mat_<uchar> red(height, width);

		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				Vec3b pixel = rgb_image.at<Vec3b>(i, j);
				uchar B = pixel[0];
				uchar G = pixel[1];
				uchar R = pixel[2];

				blue(i, j) = B;
				green(i, j) = G;
				red(i, j) = R;

			}
		}

		float kernel[5][5] = { //kernel normalizat => nu mai e nevoie de impartire la H(u,v)
			{0.0005f, 0.0050f, 0.0109f, 0.0050f, 0.0005f},
			{0.0050f, 0.0521f, 0.1139f, 0.0521f, 0.0050f},
			{0.0109f, 0.1139f, 0.2487f, 0.1139f, 0.0109f},
			{0.0050f, 0.0521f, 0.1139f, 0.0521f, 0.0050f},
			{0.0005f, 0.0050f, 0.0109f, 0.0050f, 0.0005f}
		};

		int kernel_size = 5;
		int k_offset = kernel_size / 2; //evitam marginile; nu se filtreaza pixeli care nu pot fi acoperiti complet de kernel

		Mat_<uchar> red_filtered(height, width, (uchar)0);
		Mat_<uchar> green_filtered(height, width, (uchar)0);
		Mat_<uchar> blue_filtered(height, width, (uchar)0);

		//ignorma primele k linii si k coloane
		for (int i = k_offset; i < height - k_offset; i++) {//se suprapune kernelul peste zona curenta.
			for (int j = k_offset; j < width - k_offset; j++) {

				float sum_r = 0.0f;
				float sum_g = 0.0f;
				float sum_b = 0.0f;

				for (int u = 0; u < kernel_size; u++) {
					for (int v = 0; v < kernel_size; v++) {

						int x = i + u - k_offset;
						int y = j + v - k_offset;

						sum_r += kernel[u][v] * red(x, y); //produs scalar intre kernel si valorile pixelilor vecini
						sum_g += kernel[u][v] * green(x, y);
						sum_b += kernel[u][v] * blue(x, y);
					}
				}

				red_filtered(i, j) = (uchar)sum_r;
				green_filtered(i, j) = (uchar)sum_g;
				blue_filtered(i, j) = (uchar)sum_b;

			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (i < k_offset || i >= height - k_offset || j < k_offset || j >= width - k_offset) {
					red_filtered(i, j) = red(i, j);
					green_filtered(i, j) = green(i, j);
					blue_filtered(i, j) = blue(i, j);
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				filtered_image.at<Vec3b>(i, j)[0] = blue_filtered(i, j);
				filtered_image.at<Vec3b>(i, j)[1] = green_filtered(i, j);
				filtered_image.at<Vec3b>(i, j)[2] = red_filtered(i, j);
			}
		}

		convertToLuvAndComputeStd(filtered_image, luv_image, ch1_std, ch2_std);

		//Segmentare pe baza region growing
		double value;
		cout << "Factor (value > 0) = ";
		cin >> value;

		Mat_<int> labels;
		int max_label;
		std::vector<Vec3d> region_means;

		regionGrowing(luv_image, ch1_std, ch2_std, value, labels, max_label, region_means);

		Mat final_colorized1 = reconstructFromLabelsLuvToRgb(labels, region_means);

		// Postprocesare
		Mat_<int> labels_post = labels.clone();
		postprocessLabels(labels_post);

		Mat final_colorized2 = reconstructFromLabelsLuvToRgb(labels_post, region_means); 

		imshow("Original Image", rgb_image);
		imshow("Gaussian Filtered Image", filtered_image);
		imshow("Labeled (Before Postprocessing)", final_colorized1);
		imshow("Labeled (After Postprocessing)", final_colorized2);

		waitKey();
	}
}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Gaussian Filtered Image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				gaussian_filtered();
				break;
		}
	}
	while (op!=0);
	return 0;
}