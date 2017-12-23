#ifndef __BOF_H__
#define __BOF_H__

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\nonfree\nonfree.hpp>

#include "opencv\ml.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/legacy/legacy.hpp>
#include "opencv.hpp"
#include <conio.h>


using namespace cv;
using namespace std;

#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

#define TEST_ON 1

#define MAX_TRAINING_NUM 32

void  BuildDictionary(int class_num, int trian_num);
void TestClassify();
int invoice_classify(Mat& img2);
void TrainingDataInit();



#endif