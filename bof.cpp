
#include "bof.h"
#include <io.h>


//create a nearest neighbor matcher
static Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
//create Sift feature point extracter
static Ptr<FeatureDetector> detector1(new SiftFeatureDetector());
//create Sift descriptor extractor
static Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
//create BoF (or BoW) descriptor extractor
static BOWImgDescriptorExtractor bowDE(extractor, matcher);

static CvSVM svm;

static int load_features_from_file(const string& file_name,Mat& features)
{
    FILE* fp = fopen(file_name.c_str(), "r");
    if (fp == NULL)
    {
        printf("fail to open %s\n", file_name.c_str());
        return -1;
    }
    printf("loading file %s\n", file_name.c_str());

    vector<float> inData;
    while (!feof(fp))
    {
        float tmp;
        fscanf(fp, "%f", &tmp);
        inData.push_back(tmp);
    }

    //vector to Mat
    int mat_cols = 128;
    int mat_rows = inData.size() / 128;
    features = Mat::zeros(mat_rows, mat_cols, CV_32FC1);
    int count = 0;
    for (int i = 0; i < mat_rows; i++)
    {
        for (int j = 0; j < mat_cols; j++)
        {
            features.at<float>(i, j) = inData[count++];
        }
    }

    return 0;
}

static int WriteFeatures2File(const string& file_name,const Mat& features)
{
    FILE* fp = fopen(file_name.c_str(), "a+");
    if (fp == NULL)
    {
        printf("fail to open %s\n", file_name.c_str());
        return -1;
    }

    for (int i = 0; i < features.rows; i++)
    {
        for (int j = 0; j < features.cols; j++)
        {
            int data = features.at<float>(i, j);
            fprintf(fp, "%d\t", data);
        }
        fprintf(fp,"\n");
    }

    fclose(fp);

    return 0;
}

//建立词袋，svm训练
void  BuildDictionary(int class_num,int trian_num)
{
	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[100];
	//to store the current input image
	Mat input;

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;
	//To store the SIFT descriptor of current image
	Mat descriptor;
	//To store all the descriptors that are extracted from all the images
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
	SiftDescriptorExtractor detector;

	/*
	cv::Ptr<cv::DescriptorMatcher> matcher =    cv::DescriptorMatcher::create("FlannBased");
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract( extractor, matcher );
	cv::SurfFeatureDetector detector(500);
	*/

    /*第一步，计算目录下所有训练图片的features，放进featuresUnclustered*/
    printf("step1:sift features extracting...\n");
    for (int num = 1; num < MAX_TRAINING_NUM; num++)
    {
        
        sprintf(filename, ".\\training\\%d\\train.txt", num);
        //首先先检查一下该类文件夹下有没有用于train的特征文件，有的话就不需要提取特征点了
        if (_access(filename, 0) == -1)
        {
            printf("extracting features %d class\n", num);
            for (int i = 1; i <= MAX_TRAINING_NUM; i++)
            {
                sprintf(filename, ".\\training\\%d\\%d.jpg", num, i);
                //create the file name of an image
                //open the file
                input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale		
                if (input.empty())
                {
                    break;
                }
                //resize:reduce keypoints numbers to accerlate
                resize(input, input, Size(), 0.5, 0.5);
                //detect feature points
                detector.detect(input, keypoints);
                printf("keypoints:%d\n", keypoints.size());
                //compute the descriptors for each keypoint
                detector.compute(input, keypoints, descriptor);
                //save descriptor to file
                char train_name[32] = { 0 };
                sprintf(train_name, ".\\training\\%d\\train.txt", num);
                WriteFeatures2File(train_name, descriptor);
                //put the all feature descriptors in a single Mat object 
                featuresUnclustered.push_back(descriptor);
                //train_features[num][i].push_back(descriptor);

            }
        }
        else
        {
            Mat descriptor;
            load_features_from_file(filename, descriptor);
            featuresUnclustered.push_back(descriptor);
        }
    }

    /*第二步，定义好聚类的中心数目，进行聚类，并得到词典dictionary*/
    printf("step2:clusting...\n");
	int dictionarySize = 1000;  //类心数目，即codebook num
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 1000, 0.001);  //最大迭代1000次
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;  //kmeans++初始化
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);  //聚类
	//store the vocabulary
	FileStorage fs(".\\dictionary1.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	cout << "Saving BoW dictionary\n";

    /*第三步，计算每个类别的词典直方图*/
    printf("step3:generating dic histogram...\n");
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector1(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	cout << "extracting histograms in the form of BOW for each image " << endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	int k = 0;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Mat img2;
	//extracting histogram in the form of bow for each image 
    for (int num = 1; num <= class_num; num++)
    {
        for (int i = 1; i <= trian_num; i++)
        {
            sprintf(filename, ".\\training\\%d\\%d.jpg", num,i);

            //sprintf(filename, "%d%s%d%s", j, " (", i, ").jpg");
            img2 = cvLoadImage(filename, 0);

            if (img2.empty())
            {
                break;
            }

            resize(img2, img2, Size(), 0.5, 0.5);

            detector.detect(img2, keypoint1);

            bowDE.compute(img2, keypoint1, bowDescriptor1);

            trainingData.push_back(bowDescriptor1);

            labels.push_back((float)num);
        }
    }

    /*第四步，训练SVM得到分类模型*/
    printf("SVM training...\n"); 
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);
	CvSVM svm;

	bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	svm.save(".\\svm-classifier1.xml");

	delete[] filename;
    printf("bag-of-features training done!\n");
}

//字典文件、SVM训练文件读入内存
void TrainingDataInit()
{
	FileStorage fs(".\\dictionary1.yml", FileStorage::READ);
	Mat dictionary;
	fs["vocabulary"] >> dictionary;
	fs.release();

	bowDE.setVocabulary(dictionary);

	svm.load(".\\svm-classifier1.xml");

}

//实现发票图像的分类
int invoice_classify(Mat& img)
{
    Mat img2 = img.clone();
	resize(img2, img2, Size(), 0.5, 0.5);
    cvtColor(img2, img2, CV_RGB2GRAY);
	SiftDescriptorExtractor detector;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	Mat img_keypoints_2;

	detector.detect(img2, keypoint2);

	bowDE.compute(img2, keypoint2, bowDescriptor2);

	int it = svm.predict(bowDescriptor2);

	return it;
}



void TestClassify()
{
    int total_count = 0;
    int right_count = 0;
    string tag;
    for (int num = 1; num < 30; num++)
    {
        for (int i = 1; i < 30; i++)
        {
            char path[128] = { 0 };
            sprintf(path, ".\\test\\%d\\%d.jpg", num, i);
            Mat img = imread(path);
            if (img.empty())
            {
                continue;
            }
            int type = invoice_classify(img);
            if (type == -1)
            {
                printf("reject image %s\n", path);
                continue;
            }

            total_count++;
            
            if (num == type)
            {
                tag = "CORRECT";
                right_count++;
            }
            else
            {
                tag = "WRRONG";
            }
            printf("[%s]  label: %d   predict: %d, %s\n", path, num, type, tag.c_str());
        }
    }

    printf("total image:%d  acc:%.2f\n", total_count,(float)right_count/total_count);

}
