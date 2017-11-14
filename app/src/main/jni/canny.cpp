//
// Created by lijialin on 2017/10/20.
//

#include "canny.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>
#include <iostream>
#include "Line.h"
// #include <hash_set>

#include <ext/hash_set>


#define LOG_TAG    "asdf"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__) // 定义LOGD类型
#define MAX 1.7976931348623158e+308

using namespace std;
using namespace cv;
using namespace __gnu_cxx;

/***************************** 函数声明部分 start ***********************************/

double solution(string path1, string path2);       // 程序的入口，参数为两个图片的路径
vector<Vec4f> operation(string path, Mat image); // 对输入的图像进行直线检测
vector<Line> createLine(vector<Vec4f> lines);    // 构造直线
bool canCluster(Line l1, Line l2, int th);       // 判断能否聚合或连接
bool isPointNear(Point p1, Point p2, double th); // 判断两个点是否接近
double distanceBetweenLine(Line l1, Line l2);    // 估计两条直线间的距离
int isConnect(Line l1, Line l2, int th);         // 返回连接的类型

vector<Line> connectLines(vector<Line> lines, int th, Mat dst); // 连接直线
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst); // 聚合直线
double lineDiff(vector<Line> line1, vector<Line> line2); // 计算两条直线的差距
double pointDistance(Point p1, Point p2);       // 计算两个点的距离
vector<vector<Line> > makePair(vector<Line> lineSet1,
	vector<Line>lineSet2, int th);              // 两组直线进行配对
double match(vector<Vec4f> lines1, vector<Vec4f> lines2,
	InputArray m1, InputArray m2);              // 计算两组直线的匹配度
double getAngle(double k1, double k2);          // 计算两条直线夹角
double calculateMean(vector<vector<double> > m); // 计算矩阵的相似度
double calculateCorr2(vector<vector<double> > m1,
	vector<vector<double> > m2);

/***************************** 函数声明部分 end *************************************/



// extern  "C" {








// JNIEnv 代表java环境，通过这个参数可以调用java中的方法

JNIEXPORT jintArray JNICALL Java_com_example_lijialin_myapplication_OpenCVCanny_canny(JNIEnv *env, jclass obj, jintArray buf, int w, int h);

JNIEXPORT jintArray JNICALL Java_com_example_lijialin_myapplication_OpenCVCanny_canny(JNIEnv *env, jclass obj, jintArray buf, int w, int h) {

    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, false); // 读取输入参数
    if (cbuf == NULL) {
        return 0;
    }

    Mat image(h, w, CV_8UC4, (unsigned char*) cbuf); // 初始化一个矩阵（图像）4通道的图像
    cvtColor(image, image, COLOR_BGR2GRAY); // 转为灰度图
    GaussianBlur(image, image, Size(5,5), 0, 0); // 高斯滤波
    Canny(image, image, 30, 100, 3); // 边缘检测

    // TODO 在这里添加其他方法
    Mat image2(image.size(), image.type());

    // Detect the lines
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD, 0.80);
    vector<Vec4f> lines_std;
    ls->detect(image, lines_std);

    // LOGD("channels = %d", image.channels());

    ls->drawSegments(image2, lines_std);
    cvtColor(image2, image2, COLOR_BGR2GRAY); // 转为灰度图


    double rate = match(lines_std, lines_std, image, image);
    LOGD("rate = %f", rate);



    int* outImage = new int[w * h];
    int n = 0;

    for(int i = 0; i < h; i++) {
        uchar* data = image2.ptr<uchar>(i);
        for(int j = 0; j < w; j++) {
            if(data[j] > 0) {
                outImage[n++] = 0;
            }else {
                outImage[n++] = -1;
            }
        }

    }

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, outImage);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;

}




/**
 * 对向量中的每一条直线构造Line对象
 * 返回一个向量集合，集合里的元素是Line对象
 * 输入：包含Vec4f类型直线的向量
 * 输出：包含Line对象的向量
 */
vector<Line> createLine(vector<Vec4f> lines) {
	vector<Line> LineSet;
	size_t len = lines.size();
	for (int i = 0; i < len; i++) {
		Line *line = new Line(lines[i]);
		LineSet.push_back(*line);
	}
	vector<Vec4f>().swap(lines);
	return LineSet;
}



/**
 * 过滤短小直线
 * TODO 过滤孤立的直线
 * 输入：包含Line对象的向量
 * 输出：清除短小直线后的Line对象向量
 */
vector<Line> cleanShort(vector<Line> lines) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	if (length == 0) return lines;

	// 计算average长度
	double sum = 0;
	for (int i = 0; i < length; i++) {
		Line line = lines[i];
		sum += line.length;
	}
	double avg = sum / length;

	// 过滤短小的直线
	for (int i = 0; i < length; i++) {
		Line line = lines[i];
		if (line.length >= avg) {
			(*result).push_back(line);
		}
	}
	vector<Line>().swap(lines);
	return *result;
}


/**
 * 判断两条直线是否具备聚合条件
 * 判断规则：斜率相近，直线间距相近，则可以聚合
 * 输入：两条直线（Line），和阈值
 * 输出：bool型，true表示两条直线可以聚合，false表示不能聚合
 */
bool canCluster(Line l1, Line l2, int th) {
	return abs(l1.k - l2.k) <= 0.3 &&  // 斜率差的绝对值小于0.3
		((l1.k > 0 && l2.k > 0) || (l1.k < 0 && l2.k < 0)) &&  // 斜率同号
		distanceBetweenLine(l1, l2) < th; // 距离较近
}


/**
 * 计算两个点的距离
 * 输入：两个点（Point类型的对象）
 * 输出：两个点之间的距离，double类型
 */
double pointDistance(Point p1, Point p2) {
	return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}


/**
 * 原方法：利用点到直线距离，估算两条直线的距离
 * 改进方法：直线间中点的距离
 * 输入：两条直线（Line对象）
 * 输出：直线之间的距离， double类型
 */
double distanceBetweenLine(Line l1, Line l2) {
	return pointDistance(l1.mid, l2.mid);
}


/**
 * 判断两个点是否相近
 * x，y间距均小于阈值
 * 输入：两个点（Point对象），阈值double类型
 * 输出：bool类型，true表示两个点离得很近，false表示离得远
 */
bool isPointNear(Point p1, Point p2, double th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/**
 * 判断直线首尾是否相接，并返回相连的类型
 * 输入：两条直线（Line对象），阈值 int
 * 输出：一个int型，表示的类型如下：
 *    0：不相连
 *    1：l1的end   和 l2的start   相连
 *    2：l1的end   和 l2的end     相连
 *    3：l1的start 和 l2的start   相连
 *    4：l1的start 和 l2的end     相连
*/
int isConnect(Line l1, Line l2, int th) {
	double len = max(l1.length, l2.length);
	int status = 0;
	if (isPointNear(l1.end, l2.start, th) && !isPointNear(l1.start, l2.end, len)) {
		status = 1;
	}else if (isPointNear(l1.end, l2.end, th) && !isPointNear(l1.start, l2.start, len)) {
		status = 2;
	}else if (isPointNear(l1.start, l2.start, th) && !isPointNear(l1.end, l2.end, len)) {
		status = 3;
	} else if (isPointNear(l1.start, l2.end, th) && !isPointNear(l1.end, l2.start, len)) {
		status = 4;
	}
	return status;
}


/**
 * 产生首尾相连的长直线
 * 输入：两条直线（Line对象），类型 int
 * 输出：一条连接后的直线（Line对象）
 */
Line createConnectLine(Line l1, Line l2, int type) {
	Line l = Line();
	if (type == 1) {
		l.start = l1.start;
		l.end = l2.end;
	}else if (type == 2) {
		l.start = l1.start;
		l.end = l2.start;
	}else if (type == 3) {
		l.start = l1.end;
		l.end = l2.end;
	}else if (type == 4) {
		l.start = l1.end;
		l.end = l2.start;
	}
	l.length = l.getLength();
	l.mid = l.getMidPoint();
	return l;
}


/**
 * 直线连接，如果两个直线首尾能相接，则连接成一条直线
 * 输入：包含Line对象的向量，阈值，矩阵Mat（作用就是测试的时候画图用）
 * 输出：包含Line对象的向量
 */
vector<Line> connectLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();

	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		bool useless = true;
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2, th)) { // 如果具备聚合条件
				int type = isConnect(line1, line2, th); // 计算类型
                if (type != 0) {  // 如果是连接型
					useless = false;
					Line tmp = createConnectLine(line1, line2, type);
					(*result).push_back(tmp);
					break;
				}
			}
		}
		if (useless) {
			(*result).push_back(line1);
		}
	}
	vector<Line>().swap(lines);
	return *result;
}


/**
 * 直线聚合函数,聚合的原则：
 * 如果两个直线的起点和终点相似，则保留那条长直线
 * 由于直线数量不多，采用暴力求解的方法，时间复杂度O(n2)
 * 输入：包含Line对象的向量，阈值int，矩阵Mat（作用就是测试的时候画图用）
 * 输出：包含Line对象的向量
 */
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	hash_set<int> set;
	hash_set<int>::iterator pos;

	for (int i = 0; i < length; i++) {
		pos = set.find(i);
		if (pos != set.end()) { // 如果存在
			continue;
		}
		Line line1 = lines[i];
		bool useless = true;
		for (int j = i; j < length; j++) {
			pos = set.find(j);
			if (pos != set.end()) { // 如果存在
				continue;
			}
			Line line2 = lines[j];
			if (canCluster(line1, line2, th)) { // 如果具备聚合条件
				set.insert(i);
				set.insert(j);
				useless = false;
				if (line1.length >= line2.length) {
					(*result).push_back(line1);
				}else {
					(*result).push_back(line2);
				}
				break;
			}
		}
	    if (useless) {
			(*result).push_back(line1);
		}
	}
	vector<Line>().swap(lines);
	return *result;
}

/**
 * 计算两条直线的差距，算法如下：
 *  - 计算长度差距
 *  - 计算斜率的差距
 * 返回：长度差距 * 斜率差距
 * 输入：两条直线（Line对象）
 * 输出：直线之间的差距，double类型
 */
double lineDiff(Line line1, Line line2) {
	return abs(line1.length - line2.length) * abs(line1.k - line2.k);
}

/**
 * 两个直线组中的直线，进行两两配对，或者说匹配
 * 输入：两组直线，包含Line对象的向量，阈值 int
 * 输出：一个n*2的二维向量，n表示匹配的对儿数
 */
vector<vector<Line> > makePair(vector<Line> lineSet1, vector<Line>lineSet2, int th) {
	hash_set<int> set;
	hash_set<int>::iterator pos;
	vector<vector<Line> > pairSet;
	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	for (int i = 0; i < length1; i++) {
		Line line1 = lineSet1[i];
		int bestFriendId = -1; // 最佳配对直线的id
		double minDiff = MAX;  // 两条直线最小的差距
		for (int j = 0; j < length2; j++) {
			pos = set.find(j);
			if (pos != set.end()) { // 如果存在
				continue;
			}
			Line line2 = lineSet2[j];
			// 如果直线1,2的斜率相近，长度相近，位置相近，则配对
			// 这里偷个懒，直接用canCluster()函数判断，如果能聚合，也就能配对
			if (canCluster(line1, line2, th * 2)) {
				double diff = lineDiff(line1, line2);
				if (diff < minDiff) {
					minDiff = diff;
					bestFriendId = j;
				}
			}
		}
		// 找到最佳配对的直线后，存储到二维向量中
		if (bestFriendId != -1) {
			set.insert(bestFriendId);
			vector<Line> pair;
			Line bestFriendLine = lineSet2[bestFriendId];
			pair.push_back(line1);
			pair.push_back(bestFriendLine);
			pairSet.push_back(pair);
		}
	}
	vector<Line>().swap(lineSet1); // 回收内存
	vector<Line>().swap(lineSet2);
	return pairSet;
}


/**
 * 计算两组直线的匹配度
 * 输入：两个图像的两组直线 lines1，lines2
 * 输出：匹配度，double类型
 *
 * 算法步骤如下：
 * 1. 计算每组直线的斜率，计算斜率阈值TK、距离阈值TP
 * 2. 根据斜率、距离的差值是否满足阈值，找到最佳匹配直线对
 * 3. 计算每组中的直线与本组中的其他直线之间的夹角
 * 4. 计算夹角矩阵之间的相似度，并把这个相似度，作为直线的匹配度，返回
 * TODO: 通过直线构造三种直线组合，利用直线组合还原高级特征，通过高级特征图匹配
 */
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, InputArray m1, InputArray m2) {

	// Step1 创建直线
	vector<Line> lineSet1 = createLine(lines1);
	vector<Line> lineSet2 = createLine(lines2);

	vector<Vec4f>().swap(lines1); // 回收内存
	vector<Vec4f>().swap(lines2);

	int threshold = 8; // 阈值【5-10】
	Mat dst1(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	Mat dst2(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255,255,255));

	// Step2 删除较短的直线 (可选)
	lineSet1 = cleanShort(lineSet1);
	lineSet2 = cleanShort(lineSet2);

	// Step3 先进行直线的连接，然后聚合直线
    lineSet1 = connectLines(lineSet1, threshold, dst1); // 连接
	lineSet2 = connectLines(lineSet2, threshold, dst2);

	lineSet1 = clusterLines(lineSet1, threshold, dst1); // 聚合
	lineSet2 = clusterLines(lineSet2, threshold, dst2);

	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	// Step4. 从第一张图中选择一条直线，然后遍历第二张图，找到最佳的配对直线
	vector<vector<Line> > pairSet = makePair(lineSet1, lineSet2, threshold);
	size_t pairLen = pairSet.size(); // 有多少对直线

	// Step 5. 计算直线与其他直线的夹角，构造夹角矩阵
	vector<vector<double> > angleList1, angleList2;
	for (int i = 0; i < pairLen; i++) {
		vector<Line> v1 = pairSet[i];
		vector<double> angle1, angle2;

		for (int j = i + 1; j < pairLen; j++) {
			vector<Line> v2 = pairSet[j];
			 angle1.push_back(getAngle(v1[0].k, v2[0].k));
			 angle2.push_back(getAngle(v1[1].k, v2[1].k));
		}
		angleList1.push_back(angle1);
		angleList2.push_back(angle2);

		vector<double>().swap(angle1); // 回收内存
		vector<double>().swap(angle2);
	}

	// 然后计算夹角矩阵的相似度
	double rate = calculateCorr2(angleList1, angleList2);

	if (length1 != length2) {
		rate *= (double) min(length1, length2) / max(length1, length2);
	}
	return rate;
}


/**
 * 计算两条直线夹角
 * 输入：斜率k1和k2，double类型
 * 输出：反正切角度，double类型
 */
double getAngle(double k1, double k2) {
	return atan(abs(k2 - k1) / (1 + k1 * k2));
}


/**
 * 计算两个矩阵的相似度
 * matlab中的corr2()函数，好麻烦
 * 输入：二维向量
 * 输出：二维向量的均值
 */
double calculateMean(vector<vector<double> > m) {
	double sum = 0.0;
	int num = 0;
	size_t rows = m.size();
	for (int i = 0; i < rows; i++) {
		size_t cols = m[i].size();
		for (int j = 0; j < cols; j++) {
			sum += m[i][j];
			num++;
		}
	}
	return sum / num;
}


/**
 * corr2函数，计算两个矩阵的相关系数
 * 计算结果被归一化在[0,1]区间内，数值越大说明相似度越高
 * 输入：两个二维向量
 * 输出：它们的相关系数，double类型
 */
double calculateCorr2(vector<vector<double> > m1, vector<vector<double> > m2) {

	double mean1 = calculateMean(m1);
	double mean2 = calculateMean(m2);

	// 增加一条判断，如果平均值相似，则认为是两张完全相同的图像
	if (abs(mean1 - mean2) <= 1e-6 ) { return 1.0; }

	//计算分子
	double numerator = 0;
	size_t len = m1.size();
	for (size_t i = 0; i < len; i++) {
		size_t len1 = m1[i].size();
		for (size_t j = 0; j < len1; j++) {
			numerator += (m1[i][j] - mean1) * (m2[i][j] - mean2);
		}
		for (size_t j = len1; j <= len1; j++) {
			numerator += mean1 * mean2;
		}
	}

	//计算分母 sqrt(pow(x,2) + pow(y,2));
	double d1 = 0;
	double d2 = 0;
	for (size_t i = 0; i < len; i++) {
		size_t len1 = m1[i].size();
		for (size_t j = 0; j < len1; j++) {
			d1 += pow((m1[i][j] - mean1), 2);
			d2 += pow((m2[i][j] - mean2), 2);
		}
		for (size_t j = len1; j <= len; j++) {
			d1 += pow(mean1, 2);
			d2 += pow(mean2, 2);
		}
	}
	double denominator = sqrt(d1) * sqrt(d2);

	if (numerator == 0) return 0.0;
	return numerator / denominator;

}

// }