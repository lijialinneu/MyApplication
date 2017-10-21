//
// Created by lijialin on 2017/10/20.
//

#include "canny.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>

#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__) // 定义LOGD类型

using namespace cv;

extern  "C" {

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
    Canny(image, image, 50, 150, 3); // 边缘检测

    // TODO 在这里添加其他方法

    int* outImage = new int[w * h];
    int n = 0;
    for(int i = 0; i < h; i++) {
        uchar* data = image.ptr<uchar>(i);
        for(int j = 0; j < w; j++) {
            if(data[j] == 255) {
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


}
