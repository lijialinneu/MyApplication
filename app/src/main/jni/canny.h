//
// Created by lijialin on 2017/10/20.
//

/* Header for class canny */

#include<jni.h>


#ifndef MYAPPLICATION_CANNY_H
#define MYAPPLICATION_CANNY_H

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     canny
 * Method:    canny
 * Signature: ([III)[I
 */
JNIEXPORT jintArray JNICALL Java_com_example_lijialin_myapplication_OpenCVCanny_canny (JNIEnv *, jclass, jintArray, jint, jint);

#ifdef __cplusplus
}
#endif
#endif