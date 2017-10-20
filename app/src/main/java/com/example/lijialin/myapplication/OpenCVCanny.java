package com.example.lijialin.myapplication;

/**
 * Created by lijialin on 2017/10/20.
 */

public class OpenCVCanny {
    static {
        System.loadLibrary("OpenCV"); // 加载编译好的.so动态库
    }

    /**
     * 声明native方法，调用OpenCV的边缘检测
     *
     * @param buf 图像
     * @param w 宽
     * @param h 高
     * @return 边缘图
     */
    public static native int[] canny(int[] buf, int w, int h);
}
