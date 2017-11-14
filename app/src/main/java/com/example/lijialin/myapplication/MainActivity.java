package com.example.lijialin.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 获取图片，使用边缘检测提取图像的边缘
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.mipmap.building);
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();


        int[] pix = new int[width * height];
        bitmap.getPixels(pix, 0, width, 0, 0, width, height);


        int[] resultPixels = OpenCVCanny.canny(pix, width, height);

        Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_4444);
        resultBitmap.setPixels(resultPixels, 0, width, 0, 0, width, height);

        // 将边缘图显示出来
        ImageView view = (ImageView) findViewById(R.id.resultView);
        view.setImageBitmap(resultBitmap);
        view.setVisibility(View.VISIBLE);
    }
}
