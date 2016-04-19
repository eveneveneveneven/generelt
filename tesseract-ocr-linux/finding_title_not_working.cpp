#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;

#define INPUT_FILE              "test5.jpg"
#define OUTPUT_FOLDER_PATH      string("")

int main()
{
    Mat large = imread(INPUT_FILE);
    Mat rgb;
    //pyrDown(large, rgb);
    Mat small;
    cvtColor(large, rgb, CV_BGR2GRAY);
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(rgb, grad, MORPH_GRADIENT, morphKernel);
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int i=0;
    ofstream results;
    results.open("img/Results.txt");
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])

    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
        float ratio = (float)rect.height/rect.width;
        stringstream s;
        if (r > .20 && (rect.height > 80 && rect.width > 30) && ratio <1.7 && ratio > 1.1){
            i+=1;
            s<<i;
            rect.height = rect.height/4;
            rectangle(large, rect, Scalar(0, 255, 0), 2);
            small = rgb(rect);
            imwrite(OUTPUT_FOLDER_PATH +string("small.jpg"), small);
            imwrite(OUTPUT_FOLDER_PATH + string("img/small")+s.str()+string(".jpg"), small);
            printf("%f\n", ratio);
            printf("%d\n", i);
            char *outText;

            tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
            // Initialize tesseract-ocr with English, without specifying tessdata path
            if (api->Init(NULL, "eng")) {
                fprintf(stderr, "Could not initialize tesseract.\n");
                exit(1);
            }

            // Open input image with leptonica library
            Pix *image = pixRead("small.jpg");
            api->SetImage(image);
            // Get OCR result
            outText = api->GetUTF8Text();
            printf("OCR output:\n%s", outText);

            results << "Image"+s.str() + "\n\n";
            results << outText;
            // Destroy used object and release memory
            api->End();
            delete [] outText;
            pixDestroy(&image);
        }
    }

            results.close();
    imwrite(OUTPUT_FOLDER_PATH + string("rgb2.jpg"), rgb);
    imwrite(OUTPUT_FOLDER_PATH +string("grid.jpg"), large);
    return 0;
}