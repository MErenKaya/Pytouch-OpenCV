#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main()
{
    vector<string> classes = {
        "AnnualCrop",
        "Forest",
        "Highway",
        "River"
    };

    Net net = readNetFromONNX("model.onnx"); 

    if (net.empty())
    {
        cout << "Model yuklenemedi!" << endl;
        return -1;
    }

    Mat image = imread("..."); //Input Image Path

    if (image.empty())
    {
        cout << "Resim yuklenemedi!" << endl;
        return -1;
    }

    Mat blob = blobFromImage(image,1.0 / 255.0,Size(224, 224),Scalar(),true,false);


    float mean[] = { 0.485f, 0.456f, 0.406f };
    float std[] = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                blob.ptr<float>(0, c, y)[x] =
                    (blob.ptr<float>(0, c, y)[x] - mean[c]) / std[c];
            }
        }
    }

    net.setInput(blob);
    Mat output = net.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    // Konsola da yazdıralım
    cout << "Tahmin: " << classes[classId] << " (" << confidence << ")" << endl;

    Mat displayImage;
    int targetWidth = 800;
    int targetHeight = 600; 


    double scaleX = (double)targetWidth / image.cols;
    double scaleY = (double)targetHeight / image.rows;

    double scale = max(scaleX, scaleY);

    resize(image, displayImage, Size(), scale, scale, INTER_LINEAR);

    stringstream ss;
    ss << fixed << setprecision(2) << confidence * 100;
    string labelText = classes[classId] + " : %" + ss.str();

    double fontScale = 0.55;
    int thickness = 1;
    int baseline = 0;

    Size textSize = getTextSize(labelText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

    int margin = 10;
    Point textOrg(margin, textSize.height + margin + 5); 


    Mat overlay;
    displayImage.copyTo(overlay);

    Point rectStart(textOrg.x - 5, textOrg.y - textSize.height - 5);
    Point rectEnd(textOrg.x + textSize.width + 5, textOrg.y + baseline + 5);

    rectangle(overlay, rectStart, rectEnd, Scalar(0, 0, 0), FILLED);

    float alpha = 0.7f;
    addWeighted(overlay, alpha, displayImage, 1.0 - alpha, 0, displayImage);

    putText(displayImage,labelText,textOrg,FONT_HERSHEY_SIMPLEX,fontScale,Scalar(0, 255, 0),thickness,LINE_AA);

    string windowName = "Derin Ogrenme Tahmin";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, displayImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}