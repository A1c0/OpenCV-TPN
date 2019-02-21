#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cv;
using namespace std;

VideoCapture capture;

void onTrackbarSlide(int count, void* data) {
    capture.set(CAP_PROP_POS_FRAMES, count);
}

double getCosAlKashi (const Point &A, const Point &B, const Point &C);

int main() {
    std::string winName = "video.MOV";
    capture = VideoCapture("video/video.MOV");
    int value = 0;

    Mat frame, frameGray, canny_out, dilation_out, shape_out, shape;

    if (!capture.isOpened()) return -1;
    namedWindow(winName, WINDOW_AUTOSIZE);
    int numberOfFrames = int(capture.get(CAP_PROP_FRAME_COUNT));
    createTrackbar("Progression", winName, &value, numberOfFrames, onTrackbarSlide);

    while (true) {

        capture >> frame;
        if (frame.empty()) return -1;

        resize(frame, frame, Size(600, 400), 0.0, 0.0, INTER_LINEAR);

        // Conversion l'image en niveau de gris
        cvtColor( frame, frameGray, COLOR_RGB2GRAY );

        // Application du filtre de canny;
        Canny( frameGray, canny_out, 80, 160, 3);

        // Faire une dilatation
        Mat dilatation_element = getStructuringElement( MORPH_ELLIPSE, Size( 7, 7 ), Point( 3, 3));

        dilate( canny_out, dilation_out, dilatation_element );

        // Trouver les contours
        RNG rng(12345);
        vector<vector<Point> > contours;
        vector<Point> courbe;
        vector<Vec4i> hierarchy;
        findContours( dilation_out, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

        // Gestions des contours
        shape_out = Mat::zeros( dilation_out.size(), CV_8UC3 );
        for( size_t i = 0; i< contours.size(); i++ )
        {
            double perimetre = arcLength(contours[i], true);
            approxPolyDP(contours[i], courbe, 0.02*perimetre, true);
            if (contourArea(courbe) > 5000 && isContourConvex(courbe)) {
                Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                drawContours(shape_out, contours, static_cast<int>(i), color, 4, 8, hierarchy, 0, Point());
                if (courbe.size() == 3) // c'est un triangle
                    cv::putText(shape_out, "TRIANGLE", courbe[0], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0));
                if (courbe.size() == 4) { // c'est peut être un rectangle
                    double cos1 = getCosAlKashi(courbe[2], courbe[0], courbe[1]);
                    double cos2 = getCosAlKashi(courbe[3], courbe[1], courbe[2]);
                    // or si un quadrilatère a deux angle droit alors , tous ces angles sont droits
                    if (cos1 < 0.1 && cos1 > -0.1 && cos2 < 0.1 && cos2 > -0.1) // c'est un rectangle
                        cv::putText(shape_out, "RECTANGLE", courbe[0], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0));
                }
            }

        }

        // Affichages
        imshow(winName, shape_out);

        if (waitKey(1) == 'q') break;
        int nextFrame = static_cast<int>(capture.get(CAP_PROP_POS_FRAMES));
        if (nextFrame == numberOfFrames) {
            nextFrame = 0;
            setTrackbarPos("Progression", winName, nextFrame);
            std::cout << "APPUYER SUR UNE TOUCHE POUR DEMARRER LA VIDEO" << std::endl;
            waitKey(0);
        }
        setTrackbarPos("Progression", winName, nextFrame);

    }
    capture.release();
    destroyAllWindows();

    return 0;

}

double getCosAlKashi (const Point &A, const Point &B, const Point &C) {
    double CA = cv::norm(Mat(A), Mat(C));
    double CB = cv::norm(Mat(B), Mat(C));
    double AB = cv::norm(Mat(B), Mat(A));

    return (pow(CA, 2) + pow(CB, 2) - pow(AB, 2))/(2*CA*CB);
}