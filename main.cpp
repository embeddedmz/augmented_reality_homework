#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;


//camera settings
const int camera_width = 640;
const int camera_height = 480;
const int virtual_camera_angle = 60;
unsigned char bkgnd[camera_width*camera_height * 3];

// The instensity value threshold. If the intensity is more than it -> assign it to max_value.
int threshold_value = 100;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

RNG rng(12345);

void MyLine( Mat img, Point start, Point end );
void MyCircle( Mat img, Point center );

int sampleSubPix(const cv::Mat &pSrc, const cv::Point2f &p)
{
	int x = int(floorf(p.x));
	int y = int(floorf(p.y));

	if (x < 0 || x >= pSrc.cols - 1 ||
		y < 0 || y >= pSrc.rows - 1)
		return 127;

	int dx = int(256 * (p.x - floorf(p.x)));
	int dy = int(256 * (p.y - floorf(p.y)));

	unsigned char* i = (unsigned char*)((pSrc.data + y * pSrc.step) + x);
	int a = i[0] + ((dx * (i[1] - i[0])) / 256);
	i += pSrc.step;
	int b = i[0] + ((dx * (i[1] - i[0])) / 256);
	return a + ((dy * (b - a)) / 256);
}

vector<Point2f> sampleEquidistantPointsOnLine(Point2f beginPoint, Point2f endPoint, int pointAmount)
{
    vector<Point2f> resultPointsArray(pointAmount);
    double step_amount = pointAmount + 1;

    double step_x = (endPoint.x - beginPoint.x) / step_amount;
    double step_y = (endPoint.y - beginPoint.y) / step_amount;

    double x = beginPoint.x;
    double y = beginPoint.y;

    for(int step = 0; step < pointAmount; step++)
    {
        x += step_x;
        y += step_y;
        resultPointsArray[step].x = x;
        resultPointsArray[step].y = y;
    }

    return resultPointsArray;
}

vector<vector<Point2f>> sampleEquidistantPointsOnLineAndNeighbourhood(Point2f beginPoint, Point2f endPoint, int pointAmount, double neighbour_distance=0)
{
    // Predefine the result array:
    // 3 vectors: upper line points, line itself points, lower line points.
    vector<vector<Point2f>> resultPointsArray(3);

    // Computing the h: distance between points on the line.
    double step_amount = pointAmount + 1;
    double step_x = (endPoint.x - beginPoint.x) / step_amount;
    double step_y = (endPoint.y - beginPoint.y) / step_amount;

    Point2f lineDistanceVector = Point2f(step_x, step_y);
    Point2f perpendicularNormalizedVector = Point2f(step_y, -step_x);

    perpendicularNormalizedVector = perpendicularNormalizedVector * (1 / norm(perpendicularNormalizedVector));

    // Distance between points on one line.
    double h = norm(lineDistanceVector);

    // If the neighbour distance isn't specified, use h/2 for neighbourhood lines.
    if(neighbour_distance == 0)
    {
        neighbour_distance = h / 2;
    }

    // Vector specifying perpendicular direction for neighbours and the
    // neighbour distance.
    Point2f neighbourOffsetVector = perpendicularNormalizedVector * neighbour_distance;

    Point2f currentBeginPoint, currentEndPoint;

    for(int lineOffset = -1, lineNumber=0; lineOffset <= 1; lineOffset++, lineNumber++)
    {
        currentBeginPoint = beginPoint + neighbourOffsetVector * lineOffset;
        currentEndPoint = endPoint + neighbourOffsetVector * lineOffset;
        resultPointsArray[lineNumber] = sampleEquidistantPointsOnLine(currentBeginPoint, currentEndPoint, pointAmount);
    }

    return resultPointsArray;
}

pair<vector<vector<vector<Point2f>>>, vector<Mat>>  getStripes(Point2f beginPoint, Point2f endPoint, int pointAmount, Mat &pSrc)
{

    double step_amount = pointAmount + 1;

    // Steps to the next point on the line
    double step_x = (endPoint.x - beginPoint.x) / step_amount;
    double step_y = (endPoint.y - beginPoint.y) / step_amount;

    Point2f lineDistanceVector = Point2f(step_x, step_y);
    Point2f perpendicularVector = Point2f(step_y, -step_x);

    Point2f perpendicularNormalizedVector = perpendicularVector * (1 / norm(perpendicularVector));
    Point2f lineDistanceNormalizedVector = lineDistanceVector * (1 / norm(lineDistanceVector));

    // Distance between points on one line.
    double h = norm(lineDistanceVector);

    // Compute amount of pixels needed for the perpendicular sampling.
    // It should be uneven.
    int amountOfPixels = (int) h;

    if(amountOfPixels % 2 == 0)
    {
        amountOfPixels += 1;
    }

    int stepForEachSide = amountOfPixels / 2;

    // Matrix to store the coords of related stripes.
    // Vector of matrices of Point2f coordinates.
    vector<vector<vector<Point2f>>> resultPointsArray(pointAmount, vector<vector<Point2f>>(amountOfPixels, vector<Point2f>(3)));

    Point2f neighbourOffsetVector = perpendicularNormalizedVector;

    Point2f currentBeginPoint, currentEndPoint;

    vector<Mat> stripes(pointAmount, Mat::zeros(amountOfPixels, 3, CV_8UC1));
    vector<Point2f> currentLoad;

    for(int sideShift = -1, sideShiftNumber = 0; sideShift <= 1; sideShift++, sideShiftNumber++)
    {
        for(int lineOffset = -stepForEachSide, lineNumber=0; lineOffset <= stepForEachSide; lineOffset++, lineNumber++)
        {
            // Get all the coordinates by shifting left and right on 1px on the line
            // And by shifting on the needed amount in the perpendicular direction
            currentBeginPoint = beginPoint + neighbourOffsetVector * lineOffset + lineDistanceNormalizedVector * sideShift;
            currentEndPoint = endPoint + neighbourOffsetVector * lineOffset + lineDistanceNormalizedVector * sideShift;

            // Get points for current shift
            currentLoad = sampleEquidistantPointsOnLine(currentBeginPoint, currentEndPoint, pointAmount);
            //resultPointsArray[sideShiftNumber][lineNumber] = sampleEquidistantPointsOnLine(currentBeginPoint, currentEndPoint, pointAmount);
            //currentLoad = sampleEquidistantPointsOnLine(currentBeginPoint, currentEndPoint, pointAmount);

            for(int pointNumber=0; pointNumber < pointAmount; pointNumber++)
            {
                // Save the coordinates of points and their intensity values into stripes.
                resultPointsArray[pointNumber][lineNumber][sideShiftNumber] = currentLoad[pointNumber];
                stripes[pointNumber].at<uchar>(lineNumber, sideShiftNumber) = sampleSubPix(pSrc, currentLoad[pointNumber]);
            }
        }
    }

    return pair<vector<vector<vector<Point2f>>>, vector<Mat>> (resultPointsArray, stripes);

}

double calcParabolaExtremumPosition(double y1, double y2, double y3, double x1=-1, double x2=0, double x3=1)
{
	double denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
	double A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	double B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	double C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
	double extremumPosition = -B / (2*A);

	return extremumPosition;
}

Point2f findLinesIntersectionPoint(Point2f firstLinePoint, Point2f firstLineVector, Point2f secondLinePoint, Point2f secondLineVector)
{

    float firstLineVectorMultiplier;
    Point2f intersectionPoint;

    float nominator = secondLinePoint.x * secondLineVector.y - firstLinePoint.x*secondLineVector.y +
                        firstLinePoint.y*secondLineVector.x - secondLinePoint.y*secondLineVector.x;

    float denominator = firstLineVector.x*secondLineVector.y - firstLineVector.y*secondLineVector.x;

    firstLineVectorMultiplier = nominator / denominator;

    intersectionPoint = firstLinePoint + firstLineVectorMultiplier * firstLineVector;

    return intersectionPoint;

}

vector<Point2f> refineBorderPosition( vector<vector<vector<Point2f>>> &coords, vector<Mat> &stripes)
{
    vector<Point2f> refinedLocations(stripes.size());
    Mat sobelResult;
    Mat sobelResultMiddlestripe;
    int stripeHeight = stripes[0].size().height;

    for(int pointNumber = 0; pointNumber < stripes.size(); pointNumber++)
    {
        // Replicate to make things on the borders constant.
        Sobel( stripes[pointNumber], sobelResult, CV_64F, 0, 1, 3, 1, 0, BORDER_REPLICATE);

        //cout << sobelResult << endl;

        // We take only the middle stripe.
        // We used the ones on the border only in order to get good derivatives.
        sobelResultMiddlestripe = sobelResult(Rect(1, 0, 1, sobelResult.size().height));

        //cout << sobelResultMiddlestripe << endl;

        Point min_loc, max_loc;
        double min, max;
        int maxElementPosition;

        minMaxLoc(sobelResultMiddlestripe, &min, &max, &min_loc, &max_loc);

        maxElementPosition = max_loc.y;

        //cout << maxElementPosition << ", " << max << endl;

        // If the element is on the border we can't approximate.
        // Therefore, we return the original position.
        if(maxElementPosition == 0 || maxElementPosition == (stripeHeight-1))
        {
            refinedLocations[pointNumber] = coords[pointNumber][maxElementPosition][1];
            continue;
        }

        Point2f centerElementCoord = coords[pointNumber][maxElementPosition][1];
        Point2f neighbourElementTopCoord = coords[pointNumber][maxElementPosition-1][1];
        Point2f neighbourElementBottomCoord = coords[pointNumber][maxElementPosition+1][1];

        Point2f neighbourVector = neighbourElementBottomCoord - centerElementCoord;

        double y1 = sobelResult.at<double>(maxElementPosition-1, 1);
        double y2 = sobelResult.at<double>(maxElementPosition, 1);
        double y3 = sobelResult.at<double>(maxElementPosition+1, 1);

        double exactBorderRelativePosition = calcParabolaExtremumPosition(y1, y2, y3);

        // We move in the direction of parabola extremum from the center point.
        Point2f refinedPoint = centerElementCoord + exactBorderRelativePosition*neighbourVector;

        refinedLocations[pointNumber] = refinedPoint;

//    cout << (int)y1 << "," << (int)y2 << ", " << (int)y3 << endl;
//
//    cout << exactBorderRelativePosition << endl;
//
//    cout << centerElementCoord.x << ", " << centerElementCoord.y << ";" << refinedPoint.x << ", " << refinedPoint.y << endl;
    }

    return refinedLocations;
}



/* program & OpenGL initialization */
void initGL(int argc, char *argv[])
{
	// initialize the GL library
	// pixel storage/packing stuff
	glPixelStorei(GL_PACK_ALIGNMENT, 1); // for glReadPixels​
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // for glTexImage2D​
	glPixelZoom(1.0, -1.0);

	// enable and set colors
	glEnable(GL_COLOR_MATERIAL);
	glClearColor(0, 0, 0, 1.0);

	// enable and set depth parameters
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);

	// light parameters
	GLfloat light_pos[] = { 1.0f, 1.0f, 1.0f, 0.0f };
	GLfloat light_amb[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat light_dif[] = { 0.9f, 0.9f, 0.9f, 1.0f };

	// enable lighting
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
}

void initVideoStream(cv::VideoCapture &cap)
{
	if (cap.isOpened())
		cap.release();

	cap.open(0); // open the default camera
}

void display(GLFWwindow* window, const cv::Mat &img_bgr)
{
	memcpy(bkgnd, img_bgr.data, sizeof(bkgnd));

	int width0, height0;
	glfwGetFramebufferSize(window, &width0, &height0);

	// clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// draw background image
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, camera_width, 0.0, camera_height);

	glRasterPos2i(0, camera_height - 1);


    //glDrawPixels(camera_width, camera_height, GL_LUMINANCE, GL_UNSIGNED_SHORT, bkgnd);
	glDrawPixels(camera_width, camera_height, GL_BGR_EXT, GL_UNSIGNED_BYTE, bkgnd);
	//glDrawPixels(camera_width, camera_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, bkgnd);

	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
}

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat img_final;
    Mat img_f;
    namedWindow("edges",1);
    namedWindow("stripes", 1);
    for(;;)
    {
        Mat frame;
        Mat stripe;
        Mat sobel;
        vector<vector<Point> > contours;
        vector<vector<Point> > contours0;
        vector<Vec4i> hierarchy;

        cap >> frame; // get a new frame from camera
        cvtColor(frame, img_final, CV_BGR2GRAY);
        threshold( img_final, img_f, threshold_value, max_BINARY_value, threshold_type );

        /// Find contours
        findContours( img_f, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        //contours0.resize(contours.size());
        vector<Point> current_countour;
        Rect current_rect;
        double current_area;
        Point2f current_point_start;
        Point2f current_point_end;
        double x, y, step_x, step_y;
        vector<float> previousEdge(4);
        vector<float> currentEdge(4);
        pair<vector<vector<vector<Point2f>>>, vector<Mat>> result;
        vector<vector<vector<Point2f>>> stripeCoords;
        vector<Point2f> linePoints;

        for( size_t k = 0; k < contours.size(); k++ )
        {
            approxPolyDP(Mat(contours[k]), current_countour, 3, true);

            if(current_countour.size() == 4)
            {
                current_rect = boundingRect( current_countour );
                current_area = current_rect.area();
                //cout << format("%f\n", current_area);

                // The lower threshold size of rectangle
                if(current_area > 2000)
                {

                    current_point_start = current_countour[3];
                    current_point_end = current_countour[0];

                    result = getStripes(current_point_start, current_point_end, 6, img_final);
                    stripeCoords = result.first;

                    linePoints = refineBorderPosition( stripeCoords, result.second);
                    fitLine(linePoints, previousEdge, CV_DIST_L2, 0, 0.01, 0.01);

                    for(int current_edge=0; current_edge < 4; current_edge++)
                    {
                        current_point_start = current_countour[current_edge];
                        current_point_end = current_countour[(current_edge+1)%4];

                        result = getStripes(current_point_start, current_point_end, 8, img_final);
                        stripeCoords = result.first;
//
//                        for(int shiftNumber=0; shiftNumber < stripeCoords.size(); shiftNumber++)
//                        {
//                            for(int layerNumber = 0; layerNumber < stripeCoords[shiftNumber].size(); layerNumber++)
//                            {
//                                vector<Point2f> linePoints = stripeCoords[shiftNumber][layerNumber];
//                                for(int step = 0; step < linePoints.size(); step++)
//                                    MyCircle(frame, linePoints[step]);
//                            }
//
//                        }


                            linePoints = refineBorderPosition( stripeCoords, result.second);

                            fitLine(linePoints, currentEdge, CV_DIST_L2, 0, 0.001, 0.001);

                            Point2f currentLineDirection(currentEdge[0], currentEdge[1]);
                            Point2f currentLinePoint(currentEdge[2], currentEdge[3]);

                            Point2f previousLineDirection(previousEdge[0], previousEdge[1]);
                            Point2f previousLinePoint(previousEdge[2], previousEdge[3]);

                            Point2f corner = findLinesIntersectionPoint(currentLinePoint,
                                                        currentLineDirection,
                                                        previousLinePoint,
                                                        previousLineDirection);

                            Point2f beginPoint = currentLinePoint + currentLineDirection * 80;

                            Point2f endPoint = currentLinePoint + currentLineDirection * (-80);

                            cout << corner.x << ", " << corner.y << endl;

                            MyLine( frame, beginPoint, endPoint );

                            MyCircle(frame, corner);

                            previousEdge = currentEdge;

//                            cout << "size: " << linePoints.size() << endl;
//
//                            for(int step = 0; step < linePoints.size(); step++)
//                            {
//                                cout << linePoints[step].x << ":" << linePoints[step].y;
//                                MyCircle(frame, linePoints[step]);
//                            }

                    }

//                    MyLine( frame, current_countour[0], current_countour[1] );
//                    MyLine( frame, current_countour[1], current_countour[2] );
//                    MyLine( frame, current_countour[2], current_countour[3] );
//                    MyLine( frame, current_countour[3], current_countour[0] );
//                    contours0.push_back(current_countour);
                }
            }
        }


        imshow("edges", frame);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

void MyCircle( Mat img, Point center )
{
  int thickness = 2;
  int lineType = 8;

  circle( img,
      center,
      1,
      Scalar( 0, 0, 255 ),
      thickness,
      lineType );
}

void MyLine( Mat img, Point start, Point end )
{
  int thickness = 1;
  int lineType = 8;
  line( img,
    start,
    end,
    Scalar( 155, 100, 200 ),
    thickness,
    lineType );
}
