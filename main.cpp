#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

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

typedef Point2d PVect;


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

        // We take only the middle stripe.
        // We used the ones on the border only in order to get good derivatives.
        sobelResultMiddlestripe = sobelResult(Rect(1, 0, 1, sobelResult.size().height));

        Point min_loc, max_loc;
        double min, max;
        int maxElementPosition;

        minMaxLoc(sobelResultMiddlestripe, &min, &max, &min_loc, &max_loc);

        maxElementPosition = max_loc.y;

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

vector<vector<Point2f>> detectMarkers(Mat &frame, bool visualization=false)
{

    Mat grayFrame;
    Mat thresholdedBinaryFrame;

    vector<vector<Point2f>> markersCornersPoints;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    cvtColor(frame, grayFrame, CV_BGR2GRAY);
    threshold( grayFrame, thresholdedBinaryFrame, threshold_value, max_BINARY_value, threshold_type );
    findContours( thresholdedBinaryFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<Point> currentContour;
    Rect contourBoundingRect;
    double boundingRectArea;

    for( size_t k = 0; k < contours.size(); k++ )
    {
        approxPolyDP(Mat(contours[k]), currentContour, 3, true);

        // Skip if it's not a rectangle.
        if(currentContour.size() != 4)
            continue;

        contourBoundingRect = boundingRect( currentContour );
        boundingRectArea = contourBoundingRect.area();

        // Skip if the area is too small.
        if(boundingRectArea < 2000)
            continue;

        vector<Point2f> cornersPoint(4);

        // Find the final edge in order to find intersections later.
        Point2f currentPointStart = currentContour[3];
        Point2f currentPointEnd = currentContour[0];

        pair<vector<vector<vector<Point2f>>>, vector<Mat>> result;
        vector<vector<vector<Point2f>>> stripesCoords;
        vector<Mat> stripes = result.second;
        vector<float> previousEdge(4);
        vector<float> currentEdge(4);
        vector<Point2f> linePoints;

        // Get intensity stripes and refine the border position.
        result = getStripes(currentPointStart, currentPointEnd, 20, grayFrame);
        stripesCoords = result.first;
        stripes = result.second;

        // Get the points on the edge with corrected position.
        linePoints = refineBorderPosition( stripesCoords, stripes);

        // Fit a line through the refined points.
        fitLine(linePoints, previousEdge, CV_DIST_L2, 0, 0.001, 0.001);

        for(int currentEdgeNumber=0; currentEdgeNumber < 4; currentEdgeNumber++)
        {
            currentPointStart = currentContour[currentEdgeNumber];
            currentPointEnd = currentContour[(currentEdgeNumber+1)%4];

            result = getStripes(currentPointStart, currentPointEnd, 20, grayFrame);
            stripesCoords = result.first;
            stripes = result.second;
            linePoints = refineBorderPosition( stripesCoords, result.second);

            fitLine(linePoints, currentEdge, CV_DIST_L2, 0, 0.001, 0.001);

            Point2f currentLineDirection(currentEdge[0], currentEdge[1]);
            Point2f currentLinePoint(currentEdge[2], currentEdge[3]);

            Point2f previousLineDirection(previousEdge[0], previousEdge[1]);
            Point2f previousLinePoint(previousEdge[2], previousEdge[3]);

            Point2f corner = findLinesIntersectionPoint(currentLinePoint,
                                                        currentLineDirection,
                                                        previousLinePoint,
                                                        previousLineDirection);

            cornersPoint[currentEdgeNumber] = corner;
            //cout << corner.x << ", " << corner.y << endl;

            previousEdge = currentEdge;

            if(visualization)
            {
                Point2f beginPoint = currentLinePoint + currentLineDirection * 80;
                Point2f endPoint = currentLinePoint + currentLineDirection * (-80);
                MyLine( frame, beginPoint, endPoint );
                MyCircle(frame, corner);
            }

        }

        markersCornersPoints.push_back(cornersPoint);
    }

    return markersCornersPoints;
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

	glDrawPixels(camera_width, camera_height, GL_BGR_EXT, GL_UNSIGNED_BYTE, bkgnd);

	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
}

bool checkMatrixBorderForOnes(Mat& matrixToCheck)
{
    int amountOfRows = matrixToCheck.rows;
    int amountOfCols = matrixToCheck.cols;

    // Check border columns
    for(int col = 0; col != amountOfCols; col+=amountOfCols)
        for(int row = 0; row < amountOfRows; row++)
            if(matrixToCheck.at<uchar>(col, row) != 0)
                return false;

    // Check border rows
    for(int row = 0; row != amountOfRows; row+=amountOfRows)
        for(int col = 0; col < amountOfCols; col++)
            if(matrixToCheck.at<uchar>(col, row) != 0)
                return false;


    return true;
}

Mat rotate_cw(const Mat& image, int degrees)
{
    cv::Mat res;
    switch (degrees % 360) {
        case 0:
            res = image;
            break;
        case 90:
            res = image.t();
            cv::flip(res, res, 1);
            break;
        case 180:
            cv::flip(image, res, -1);
            break;
        case 270:
            res = image.t();
            cv::flip(res, res, 0);
            break;
        default:
            cv::Mat r = cv::getRotationMatrix2D({image.cols/2.0F, image.rows/2.0F}, degrees, 1.0);
            int len = std::max(image.cols, image.rows);
            cv::warpAffine(image, res, r, cv::Size(len, len));
            break; //image size will change
    }
    return res;
}

PVect computeSpeed(PVect v, double s)
{
    // v has values in [-1;+1]
    double angle = atan2(v.y, v.x);
    return PVect(s * cos(angle), s * sin(angle));
}

void drawMargin(Mat& image, int x)
{
    line(image, Point(x, 0), Point(x, image.rows),
            Scalar(0xFF, 0xFF, 0xFF), 2, CV_AA);
}

PVect processSpeed(PVect v, double dy)
{
    double Y = v.y + dy;
    double norm = sqrt(v.x * v.x + Y * Y);
    return PVect(-v.x / norm, Y / norm); // -x for the x-bounce
}

// global variables
bool smooth = false; //< whether to smooth the camera image
bool game = false; //< whether the game is running
bool paused = false; //< running status

bool pollKeys()
{
    char c = (char) waitKey(10);
    if (c == 27) {
        // we're done, let's exit
        return false;
    }
    // possible key events
    switch (c) {
        case 's':
        case 'g':
            smooth = !smooth;
            break;
        case 'p':
            paused = !paused;
            break;
        default:
            ;
    }
    return true;
}

void onMouse(int event, int x, int y, int, void*)
{
    // the mouse event
    switch (event) {
        case CV_EVENT_LBUTTONUP:
            // we restart the game
            game = true;
            break;
    }
}

int getMarkerCode(Mat markerMatrix)
{
    int id = 0;
    int currentValue = 0;

    for(int j = 0; j < markerMatrix.rows; j++)
    {
        for(int i = 0; i < markerMatrix.cols; i++)
        {
            currentValue = int(markerMatrix.at<uchar>(j,i))? 0: 1;
            id += currentValue *pow(2, ( (markerMatrix.rows - j - 1)*4 + (markerMatrix.cols - i) - 1 ));
        }

    }

    return id;
}

pair<int, int> getMarkerSmallestId(Mat markerMatrix)
{
    int newCode;
    int minCode = getMarkerCode(markerMatrix);
    int rotation = 0;
    Mat matrix = markerMatrix;

    for(int i = 0; i < 3; i++)
    {
        matrix = rotate_cw(matrix, 90);
        newCode = getMarkerCode(matrix);
        if(newCode < minCode)
        {
            minCode = newCode;
            rotation = 90 * i;
        }
    }

    return pair<int, int>(minCode, rotation);
}

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    // game parameters
    const int margin = 50; //< pixel x-margin
    const int pingsz = 100; //< ping pong p
    const int rectsz = 20; //< ball radius
    const int speed = 10; //< ball velocity
    PVect recPos; //< the ball position
    PVect recSpeed; //< the ball speed
    int win = 0; //< the winner 1=left, 2=right
    int previousLastLeftPos = 0;
    int previousLastRightPos = 0;
    int lastLeftPos = 0;
    int lastRightPos = 0;
    const string window = "Marker Ping PONG !";

    Mat grayFrame;
    Mat thresholdedBinaryFrame;
    namedWindow(window, 1);
    //namedWindow("transform", 1);
    //namedWindow("cat", 1);

    vector<Point2f> markerNormalCoords(4);

    markerNormalCoords[0] = Point2f(-1/2.0, -1/2.0);
    markerNormalCoords[1] = Point2f(11/2.0, -1/2.0);
    markerNormalCoords[2] = Point2f(11/2.0, 11/2.0);
    markerNormalCoords[3] = Point2f(-1/2.0, 11/2.0);

    Mat image;


    Mat image_bat_one = imread("bat.jpg", CV_LOAD_IMAGE_COLOR);

    vector<Point2f> batOneCoords(4);

    batOneCoords[0] = Point2f(0, 0);
    batOneCoords[1] = Point2f(image_bat_one.cols, 0);
    batOneCoords[2] = Point2f(image_bat_one.cols, image_bat_one.rows);
    batOneCoords[3] = Point2f(0, image_bat_one.rows);

    Mat image_bat_two = imread("second_bat.jpg", CV_LOAD_IMAGE_COLOR);

    vector<Point2f> batTwoCoords(4);

    batTwoCoords[0] = Point2f(0, 0);
    batTwoCoords[1] = Point2f(image_bat_two.cols, 0);
    batTwoCoords[2] = Point2f(image_bat_two.cols, image_bat_two.rows);
    batTwoCoords[3] = Point2f(0, image_bat_two.rows);

    // we put the ball at the center
    recPos = PVect(camera_width / 2, camera_height / 2);
    //recSpeed = PVect(-1.0, 0.0);
    Scalar leftColor(0x00, 0x00, 0xFF);
    Scalar rightColor(0xFF, 0x00, 0x00);


    for(;;)
    {
        Mat frame;
        Mat visualizationCopy;

        cap >> frame; // get a new frame from camera

        Mat grayFrame;

        cvtColor(frame, grayFrame, CV_BGR2GRAY);


        visualizationCopy = frame.clone();
        vector<vector<Point2f>> result = detectMarkers(visualizationCopy, true);


        // ### 2 = draw the players
        drawMargin(frame, margin);
        Rect pad1(0, std::max(0, lastLeftPos - pingsz), margin, 2 * pingsz);
        rectangle(frame, pad1, leftColor, CV_FILLED);

        drawMargin(frame, camera_width - margin);
        // the display range
        Rect pad2(camera_width - margin, std::max(0, lastRightPos - pingsz),
                margin, 2 * pingsz);
        rectangle(frame, pad2, rightColor, CV_FILLED);

        // ### 3 = game logic
        if (!game) {
            // we put the ball at the center
            recPos = PVect(camera_width / 2, camera_height / 2);
            string title;
            Scalar color;
            switch (win) {
                case 0:
                    title = "Click to start";
                    color = Scalar(0x66, 0x66, 0x66);
                    break;
                case 1: // left won
                    title = "Left won !";
                    color = leftColor;
                    break;
                case 2: // right won
                    title = "Right won !";
                    color = rightColor;
                    break;
            }
            // we display the title
            Size textsize = getTextSize(title, CV_FONT_HERSHEY_COMPLEX, 2, 5, 0);
            Point org((camera_width - textsize.width) / 2, 3 * camera_height / 4);
            putText(frame, title, org, CV_FONT_HERSHEY_COMPLEX, 2, color, 5, CV_AA);
        }

        // we draw the ball anyway
        circle(frame, recPos, rectsz, Scalar(0, 10, 0), CV_FILLED, CV_AA);

        // the game in itself is simple
        if (game) {
            // the real thing happens here
            // 0=check won or not
            if (recPos.x < rectsz) {
                // left lost !
                win = 2;
                game = false;
            } else if (camera_width - recPos.x < rectsz) {
                // right lost !
                win = 1;
                game = false;
            } else {
                // we compute the ball displacement
                if (recSpeed.x == 0) {
                    // we reset the direction
                    recSpeed = PVect(1.0, 0.0);
                }

                // the new position
                PVect v = computeSpeed(recSpeed, speed);
                recPos = recPos + v;

                // computation of the new speed
                PVect nextPos = recPos + v;
                if (nextPos.x < margin + rectsz) {
                    // check for x-bounce on left
                    if (abs(lastLeftPos - nextPos.y) < pingsz + 0.75 * rectsz) {
                        // we bounce !
                        // we use the history to check the pad "speed"
                        // orientation and put that in the ball new speed
                        int disp = lastLeftPos - previousLastLeftPos;
                        double orient = disp / double(camera_height); // div by disp.t ?
                        orient *= 5.0;
                        // => unit vector of (0,orient) + recSpeed
                        recSpeed = processSpeed(recSpeed + PVect(-1.0, 0.0), orient);
                    }
                } else if (camera_width - nextPos.x < margin + rectsz) {
                    // check for x-bounce on right
                    if (abs(lastRightPos - nextPos.y) < pingsz + 0.75 * rectsz) {
                        // we bounce !
                        int disp = lastRightPos - previousLastRightPos;
                        double orient = disp / double(camera_height); // div by disp.t ?
                        orient *= 5.0;
                        recSpeed = processSpeed(recSpeed + PVect(1.0, 0.0), orient);
                    }
                }
                if (nextPos.y < 0 || nextPos.y > camera_height) {
                    // y-bounce
                    recSpeed = PVect(recSpeed.x, -recSpeed.y);
                }
            }
        }

        for(int markerCount = 0; markerCount < result.size(); markerCount++)
        {

            Mat transformMatrix, output;
            output = Mat::zeros( 6, 6, grayFrame.type() );
            vector<Point2f> oneMarker = result[markerCount];

            //Sort4PointsClockwise(oneMarker);

            transformMatrix = getPerspectiveTransform(oneMarker, markerNormalCoords);

            //cout << transformMatrix << endl;

            warpPerspective(grayFrame, output, transformMatrix,output.size() );

            threshold( output, output, threshold_value, 1, threshold_type );

            if(checkMatrixBorderForOnes(output))
            {
                output = output(Range(1, 5), Range(1,5));

                pair<int, int> container = getMarkerSmallestId(output);
                int id = container.first;
                int rotation = container.second;

                if(id == 0x005A)
                {
                    //imshow("transform", output*100);
                    Point2f averagePoint = (oneMarker[0] + oneMarker[1] + oneMarker[2] + oneMarker[3])*0.25;
                    circle(frame, averagePoint, rectsz, Scalar(0, 10, 30), CV_FILLED, CV_AA);
                    previousLastLeftPos = lastLeftPos;
                    lastLeftPos = averagePoint.y;

                    Rect boundBox = boundingRect( oneMarker );

                    Mat warpedImageToPutOutput = Mat::zeros( frame.rows, frame.cols, frame.type() );

                    Mat imageToPutTransformMatrix = getPerspectiveTransform(batOneCoords, oneMarker);

                    Mat new_img = rotate_cw(image_bat_one, rotation);

                    warpPerspective(new_img, warpedImageToPutOutput, imageToPutTransformMatrix, warpedImageToPutOutput.size() );

                    frame = frame + warpedImageToPutOutput;
                }

                if(id == 0x0690)
                {
                    //imshow("transform", output*100);
                    Point2f averagePoint = (oneMarker[0] + oneMarker[1] + oneMarker[2] + oneMarker[3])*0.25;
                    circle(frame, averagePoint, rectsz, Scalar(60, 10, 30), CV_FILLED, CV_AA);
                    previousLastRightPos = lastRightPos;
                    lastRightPos = averagePoint.y;

                    Rect boundBox = boundingRect( oneMarker );

                    Mat warpedImageToPutOutput = Mat::zeros( frame.rows, frame.cols, frame.type() );

                    Mat imageToPutTransformMatrix = getPerspectiveTransform(batTwoCoords, oneMarker);

                    Mat new_img = rotate_cw(image_bat_two, rotation);

                    warpPerspective(new_img, warpedImageToPutOutput, imageToPutTransformMatrix, warpedImageToPutOutput.size() );

                    frame = frame + warpedImageToPutOutput;
                }

                //cout << getMarkerSmallestId(output) << ";" << endl;
            }

        }

        /*
        if(result.size() > 0)
        {

            Mat markerTransformMatrix, markerAlignedOutput, imageToPutTransformMatrix, warpedImageToPutOutput;

            markerAlignedOutput = Mat::zeros( 6, 6, grayFrame.type() );

            for(int markerNumber = 0; markerNumber < result.size(); markerNumber++)
            {

                vector<Point2f> oneMarker = result[markerNumber];

                markerTransformMatrix = getPerspectiveTransform(oneMarker, markerNormalCoords);

                warpPerspective(grayFrame, markerAlignedOutput, markerTransformMatrix, markerAlignedOutput.size() );

                threshold( markerAlignedOutput, markerAlignedOutput, threshold_value, max_BINARY_value, threshold_type );

                 if(checkMatrixBorderForOnes(markerAlignedOutput))
                 {
                    imshow("transform", markerAlignedOutput);

                    Rect boundBox = boundingRect( oneMarker );

                    warpedImageToPutOutput = Mat::zeros( frame.rows, frame.cols, frame.type() );

                    imageToPutTransformMatrix = getPerspectiveTransform(catCoords, oneMarker);

                    warpPerspective(image, warpedImageToPutOutput, imageToPutTransformMatrix, warpedImageToPutOutput.size() );

                    frame = frame + warpedImageToPutOutput;
                 }
            }

        }
        */


        imshow(window, frame);

        setMouseCallback(window, onMouse, 0);

        // check keyboard events
        if (!pollKeys()) {
            break;
        }

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
