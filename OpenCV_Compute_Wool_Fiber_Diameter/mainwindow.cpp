#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QGraphicsScene"
#include "iostream"
#include <sstream>
#include <iomanip>
#include "math.h"
#include "QFileDialog"
#include "QDebug"

using namespace cv;
using namespace std;



QImage Mat2QImage(cv::Mat& mat);
void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat &dst, double min_area);
bool expandEdge(const Mat & img, int edge[], const int edgeID);
cv::Rect InSquare(Mat &img,const Point center);
vector<double> Conpute_dim(vector<Mat> Contours_choose, vector<vector<Point>> &all_longedge_contours);
void show_Conpute_dim(vector<Mat> Contours_choose);

vector<Mat> global_Contours_choose;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->open_imageAc,SIGNAL(triggered()),this,SLOT(open_imageAcMsg()));
    connect(ui->close_imageAc,SIGNAL(triggered()),this,SLOT(close_imageAcMsg()));
    connect(ui->image_processAc,SIGNAL(triggered()),this,SLOT(image_processAcMsg()));
    connect(ui->compute_dimAc,SIGNAL(triggered()),this,SLOT(compute_dimAcMsg()));
    connect(ui->show_computeAc,SIGNAL(triggered()),this,SLOT(show_computeAcMsg()));
    ui->image_processAc->setDisabled(true);
    ui->compute_dimAc->setDisabled(true);
    ui->show_computeAc->setDisabled(true);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::open_imageAcMsg()
{
    open_image();
}

void MainWindow::close_imageAcMsg()
{
    ui->graphicsView->close();
    ui->image_processAc->setDisabled(true);
}

void MainWindow::image_processAcMsg()
{
    /*预处理*/
    Mat image0, image1, image2, image3;
    Mat image4;

    /*自适应二值化*/
    cv::adaptiveThreshold(ROI, image0, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 35, 5);
    //cv::imshow("image0",image0);

    /*中值滤波*/
    cv::medianBlur(image0,image1,3);
    //cv::imshow("image1",image1);

    /*去除小面积像素区域*/
    Clear_MicroConnected_Areas(image1, image2, 1000);
    //cv::imshow("去除小面积像素区域", image2);

    /*先腐蚀再膨胀，使样本轮廓尽可能闭合*/
    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(6, 6));
    cv::erode(image2, image3, element);
    cv::dilate(image3,image4,element);
    //cv::imshow("封闭轮廓",image4);

    image5 = image4.clone();
    /*填充样本内部空洞*/
    cv::floodFill(image4,Point(0,0),Scalar(0));
    //cv::imshow("填充后",image4);

    cv::Mat Imdest;
    Point pnt;
    std::vector<std::vector<cv::Point>> contoursDest;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(image4, contoursDest, hierarchy, cv::RETR_EXTERNAL/* RETR_TREE*/, cv::CHAIN_APPROX_SIMPLE);//RETR_EXTERNAL,内轮廓不重复计算
//    /*绘制所有的轮廓点集。*/
//    Mat Contours=Mat::zeros(image4.size(),CV_8UC1);  //绘制
//    cout << contoursDest.size() << endl;
//    for(int i=0;i<contoursDest.size();i++)
//    {
//        //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
//        for(int j=0;j<contoursDest[i].size();j++)
//        {
//            //绘制出contours向量内所有的像素点
//            Point P=Point(contoursDest[i][j].x,contoursDest[i][j].y);
//            Contours.at<uchar>(P)=255;
//        }
//    }
//      imshow("Point of Contours",Contours);   //向量contours内保存的所有轮廓点集

    /*填充轮廓，先找到轮廓中的一个点，然后从该点利用flooddill函数填充整个封闭轮廓*/
    for(int i=0;i<contoursDest.size();i++)
    {
        int min_x = 1000, min_j;
        //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
        for(int j=0;j<contoursDest[i].size();j++)
        {
            //找出轮廓的最左侧角点
            if(contoursDest[i][j].x < min_x){
                min_x = contoursDest[i][j].x;
                min_j = j;
            }
        }
        pnt = Point(contoursDest[i][min_j].x + 8,contoursDest[i][min_j].y);
        //cout << pnt.x << " " << pnt.y << endl;
        cv::floodFill(image5,pnt,Scalar(0));
    }

    //反转图像像素值
    cv::bitwise_not(image5,image5);
    //cv::imshow("预处理后",image5);

    /*填充轮廓后进行边缘提取*/
    vector<Vec4i> hierarchy1;
    findContours(image5, contours, hierarchy1, RETR_TREE, CV_CHAIN_APPROX_NONE, Point());
//    cout << "【周长筛选前总共轮廓个数为】：" << (int)contours.size() << endl;
//    for (int i = 0; i < (int)contours.size(); i++){
//        double lenth  = arcLength(contours[i], true);
//        cout << "【用轮廓周长计算函数计算出来的第" << i << "个轮廓的周长为：】" << lenth << endl;
//    }

    /*筛选剔除掉周长小于100的轮廓*/
    vector <vector<Point>>::iterator iter = contours.begin();
    for (; iter != contours.end();){
        double lenth = arcLength(*iter,true);
        if (lenth < 100){
            iter = contours.erase(iter);
        }
        else{
            ++iter;
        }
    }

    std::cout << "【周长筛选后总共轮廓个数为：" << (int)contours.size() << endl;
    for (int i = 0; i < (int)contours.size(); i++){
        double lenth = arcLength(contours[i], true);
        std::cout << "【用轮廓周长计算函数计算出来的第" << i << "个轮廓的周长为：】" << lenth << endl;
    }

    cout << "【面积筛选前总共轮廓个数为】：" << (int)contours.size() << endl;
    for (int i = 0; i < (int)contours.size(); i++){
        double Area  = contourArea(contours[i]);
        cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：】" << Area << endl;
    }

    /*筛选剔除掉面积小于1500的轮廓*/
    vector <vector<Point>>::iterator iter1 = contours.begin();
    for (; iter1 != contours.end();){
        double Area = contourArea(*iter1);
        if (Area < 1600){
            iter1 = contours.erase(iter1);
        }
        else{
            ++iter1;
        }
    }
    cout << "【面积筛选后总共轮廓个数为：" << (int)contours.size() << endl;
    for (int i = 0; i < (int)contours.size(); i++){
        double Area = contourArea(contours[i]);
        cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：】" << Area << endl;
    }
    //显示图片到GrapthView
    Mat imageContours=ROI.clone();
    //将二值化图片转化为彩色图，并在初始图上画出轮廓。
    cv::cvtColor(imageContours,imageContours,COLOR_GRAY2RGB);
    drawContours(imageContours, contours, -1, Scalar(0, 0, 255), 3);   // -1 表示所有轮廓
    //cv::imshow("imageContours", imageContours);
    show_image(imageContours);

    ui->compute_dimAc->setEnabled(true);
}

void MainWindow::compute_dimAcMsg()
{

    /*绘制所有的轮廓点集。*/
    Mat Contours=Mat::zeros(ROI.size(),CV_8UC1);  //绘制
    for(int i=0;i<contours.size();i++)
    {
        //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
        for(int j=0;j<contours[i].size();j++)
        {
            //绘制出contours向量内所有的像素点
            Point P=Point(contours[i][j].x,contours[i][j].y);
            Contours.at<uchar>(P)=255;
        }
    }
    //imshow("Point of Contours",Contours);   //向量Contours内保存的所有轮廓点集

    /*求所有轮廓的最小外接矩形与最小外接矩形的中心点*/
    Point poc;
    vector<float> longedge;
    vector<Point> pocv;
    Mat show_dim = ROI.clone();
    cv::cvtColor(show_dim,show_dim,COLOR_GRAY2RGB);
    for (int c = 0; c < contours.size(); ++c)
    {
        cv::RotatedRect rotateRect = cv::minAreaRect(contours[c]);//轮廓最小外接矩形
        poc = Point(rotateRect.center.x, rotateRect.center.y);//最小外接矩形的中心点坐标
        pocv.push_back(poc);
        //画出最小外接矩形
        cv::Point2f rect_points[4];
        rotateRect.points(rect_points);
        for (int i = 0; i < 4; i++)
        {
            line(show_dim, rect_points[i], rect_points[(i + 1) % 4], Scalar(0, 255, 255), 2);
        }

        //计算长边
        if(rotateRect.size.width > rotateRect.size.height){
            longedge.push_back(rotateRect.size.width);
        }else{
            longedge.push_back(rotateRect.size.height);
        }
    }

    /*保留长边上的轮廓点*/
    vector<Mat> Contours_choose;
    Mat Contours_choose_in_one = Mat::zeros(image5.size(),CV_8UC1);
    vector<vector<Point>> longedge_contours;
    for(int i=0;i<contours.size();i++)
    {
        Mat M = Mat::zeros(image5.size(),CV_8UC1);
        vector<Point> longedgecontours;
        //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
        for(int j=0;j<contours[i].size();j++)
        {
            //绘制出Contours_choose向量内所有的像素点
            float dist2poc = sqrt(pow(contours[i][j].x - pocv[i].x,2)+pow(contours[i][j].y - pocv[i].y,2));
            if(dist2poc < (0.25 * longedge[i])){
                Point P=Point(contours[i][j].x,contours[i][j].y);
                longedgecontours.push_back(P);
                M.at<uchar>(P)=255;
                Contours_choose_in_one.at<uchar>(P)=255;
            }
        }
        Contours_choose.push_back(M);
        global_Contours_choose = Contours_choose;
        longedge_contours.push_back(longedgecontours);
        //string str = "Point of Contours_choose" + to_string(i);
        //imshow(str,Contours_choose[i]);              //向量Contours_choose内保存的所有轮廓点集
    }

    //imshow("Point of Contours_choose",Contours_choose_in_one);

    vector<double> average_mindist;
    vector<vector<Point>> all_longedge_contours;
    average_mindist = Conpute_dim(Contours_choose, all_longedge_contours);
    for (int c = 0; c < contours.size(); ++c)
    {
        std::stringstream ss;
        ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << average_mindist[c];
        cv::putText(show_dim,"Diameter=" + ss.str(),Point(pocv[c].x-140,pocv[c].y+30),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,120,0), 2, 2);
        cv::circle(show_dim,pocv[c],5,Scalar(0, 0, 0),-1);
    }

    Mat imageC=ROI.clone();
    //将二值化图片转化为彩色图，并在初始图上画出轮廓。
    cv::cvtColor(imageC,imageC,COLOR_GRAY2RGB);
    drawContours(imageC, all_longedge_contours, -1, Scalar(0, 0, 255), 3);   // -1 表示所有轮廓
    cv::imshow("显示长边轮廓",imageC);
    waitKey(0);
    cv::imshow("显示直径",show_dim);
    show_image(show_dim);
    waitKey(0);
    cv::destroyAllWindows();
    ui->compute_dimAc->setDisabled(true);
    ui->show_computeAc->setEnabled(true);
}

void MainWindow::show_computeAcMsg()
{
    vector<Mat> Contours_choose =global_Contours_choose;
    show_Conpute_dim(Contours_choose);
}

void MainWindow::open_image(){
    //读取原图片
    QString filename=QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::homePath(),tr("(*.jpg)\n(*.bmp)\n(*.png)"));   //打开图片文件，选择图片
    Mat image_read = imread(filename.toStdString(),0);
    if(!image_read.empty())
    {
        ui->statusBar->showMessage(tr("Open Image Success!"),3000); //打开成功时显示的内容
        ui->image_processAc->setEnabled(true);
    }
    else
    {
        ui->statusBar->showMessage(tr("Save Image Failed!"),3000);
        return;
    }

    Mat image_resize;
    /*缩小图像成620×480*/
    Size dsize = Size(660, 500);
    cv::resize(image_read,image_resize, dsize, 0, 0, INTER_AREA);

    /*裁剪图像边缘*/
    Rect m_select = Rect(10,10,650,490);
    ROI = image_resize(m_select);
    show_image(ROI);
}


void MainWindow::show_image(cv::Mat& image1){
    QImage image = Mat2QImage(image1);
    QGraphicsScene *scene = new QGraphicsScene;
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->resize(image.width() + 10, image.height() + 10);
    ui->graphicsView->show();

}

QImage Mat2QImage(cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        cv::cvtColor(mat,mat,cv::COLOR_BGR2RGB);
        const uchar *pSrc = (const uchar*)mat.data;
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        //image = image.rgbSwapped();
        return image.copy();
    }
    else if(mat.type() == CV_8UC4)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        qDebug("ERROR: Mat could not be converted to QImage.");
        return QImage();
    }
}


/**
* @brief  Clear_MicroConnected_Areas         清除微小面积连通区函数
* @param  src                                输入图像矩阵
* @param  dst                                输出结果
* @return min_area                           设定的最小面积清除阈值
*/
void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat &dst, double min_area)
{
    // 备份复制
    dst = src.clone();
    std::vector<std::vector<cv::Point> > contours;  // 创建轮廓容器
    std::vector<cv::Vec4i>   hierarchy;

    // 寻找轮廓的函数
    // 第四个参数CV_RETR_EXTERNAL，表示寻找最外围轮廓
    // 第五个参数CV_CHAIN_APPROX_NONE，表示保存物体边界上所有连续的轮廓点到contours向量内
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point());

    if (!contours.empty() && !hierarchy.empty())
    {
        std::vector<std::vector<cv::Point> >::const_iterator itc = contours.begin();
        // 遍历所有轮廓
        while (itc != contours.end())
        {
            // 定位当前轮廓所在位置
            cv::Rect rect = cv::boundingRect(cv::Mat(*itc));
            // contourArea函数计算连通区面积
            double area = contourArea(*itc);
            // 若面积小于设置的阈值
            if (area < min_area)
            {
                // 遍历轮廓所在位置所有像素点
                for (int i = rect.y; i < rect.y + rect.height; i++)
                {
                    uchar *output_data = dst.ptr<uchar>(i);
                    for (int j = rect.x; j < rect.x + rect.width; j++)
                    {
                        // 将连通区的值置0
                        if (output_data[j] == 0)
                        {
                            output_data[j] = 255;

                        }
                    }
                }
            }
            itc++;
        }
    }
}

/**
* @brief  Conpute_dim                        直径计算函数
* @param  src                                输入长边图像向量
* @return average_mindist                    输出结果
*/
vector<double> Conpute_dim(vector<Mat> Contours_choose, vector<vector<Point>> &all_longedge_contours)
{
    vector<double> average_mindist;
    /*计算两条长边之间的距离。*/
    for(int i = 0;i < Contours_choose.size();i++)
    {
        vector<vector<Point>> longedge_contours;
        vector<Vec4i> hierarchy2;
        findContours(Contours_choose[i], longedge_contours, hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
//        cout << "【筛选前总共轮廓个数为】：" << (int)longedge_contours.size() << endl;
//        for (int i = 0; i < (int)longedge_contours.size(); i++){
//            double lenth  = arcLength(longedge_contours[i], true);
//            cout << "【用轮廓周长计算函数计算出来的第" << i << "个轮廓的周长为：】" << lenth << endl;
//        }

        for (int i = 0; i < (int)longedge_contours.size(); i++){
            vector<Point> longedge_contours_i;
            for(int j = 0; j < (int)longedge_contours[i].size(); j++){
                longedge_contours_i.push_back(longedge_contours[i][j]);
            }
            all_longedge_contours.push_back(longedge_contours_i);
        }


        //计算距离
        vector<int> pt0x, pt0y, pt1x, pt1y, pt0px, pt0py;
        for(int j = 0;j < longedge_contours[0].size();j++)
        {
            int x = longedge_contours[0][j].x;
            int y = longedge_contours[0][j].y;
            pt0x.push_back(x);
            pt0y.push_back(y);
        }
        for(int j = 0;j < longedge_contours[1].size();j++)
        {
            int x = longedge_contours[1][j].x;
            int y = longedge_contours[1][j].y;
            pt1x.push_back(x);
            pt1y.push_back(y);
        }

        double dist, all_mindist = 0;
        for(int i = 0;i < pt0x.size();i++)
        {
            double mindist = 10000;
            int x, y;
            for(int j = 0;j < pt1x.size();j++){
                dist = sqrt(pow(pt0x[i]-pt1x[j],2)+pow(pt0y[i]-pt1y[j],2));
                if(dist < mindist){
                    mindist = dist;
                    x = pt1x[j];
                    y = pt1y[j];
                }
            }
//            cout << "mindist为：" << mindist << endl;
            all_mindist = all_mindist + mindist;
            Point pt0,pt1;
            pt0.x = pt0x[i],pt0.y=pt0y[i];
            pt1.x = x,pt1.y = y;
        }
        double average_dist = all_mindist/pt0x.size();
        cout << all_mindist/pt0x.size() << endl;
        average_mindist.push_back(average_dist);
    }

    return average_mindist;

}

/**
* @brief  Conpute_dim                        直径计算函数
* @param  src                                输入长边图像向量
* @return average_mindist                    输出结果
*/
void show_Conpute_dim(vector<Mat> Contours_choose)
{
    vector<double> average_mindist;
    /*计算两条长边之间的距离。*/
    for(int i=0;i<Contours_choose.size();i++)
    {
        vector<vector<Point>> longedge_contours;
        vector<vector<Point>> all_longedge_contours;
        vector<Vec4i> hierarchy;
        Mat tempp = Contours_choose[i].clone();
        cv::bitwise_not(tempp,tempp);

        findContours(Contours_choose[i], longedge_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
        cout << "【筛选前总共轮廓个数为】：" << (int)longedge_contours.size() << endl;
        for (int i = 0; i < (int)longedge_contours.size(); i++){
            double lenth  = arcLength(longedge_contours[i], true);
            cout << "【用轮廓周长计算函数计算出来的第" << i << "个轮廓的周长为：】" << lenth << endl;
        }

        //计算距离
        vector<int> pt0x, pt0y, pt1x, pt1y, pt0px, pt0py;
        for(int j=0;j<longedge_contours[0].size();j++)
        {
            int x = longedge_contours[0][j].x;
            int y = longedge_contours[0][j].y;
            pt0x.push_back(x);
            pt0y.push_back(y);
        }
        for(int j=0;j<longedge_contours[1].size();j++)
        {
            int x = longedge_contours[1][j].x;
            int y = longedge_contours[1][j].y;
            pt1x.push_back(x);
            pt1y.push_back(y);
        }

        double dist, all_mindist = 0;

        for(int c=0;c<pt0x.size();c++)
        {
            double mindist = 10000;
            int x, y;
            for(int j=0;j<pt1x.size();j++){
                dist = sqrt(pow(pt0x[c]-pt1x[j],2)+pow(pt0y[c]-pt1y[j],2));
                if(dist < mindist){
                    mindist = dist;
                    x = pt1x[j];
                    y = pt1y[j];
                }
            }
            all_mindist = all_mindist + mindist;
            Point pt0,pt1;
            pt0.x = pt0x[c],pt0.y=pt0y[c];
            pt1.x = x,pt1.y = y;

            std::stringstream ss;
            ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << mindist;
            Mat temp = tempp.clone();
            cv::cvtColor(temp,temp,COLOR_GRAY2BGR);
            cv::line(temp,pt0,pt1,Scalar(0,0,255),2,cv::LINE_AA);
            cv::circle(temp,pt0,3,Scalar(255,0,0),-1,cv::LINE_AA);
            cv::circle(temp,pt1,3,Scalar(0,255,0),-1,cv::LINE_AA);
            cv::putText(temp,ss.str(),Point(pt1.x+5,pt1.y+15), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
            cv::imshow("img",temp);
            waitKey(25);

        }
        waitKey(0);
        cv::destroyAllWindows();
    }
}







