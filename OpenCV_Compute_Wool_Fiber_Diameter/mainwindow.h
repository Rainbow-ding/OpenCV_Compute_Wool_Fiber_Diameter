#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void open_imageAcMsg();
    void close_imageAcMsg();
    void image_processAcMsg();
    void compute_dimAcMsg();
    void show_computeAcMsg();

private:
    Ui::MainWindow *ui;
    cv::Mat ROI,image5;
    std::vector<std::vector<cv::Point>> contours;
    void open_image();
    void show_image(cv::Mat& image);

};

#endif // MAINWINDOW_H
