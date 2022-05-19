#include "mainwindow.h"
#include <QApplication>
#include <QSplashScreen>
#include <QThread>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QPixmap pixmap(":/res/11.jpeg");
    QSplashScreen *splash = new QSplashScreen;
    splash->showMessage("程序启动中...", Qt::AlignLeft, Qt::white);	//显示文字、对齐方式、文字颜色
    splash->setPixmap(pixmap);										//绑定图片
    splash->show();

    QThread::sleep(2);

    MainWindow w;
    w.show();

    splash->finish(&w);											//关闭启动界面
    delete splash;

    return a.exec();
}
