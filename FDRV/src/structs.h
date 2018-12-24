//
// Created by Alexandre on 24-Dec-18.
//

#ifndef FDRV_STRUCTS_H
#define FDRV_STRUCTS_H

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct bookmark
{
    cv::Mat frame;
    int frame_position;
};

class TimeFrame
{
private:
    bookmark start;
    bookmark end;
    int frames_between;
    double elapsed_seconds;

public:
    TimeFrame(bookmark starter_marker, bookmark end_marker, int number_between_frames, double second)
    {
        start = starter_marker;
        end = end_marker;
        frames_between = number_between_frames;
        elapsed_seconds = second;
    }
};

#endif //FDRV_STRUCTS_H
