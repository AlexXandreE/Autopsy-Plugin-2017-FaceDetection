/*
 * Filename: c:\Users\alexf\OneDrive\Desktop\projetofinalcurso\FDRV\main.cpp
 * Path: c:\Users\alexf\OneDrive\Desktop\projetofinalcurso\FDRV
 * Created Date: Saturday, November 3rd 2018, 11:31:05 am
 * Author: Alexandre Frazao
 * 
 * Copyright (c) 2018  
 */

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudacodec.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <stdio.h>
#include <ctime>

using namespace std;
using namespace dlib;
using namespace boost;
using namespace cv;

using std::ofstream;


/******************* Data structures ************************/
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

struct thread_data
{
    dlib::matrix<dlib::rgb_pixel> *target;
    cv_image<bgr_pixel> *source;
    int id;
};
/************************************************************/

std::vector<filesystem::path> fileSearch(string path);
void threaded_copy(void *arg);


template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


int thread_count = std::thread::hardware_concurrency();
int num = std::thread::hardware_concurrency();
dlib::mutex count_mutex;
dlib::signaler count_signaler(count_mutex);





/**
 * Global logger object
 * */
logger dlog("FDRV_log");

int main(int argc, char **argv)
{
    dlog.set_level(LALL);
    net_type cnn_detector;
    // TODO:  Receber configs como o principal
    deserialize(argv[2]) >> cnn_detector;

    // TODO: Por com CUDA
    cv::VideoCapture video_reader;
    int apiID = cv::CAP_ANY; // 0 = autodetect default API

    // Can be replaced by 0 to show webcam stream
    video_reader.open(argv[1]);

    if (!video_reader.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    //TODO:: Colocar tempo do video no video final
    Mat frame;
    int total_frames = 0;

    std::vector<TimeFrame> identified_frames;

    clock_t begin = clock();

    bookmark s, e;

    cv::VideoWriter video_writer(argv[3],
                                 video_reader.get(CV_CAP_PROP_FOURCC),
                                 video_reader.get(CV_CAP_PROP_FPS),
                                 cv::Size(video_reader.get(CV_CAP_PROP_FRAME_WIDTH),
                                          video_reader.get(CV_CAP_PROP_FRAME_HEIGHT)),
                                 true);
    matrix<rgb_pixel> matrix;

    while (video_reader.read(frame))
    {
        if (frame.empty())
        {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        cv_image<bgr_pixel> image(frame);
        
        //assign_image(matrix, image);
        
        matrix.set_size(image.nr(), image.nc());
        //cout << "Starting convertion" << endl;
        std::vector<thread_data> t_data;
        //cout << "Creating threads" << endl;
        for (int t_id = 0; t_id < num; t_id++)
        {
            thread_data da;
            da.target = &matrix;
            da.source = &image;
            da.id = t_id;
            t_data.push_back(da);
            dlib::create_new_thread(threaded_copy, (void *)&t_data[t_id]);
        }

        dlib::auto_mutex abcd(count_mutex);
        while (thread_count > 0)
        {
            count_signaler.wait();
        }

        thread_count = std::thread::hardware_concurrency();
        
        std::vector<dlib::mmod_rect> faces_found = cnn_detector(matrix);
        // Write frames with faces found so that we can analise them
        if (faces_found.size() == 0)
        {
            if (s.frame.empty())
            {
                e.frame = frame;
                e.frame_position = total_frames - 1;
                double frame_sec = total_frames / video_reader.get(CV_CAP_PROP_FPS);

                identified_frames.push_back(TimeFrame(s, e, e.frame_position - s.frame_position, frame_sec));
            }
        }
        else
        {
            s.frame_position = total_frames - 1;
            s.frame = frame;
            double frame_sec = total_frames / video_reader.get(CV_CAP_PROP_FPS);
            
            cv::putText(frame, "Time: " + to_string(frame_sec), cvPoint(50, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 255, 250));

            if (true)
            {
                video_writer.write(frame);
            }
        }
        total_frames++;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor

    cout << "Total number of frames: " << total_frames << endl;
    double total_time = double(clock() - begin) / CLOCKS_PER_SEC;
    cout << "Total time (seconds): " << total_time << endl;
    cout << "FPS: " << total_frames / total_time << endl;

    video_reader.release();
    video_writer.release();
    return 0;
}
// TODO: FDXML com tempos, etc...

void create_DFXML()
{
}

void threaded_copy(void *arg)
{
    thread_data *data = (thread_data *)arg;

    int start = data->id;
    int jump = num;

    for (long r = start; r < (*data->source).nr(); r += jump)
    {
        for (long c = 0; c < (*data->source).nc(); c++)
        {
            assign_pixel((*data->target)(r, c), (*data->source)(r, c));
        }
    }
    //cout << "Done my job -- closing" << endl;
    dlib::auto_mutex locker(count_mutex);
    thread_count--;
    // Now we signal this change.  This will cause one thread that is currently waiting
    // on a call to count_signaler.wait() to unblock.
    count_signaler.signal();
}

void parse_config(std::string path)
{
    boost::property_tree::ptree json_tree;
    try
    {
        read_json(path, json_tree);
    }
    catch (std::exception &e)
    {
        cout << "Error parsing json" << endl;
        exit(2);
    }
    /*
    try
    {
        workspace = json_tree.get<string>("workspace");
        imagesToFindPath = json_tree.get<string>("imagesPath");
        positiveImgPath = json_tree.get<string>("wanted_faces");
        detectorPath = json_tree.get_child("paths.").get<string>("0");
        recognitionPath = json_tree.get_child("paths.").get<string>("1");
        shapePredictorPath = json_tree.get_child("paths.").get<string>("2");
        doRecognition = json_tree.get<bool>("doRecognition");
    }
    catch (std::exception &e)
    {
        cout << "Error Initiating variables" << endl;
        exit(3);
    }
    */
}

std::vector<filesystem::path> fileSearch(string path)
{
    std::vector<filesystem::path> images_path;
    std::vector<string> targetExtensions;

    targetExtensions.push_back(".mp4");
    targetExtensions.push_back(".avi");

    if (!filesystem::exists(path))
    {
        exit(4);
    }

    for (filesystem::recursive_directory_iterator end, dir(path); dir != end; ++dir)
    {
        string extension = filesystem::path(*dir).extension().generic_string();
        transform(extension.begin(), extension.end(), extension.begin(), ::toupper);
        if (std::find(targetExtensions.begin(), targetExtensions.end(), extension) != targetExtensions.end())
        {
            images_path.push_back(filesystem::path(*dir));
        }
    }
    return images_path;
}