#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "cnpy.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

#ifdef WIN32
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void saveFlow(const std::string name, 
                     const GpuMat& d_flow, 
                     const std::string path)
{
    Mat mat(d_flow);
    mat.convertTo(mat,CV_32F);
    std::vector<float> array((float*)mat.data, (float*)mat.data + mat.rows * mat.cols);
    // cout<<"Processing frame: "<<path<<endl; 
    cnpy::npy_save(path + ".npy", array, "w");
}

static void showFlow(const std::string name, const GpuMat& d_flow, 
        const float max_motion, const bool write, const std::string path)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, max_motion);
    // cout<<"Processing frame: "<<path<<endl; 
    if (write){
        imwrite(path + ".jpg", out);
    }
    else{
        imshow(name, out);
    }
}

int main(int argc, const char* argv[])
// This code expects two arguments, the absolute path to the root directory of
// the dataset (where the OF will be stored) and the relative subpath of the
// location where the videos are. This code works on the assumption that the
// last dir of this subpath is the prefix. 
// It will save the OFs in argv[1]/OF/OF_type/prefix/filename.jpg
{
    std::string src_path_video, src_fullpath_video, dest_path_video, dest_path_filename;

    if (argc != 3) {
        cerr << "Usage : " << argv[0] << "<absolute path to root directory of the dataset>" << argv[1] << "<subpath relative to root to videos>" << endl;
        return -1;
    } else {
        src_path_video = argv[1];
        src_fullpath_video = string(argv[1]) + OS_SEP + string(argv[2]);
    }
    
    // std::string folder_name = "JPEGImages";  // hardcoded for DAVIS
    std::string prefix;
    std::string filename;
    std::string of_type;
    
    std::vector<cv::String> filenames; 
    std::size_t filename_idx; 
    std::size_t filename_idx_ext; 
    std::size_t prefix_idx; 
    cv::glob(src_fullpath_video, filenames, false);

    for (std::vector<cv::String>::iterator it = filenames.begin(); it != filenames.end()-1; ++it) {
        
        cv::String current_file = (cv::String) *it;
        cv::String next_file = (cv::String) *(it+1);
        Mat frame0 = imread(current_file, IMREAD_GRAYSCALE);
        Mat frame1 = imread(next_file, IMREAD_GRAYSCALE);

        if (frame0.empty()) {
            cerr << "Can't open image ["  << current_file << "]" << endl;
            return -1;
        }
        if (frame1.empty()) {
            cerr << "Can't open image ["  << next_file << "]" << endl;
            cerr << "Aborting" << endl;
            return -1;
        }

        if (frame1.size() != frame0.size()) {
            cerr << "Images should be of equal sizes" << endl;
            return -1;
        }


        // Get the filename and the prefix
        dest_path_filename = std::string(next_file);
        filename_idx = dest_path_filename.rfind(OS_SEP) + 1;
        filename = dest_path_filename.substr(filename_idx);
        // Remove extension from filename
        filename_idx_ext = filename.rfind('.');
        filename = filename.substr(0, filename_idx_ext);

        
        // dest_path_filename.substr(0, filename_idx - 1);  // path minus filename
        prefix_idx = dest_path_filename.substr(0, filename_idx - 1).rfind(OS_SEP) + 1;  // index of second last OS_SEP
        prefix = dest_path_filename.substr(prefix_idx, filename_idx - prefix_idx - 1);

        GpuMat d_frame0(frame0);
        GpuMat d_frame1(frame1);

        GpuMat d_flow(frame0.size(), CV_32FC2);
        GpuMat d_flow_brox(frame0.size(), CV_32FC2);
        GpuMat d_flow_tvl1(frame0.size(), CV_32FC2);
        GpuMat d_flow_farn(frame0.size(), CV_32FC2);
        GpuMat d_flow_lk(frame0.size(), CV_32FC2);

        Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
        Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
        Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
        Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

        // GpuMat out_gpu;
        
        {
            GpuMat d_frame0f;
            GpuMat d_frame1f;
            d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

            const int64 start = getTickCount();

            brox->calc(d_frame0f, d_frame1f, d_flow_brox);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "Brox : " << timeSec << " sec" << endl;
            
            of_type = "Brox";
            dest_path_video = src_path_video + "OF" + OS_SEP + of_type + OS_SEP + prefix;
            // cout<<"Creating path: "<<dest_path_video<<endl; 
            system(("mkdir -p "+ dest_path_video).c_str());  // create OF dir if does not exist
            dest_path_video = dest_path_video + OS_SEP + filename;
            // cout << "Writing to " << dest_path_video << endl;
            // cout<<dest_path_video<<endl;
            saveFlow(of_type, d_flow_brox, dest_path_video);
            // waitKey(1);
        }

        {
             const int64 start = getTickCount();

             lk->calc(d_frame0, d_frame1, d_flow_lk);

             const double timeSec = (getTickCount() - start) / getTickFrequency();
             cout << "LK : " << timeSec << " sec" << endl;
            
             of_type = "LK";
             dest_path_video = src_path_video + "OF" + OS_SEP + of_type + OS_SEP + prefix;
             // cout<<"Creating path: "<<dest_path_video<<endl; 
             system(("mkdir -p "+ dest_path_video).c_str());  // create OF dir if does not exist
             dest_path_video = dest_path_video + OS_SEP + filename;
             saveFlow(of_type, d_flow_lk, dest_path_video);
             // cout << "Writing to " << dest_path_video << endl;
             // showFlow(of_type, d_flow_lk, 10, true, dest_path_video);
             // showFlow("LK", d_flow);
        }

        {
            const int64 start = getTickCount();

            farn->calc(d_frame0, d_frame1, d_flow_farn);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "Farn : " << timeSec << " sec" << endl;
            
            of_type = "Farn";
            dest_path_video = src_path_video + "OF" + OS_SEP + of_type + OS_SEP + prefix;
            // cout<<"Creating path: "<<dest_path_video<<endl; 
            system(("mkdir -p "+ dest_path_video).c_str());  // create OF dir if does not exist
            dest_path_video = dest_path_video + OS_SEP + filename;
            // saveFlow(of_type, d_flow_farn, dest_path_video);
            // cout << "Writing to " << dest_path_video << endl;
            // showFlow(of_type, d_flow_farn, 10, true, dest_path_video);
            // showFlow("Farn", d_flow);
        }

        {
            const int64 start = getTickCount();

            tvl1->calc(d_frame0, d_frame1, d_flow_tvl1);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "TVL1 : " << timeSec << " sec" << endl;
            
            of_type = "TVL1";
            dest_path_video = src_path_video + "OF" + OS_SEP + of_type + OS_SEP + prefix;
            // cout<<"Creating path: "<<dest_path_video<<endl; 
            system(("mkdir -p "+ dest_path_video).c_str());  // create OF dir if does not exist
            dest_path_video = dest_path_video + OS_SEP + filename;
            saveFlow(of_type, d_flow_tvl1, dest_path_video);
            // cout << "Writing to " << dest_path_video << endl;
            // showFlow(of_type, d_flow_tvl1, 10, true, dest_path_video);
            
            // Mat out_brox = (Mat) d_flow_brox;
            // Mat out_tvl1 = (Mat) d_flow_tvl1;
            // Mat out_farn = (Mat) d_flow_farn;
            // Mat out;
            // hconcat(out_brox, out_tvl1, out); 
            // out_gpu.upload(out);
            // showFlow("TVL1", out_gpu);
            // waitKey(10);
        }

    }
    return 0;
}
