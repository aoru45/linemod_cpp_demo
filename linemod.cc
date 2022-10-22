#include <opencv2/opencv.hpp>
#include <iostream>
static const unsigned char SIMILARITY_LUT[256] = {
      0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4,
      0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4,
      3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
      0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3,
      0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4,
      0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4,
      3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
      0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

void Quantize(cv::Mat& src, cv::Mat& dst, std::vector<std::pair<cv::Point, int>>& pts){
    cv::Mat smoothed;
    cv::GaussianBlur(src, smoothed, cv::Size(7, 7), 0, 0,
               cv::BORDER_REPLICATE);
    cv::Mat sobel_dx, sobel_dy, angle, angle_q;
    cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Mat magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
    cv::phase(sobel_dx, sobel_dy, angle, true);
    // 到范围0-16
    angle.convertTo(angle_q, CV_8U, 16 / 360.f);
    cv::Size s = angle_q.size();
    // 把180度以上的置0, 梯度太小的去掉
    for(int Y=0;Y<s.height;++Y){
        for(int X=0;X<s.width;++X){
            if(magnitude.at<float>(Y,X) < 100.f){
                angle_q.at<uchar>(Y,X)=0;
            }else{
                uchar label = angle_q.at<uchar>(Y,X)&7;
                angle_q.at<uchar>(Y,X) = 1 << label; // 到0-8
                // label = 0 对应 0000 0001
                // label = 1 对应 0000 0010
                // ...
                pts.push_back({cv::Point(X,Y), int(label)});
            }
        }
    }
    dst = angle_q;
}
void Spread(cv::Mat& src, cv::Mat& dst, int T){
    cv::Size s = src.size();
    dst = src.clone();
    for(int TX=0;TX<T;TX++){
        for(int TY=0;TY<T;TY++){
            for(int Y=0;Y<s.height - TY;Y++){
                for(int X=0;X<s.width-TX;X++){
                    dst.at<uchar>(Y,X) |= src.at<uchar>(Y + TY, X + TX);
                }
            }
        }
    }
}

void Detect(cv::Mat& scene_mat,
cv::Mat& temp_mat){
    cv::Mat temp_q, scene_q, scene_q_spread;
    std::vector<std::pair<cv::Point, int>> feature_pts, tmp;
    // 注册模板 -> 提取特征点
    Quantize(temp_mat, temp_q, feature_pts);
    // 滑窗检测
    Quantize(scene_mat, scene_q, tmp);
    Spread(scene_q, scene_q_spread, 1);

    std::vector<cv::Mat> response_maps(8);
    // 生成cvmat格式response_maps
    for(int i=0;i<8;++i){
        cv::Size s = scene_q.size();
       cv::Mat response_map(s, CV_8U); 
       for(int Y=0;Y<s.height;++Y){
        for(int X=0;X<s.width;++X){
            uchar val = scene_q_spread.at<uchar>(Y,X); 
            // 低4位16种组合，高4位16种组合
            int idx_low = i * (16+16) + (val & 15); // 0000 1111
            int idx_high = i * (16 + 16) + 16 + ((val & 240) >> 4); // 1111 0000
            // 取最大
            response_map.at<uchar>(Y,X) = std::max(SIMILARITY_LUT[idx_low], SIMILARITY_LUT[idx_high]);
        }
       }
       response_maps[i] = response_map;
    }
    // 检测
    int res_w = scene_mat.cols - temp_mat.cols;
    int res_h = scene_mat.rows - temp_mat.rows;

    cv::Mat res(cv::Size(res_w, res_h), CV_32F);
    for(int Y=0;Y<res_h;++Y){
        for(int X=0;X<res_w;++X){
            float val = 0;
            for(auto& feat:feature_pts){
               cv::Point p = feat.first; 
               int label = feat.second;
                cv::Mat& response_map = response_maps[label];
                val += float(response_map.at<uchar>(Y + p.y, X + p.x));
            }
            res.at<float>(Y,X) = val;
        }
    }
    cv::Mat m;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    res.convertTo(m, CV_16U, 65535.f / maxVal);
    cv::imwrite("C:/Users/xueaoru/Desktop/xx.png", m);
    int num_features = feature_pts.size();
    std::cout << "Loc: " << maxLoc.x << "," << maxLoc.y <<" Score:" << 100 * maxVal / (4.f * num_features) << std::endl;
    cv::rectangle(scene_mat, cv::Rect(maxLoc.x, maxLoc.y, temp_mat.cols, temp_mat.rows),cv::Scalar(128), 1);
    cv::imshow("res", scene_mat);
    cv::waitKey();
}

int main(){
    cv::Mat temp_mat = cv::imread("C:/Users/xueaoru/Desktop/test_job/temp_image7.png", 0);
    cv::Mat scene_mat = cv::imread("C:/Users/xueaoru/Desktop/test_job/gray7.png", 0);
    cv::resize(temp_mat, temp_mat, cv::Size(), 0.25, 0.25);
    cv::resize(scene_mat, scene_mat, cv::Size(), 0.25, 0.25);
    Detect(scene_mat, temp_mat);
    return 0;
}