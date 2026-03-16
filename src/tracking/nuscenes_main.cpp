#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <array>

#include "imm_ukf_jpda.h"

using namespace std;
using namespace pcl;
using namespace Eigen;

static const string GLOBAL_FRAME = "map";

static double g_playback_rate = 2.0;
static std::mutex g_rate_mutex;

void playbackRateCallback(const std_msgs::Float64::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(g_rate_mutex);
    g_playback_rate = std::max(msg->data, 0.1);
    ROS_INFO("Playback rate changed to %.2f Hz", g_playback_rate);
}

// ─── data structures ──────────────────────────────────────────

struct NuScenesBox {
    PointCloud<PointXYZ> bbox;
    string instance_token;
    string category;
};

struct CameraInfo {
    string channel;
    string filepath;      // relative to dataroot
    double fx, fy, cx, cy;
    std::array<double, 16> global_to_cam;  // row-major 4x4
};

struct LanePolyline {
    string token;
    vector<PointXYZ> points;
};

struct NuScenesFrame {
    double timestamp;
    double ego_x, ego_y, ego_yaw;
    vector<NuScenesBox> boxes;
    vector<LanePolyline> lanes;

    string lidar_filepath;                   // relative to dataroot
    std::array<double, 16> lidar_to_global;  // row-major 4x4
    bool has_lidar = false;

    vector<CameraInfo> cameras;
};

// ─── file reading ─────────────────────────────────────────────

vector<NuScenesFrame> readFrames(const string& filepath)
{
    vector<NuScenesFrame> frames;
    ifstream file(filepath);
    if (!file.is_open()) {
        ROS_ERROR("Cannot open data file: %s", filepath.c_str());
        return frames;
    }

    string line;
    NuScenesFrame currentFrame;
    bool inFrame = false;

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        string tag;
        iss >> tag;

        if (tag == "FRAME") {
            if (inFrame) frames.push_back(currentFrame);
            currentFrame = NuScenesFrame();
            iss >> currentFrame.timestamp
                >> currentFrame.ego_x >> currentFrame.ego_y
                >> currentFrame.ego_yaw;
            inFrame = true;
        }
        else if (tag == "LIDAR" && inFrame) {
            iss >> currentFrame.lidar_filepath;
            for (int i = 0; i < 16; i++)
                iss >> currentFrame.lidar_to_global[i];
            currentFrame.has_lidar = true;
        }
        else if (tag == "CAMERA" && inFrame) {
            CameraInfo cam;
            iss >> cam.channel >> cam.filepath
                >> cam.fx >> cam.fy >> cam.cx >> cam.cy;
            for (int i = 0; i < 16; i++)
                iss >> cam.global_to_cam[i];
            currentFrame.cameras.push_back(cam);
        }
        else if (tag == "BOX" && inFrame) {
            NuScenesBox box;
            double x1, y1, x2, y2, x3, y3, x4, y4, minZ, maxZ;
            iss >> x1 >> y1 >> x2 >> y2
                >> x3 >> y3 >> x4 >> y4
                >> minZ >> maxZ;
            iss >> box.instance_token >> box.category;

            PointXYZ o;
            o.x = x1; o.y = y1; o.z = minZ; box.bbox.push_back(o);
            o.x = x2; o.y = y2; o.z = minZ; box.bbox.push_back(o);
            o.x = x3; o.y = y3; o.z = minZ; box.bbox.push_back(o);
            o.x = x4; o.y = y4; o.z = minZ; box.bbox.push_back(o);
            o.x = x1; o.y = y1; o.z = maxZ; box.bbox.push_back(o);
            o.x = x2; o.y = y2; o.z = maxZ; box.bbox.push_back(o);
            o.x = x3; o.y = y3; o.z = maxZ; box.bbox.push_back(o);
            o.x = x4; o.y = y4; o.z = maxZ; box.bbox.push_back(o);

            currentFrame.boxes.push_back(box);
        }
        else if (tag == "LANE" && inFrame) {
            LanePolyline lane;
            int n_points = 0;
            iss >> lane.token >> n_points;
            if (n_points < 2) continue;

            lane.points.reserve(n_points);
            for (int i = 0; i < n_points; i++) {
                double x, y;
                if (!(iss >> x >> y)) break;
                PointXYZ p;
                p.x = static_cast<float>(x);
                p.y = static_cast<float>(y);
                p.z = 0.0f;
                lane.points.push_back(p);
            }
            if (lane.points.size() >= 2)
                currentFrame.lanes.push_back(lane);
        }
        else if (tag == "ENDFRAME" && inFrame) {
            frames.push_back(currentFrame);
            inFrame = false;
        }
    }
    if (inFrame) frames.push_back(currentFrame);

    file.close();
    return frames;
}

// ─── LiDAR point cloud loading ───────────────────────────────

sensor_msgs::PointCloud2 loadLidarCloud(
    const string& bin_path,
    const std::array<double, 16>& T,
    const ros::Time& stamp)
{
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = GLOBAL_FRAME;
    cloud_msg.header.stamp = stamp;

    ifstream fin(bin_path, ios::binary);
    if (!fin.is_open()) {
        ROS_WARN_ONCE("Cannot open LiDAR file: %s", bin_path.c_str());
        return cloud_msg;
    }

    fin.seekg(0, ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, ios::beg);

    size_t num_points = file_size / (5 * sizeof(float));
    vector<float> raw(num_points * 5);
    fin.read(reinterpret_cast<char*>(raw.data()), file_size);
    fin.close();

    // output fields: x y z intensity
    cloud_msg.height = 1;
    cloud_msg.width = num_points;
    cloud_msg.is_dense = false;
    cloud_msg.is_bigendian = false;
    cloud_msg.point_step = 4 * sizeof(float);
    cloud_msg.row_step = cloud_msg.point_step * num_points;

    sensor_msgs::PointField fx, fy, fz, fi;
    fx.name = "x"; fx.offset = 0;  fx.datatype = sensor_msgs::PointField::FLOAT32; fx.count = 1;
    fy.name = "y"; fy.offset = 4;  fy.datatype = sensor_msgs::PointField::FLOAT32; fy.count = 1;
    fz.name = "z"; fz.offset = 8;  fz.datatype = sensor_msgs::PointField::FLOAT32; fz.count = 1;
    fi.name = "intensity"; fi.offset = 12; fi.datatype = sensor_msgs::PointField::FLOAT32; fi.count = 1;
    cloud_msg.fields = {fx, fy, fz, fi};

    cloud_msg.data.resize(cloud_msg.row_step);
    float* out = reinterpret_cast<float*>(cloud_msg.data.data());

    for (size_t i = 0; i < num_points; i++) {
        float lx = raw[i * 5 + 0];
        float ly = raw[i * 5 + 1];
        float lz = raw[i * 5 + 2];
        float intensity = raw[i * 5 + 3];

        float gx = T[0]*lx + T[1]*ly + T[2]*lz  + T[3];
        float gy = T[4]*lx + T[5]*ly + T[6]*lz  + T[7];
        float gz = T[8]*lx + T[9]*ly + T[10]*lz + T[11];

        out[i * 4 + 0] = gx;
        out[i * 4 + 1] = gy;
        out[i * 4 + 2] = gz;
        out[i * 4 + 3] = intensity;
    }

    return cloud_msg;
}

// ─── camera image loading with projected boxes ───────────────

cv::Point2d projectGlobalToCamera(
    double gx, double gy, double gz,
    const std::array<double, 16>& T,
    double fx, double fy, double cx, double cy,
    bool& valid)
{
    double cx_ = T[0]*gx + T[1]*gy + T[2]*gz  + T[3];
    double cy_ = T[4]*gx + T[5]*gy + T[6]*gz  + T[7];
    double cz_ = T[8]*gx + T[9]*gy + T[10]*gz + T[11];

    valid = (cz_ > 0.5);
    if (!valid) return {0, 0};

    double u = fx * cx_ / cz_ + cx;
    double v = fy * cy_ / cz_ + cy;
    return {u, v};
}

sensor_msgs::ImagePtr loadCameraImage(
    const string& img_path,
    const CameraInfo& cam,
    const vector<NuScenesBox>& det_boxes,
    const PointCloud<PointXYZ>& targetPoints,
    const vector<int>& trackManage,
    const vector<vector<double>>& targetVandYaw,
    const vector<PointCloud<PointXYZ>>& visBBs,
    const vector<int>& trackIds,
    const ros::Time& stamp)
{
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        ROS_WARN_ONCE("Cannot open camera image: %s", img_path.c_str());
        return nullptr;
    }

    auto& T = cam.global_to_cam;

    auto projectPt = [&](double gx, double gy, double gz, bool& ok) -> cv::Point {
        cv::Point2d p = projectGlobalToCamera(gx, gy, gz, T,
                                              cam.fx, cam.fy, cam.cx, cam.cy, ok);
        return cv::Point(static_cast<int>(p.x), static_cast<int>(p.y));
    };

    auto inImage = [&](const cv::Point& p) -> bool {
        return p.x >= -200 && p.x < img.cols + 200 &&
               p.y >= -200 && p.y < img.rows + 200;
    };

    // Draw detection boxes (blue)
    // for (const auto& box : det_boxes) {
    //     if (box.bbox.size() < 8) continue;
    //     vector<cv::Point> pts(8);
    //     bool all_valid = true;
    //     int front_count = 0;
    //     for (int i = 0; i < 8; i++) {
    //         bool ok;
    //         pts[i] = projectPt(box.bbox[i].x, box.bbox[i].y, box.bbox[i].z, ok);
    //         if (ok) front_count++;
    //     }
    //     if (front_count < 4) continue;

    //     auto drawEdge = [&](int a, int b) {
    //         bool okA, okB;
    //         cv::Point pA = projectPt(box.bbox[a].x, box.bbox[a].y, box.bbox[a].z, okA);
    //         cv::Point pB = projectPt(box.bbox[b].x, box.bbox[b].y, box.bbox[b].z, okB);
    //         if (okA && okB && inImage(pA) && inImage(pB))
    //             cv::line(img, pA, pB, cv::Scalar(255, 150, 0), 2);
    //     };

    //     for (int i = 0; i < 4; i++) {
    //         drawEdge(i, (i+1)%4);
    //         drawEdge(i+4, (i+1)%4+4);
    //         drawEdge(i, i+4);
    //     }
    // }

    // // Draw tracked boxes (green) with ID labels
    // for (size_t oi = 0; oi < visBBs.size(); oi++) {
    //     if (visBBs[oi].size() < 8) continue;
    //     if (trackManage.size() <= oi || trackManage[oi] == 0) continue;

    //     int front_count = 0;
    //     vector<cv::Point> pts(8);
    //     for (int i = 0; i < 8; i++) {
    //         bool ok;
    //         pts[i] = projectPt(visBBs[oi][i].x, visBBs[oi][i].y, visBBs[oi][i].z, ok);
    //         if (ok) front_count++;
    //     }
    //     if (front_count < 4) continue;

    //     auto drawEdge = [&](int a, int b) {
    //         bool okA, okB;
    //         cv::Point pA = projectPt(visBBs[oi][a].x, visBBs[oi][a].y, visBBs[oi][a].z, okA);
    //         cv::Point pB = projectPt(visBBs[oi][b].x, visBBs[oi][b].y, visBBs[oi][b].z, okB);
    //         if (okA && okB && inImage(pA) && inImage(pB))
    //             cv::line(img, pA, pB, cv::Scalar(0, 255, 0), 2);
    //     };

    //     for (int i = 0; i < 4; i++) {
    //         drawEdge(i, (i+1)%4);
    //         drawEdge(i+4, (i+1)%4+4);
    //         drawEdge(i, i+4);
    //     }

    //     // Draw track ID at the top-center of the box
    //     bool okTop;
    //     double top_z = visBBs[oi][4].z;
    //     double cx = 0, cy = 0;
    //     for (int i = 0; i < 4; i++) {
    //         cx += visBBs[oi][i].x; cy += visBBs[oi][i].y;
    //     }
    //     cx /= 4; cy /= 4;
    //     cv::Point label_pt = projectPt(cx, cy, top_z, okTop);
    //     if (okTop && inImage(label_pt)) {
    //         char buf[64];
    //         int tid = (oi < trackIds.size()) ? trackIds[oi] : (int)oi;
    //         snprintf(buf, sizeof(buf), "T%d", tid);
    //         cv::putText(img, buf, cv::Point(label_pt.x - 10, label_pt.y - 8),
    //                     cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //                     cv::Scalar(0, 255, 0), 2);
    //     }
    // }

    // Draw tracked point positions (yellow dots)
    for (size_t i = 0; i < targetPoints.size(); i++) {
        if (trackManage[i] == 0) continue;
        if (std::isnan(targetPoints[i].x) || std::isnan(targetPoints[i].y)) continue;
        bool ok;
        cv::Point p = projectPt(targetPoints[i].x, targetPoints[i].y, 0.5, ok);
        if (ok && p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows)
            cv::circle(img, p, 5, cv::Scalar(0, 255, 255), -1);
    }

    // Channel label
    cv::putText(img, cam.channel, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    std_msgs::Header header;
    header.stamp = stamp;
    header.frame_id = cam.channel;
    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, img);
    return cv_img.toImageMsg();
}

// ─── visualization publishing ─────────────────────────────────

void publishVisualization(
    ros::Publisher& vis_pub,
    ros::Publisher& vis_pub2,
    ros::Publisher& lane_pub,
    ros::Publisher& box_pub,
    ros::Publisher& hud_pub,
    ros::Publisher& label_pub,
    const NuScenesFrame& frame,
    const PointCloud<PointXYZ>& targetPoints,
    const vector<vector<double>>& targetVandYaw,
    const vector<int>& trackManage,
    const vector<bool>& isStaticVec,
    const vector<bool>& isVisVec,
    const vector<PointCloud<PointXYZ>>& visBBs,
    const vector<int>& trackIds,
    const ros::Time& stamp,
    size_t frame_idx,
    size_t total_frames)
{
    // --- Lane centerlines (light gray) ---
    visualization_msgs::Marker lanes_marker;
    lanes_marker.header.frame_id = GLOBAL_FRAME;
    lanes_marker.header.stamp = stamp;
    lanes_marker.ns = "lanes";
    lanes_marker.id = 0;
    lanes_marker.action = visualization_msgs::Marker::ADD;
    lanes_marker.type = visualization_msgs::Marker::LINE_LIST;
    lanes_marker.pose.orientation.w = 1.0;
    lanes_marker.scale.x = 0.08;
    lanes_marker.color.r = 0.85f;
    lanes_marker.color.g = 0.85f;
    lanes_marker.color.b = 0.85f;
    lanes_marker.color.a = 0.75f;
    lanes_marker.lifetime = ros::Duration(1.0);
    for (const auto& lane : frame.lanes) {
        if (lane.points.size() < 2) continue;
        for (size_t i = 0; i + 1 < lane.points.size(); i++) {
            geometry_msgs::Point p0, p1;
            p0.x = lane.points[i].x;     p0.y = lane.points[i].y;     p0.z = 0.03;
            p1.x = lane.points[i + 1].x; p1.y = lane.points[i + 1].y; p1.z = 0.03;
            lanes_marker.points.push_back(p0);
            lanes_marker.points.push_back(p1);
        }
    }
    lane_pub.publish(lanes_marker);

    // --- Tracking points (colored by status) ---
    visualization_msgs::Marker pointsY, pointsG, pointsR, pointsB;
    pointsY.header.frame_id = pointsG.header.frame_id =
        pointsR.header.frame_id = pointsB.header.frame_id = GLOBAL_FRAME;
    pointsY.header.stamp = pointsG.header.stamp =
        pointsR.header.stamp = pointsB.header.stamp = stamp;
    pointsY.ns = pointsG.ns = pointsR.ns = pointsB.ns = "tracking_points";
    pointsY.action = pointsG.action = pointsR.action = pointsB.action =
        visualization_msgs::Marker::ADD;
    pointsY.pose.orientation.w = pointsG.pose.orientation.w =
        pointsR.pose.orientation.w = pointsB.pose.orientation.w = 1.0;
    pointsY.id = 1; pointsG.id = 2; pointsR.id = 3; pointsB.id = 4;
    pointsY.type = pointsG.type = pointsR.type = pointsB.type =
        visualization_msgs::Marker::POINTS;
    pointsY.scale.x = pointsG.scale.x = pointsR.scale.x = pointsB.scale.x = 0.5;
    pointsY.scale.y = pointsG.scale.y = pointsR.scale.y = pointsB.scale.y = 0.5;
    pointsY.lifetime = pointsG.lifetime = pointsR.lifetime = pointsB.lifetime =
        ros::Duration(1.0);

    pointsY.color.r = 1.0f; pointsY.color.g = 1.0f; pointsY.color.a = 1.0;
    pointsG.color.g = 1.0f; pointsG.color.a = 1.0;
    pointsR.color.r = 1.0f; pointsR.color.a = 1.0;
    pointsB.color.b = 1.0f; pointsB.color.a = 1.0;

    for (size_t i = 0; i < targetPoints.size(); i++) {
        if (trackManage[i] == 0) continue;
        geometry_msgs::Point p;
        p.x = targetPoints[i].x; p.y = targetPoints[i].y; p.z = 0;
        if (std::isnan(p.x) || std::isnan(p.y)) continue;
        if (isStaticVec[i])          pointsB.points.push_back(p);
        else if (trackManage[i] < 5) pointsY.points.push_back(p);
        else if (trackManage[i] == 5) pointsG.points.push_back(p);
        else                          pointsR.points.push_back(p);
    }
    vis_pub.publish(pointsY); vis_pub.publish(pointsG);
    vis_pub.publish(pointsR); vis_pub.publish(pointsB);

    // --- Track ID labels ---
    visualization_msgs::MarkerArray label_array;
    {
        visualization_msgs::Marker del;
        del.header.frame_id = GLOBAL_FRAME;
        del.header.stamp = stamp;
        del.ns = "track_ids";
        del.action = visualization_msgs::Marker::DELETEALL;
        label_array.markers.push_back(del);
    }
    int label_id = 0;
    const double ASSOC_DIST = 3.0;  // max distance (m) to match track to detection for category
    for (size_t i = 0; i < targetPoints.size(); i++) {
        if (trackManage[i] == 0) continue;
        if (std::isnan(targetPoints[i].x) || std::isnan(targetPoints[i].y)) continue;
        // Match track to nearest detection to get category (vehicle vs pedestrian)
        bool isVehicle = false;
        if (!frame.boxes.empty()) {
            double tx = targetPoints[i].x, ty = targetPoints[i].y;
            double bestD = ASSOC_DIST + 1;
            size_t bestIdx = 0;
            for (size_t b = 0; b < frame.boxes.size(); b++) {
                if (frame.boxes[b].bbox.size() < 4) continue;
                VectorXd cp = getCpFromBbox(frame.boxes[b].bbox);
                double d = sqrt((tx - cp(0))*(tx - cp(0)) + (ty - cp(1))*(ty - cp(1)));
                if (d < bestD) { bestD = d; bestIdx = b; }
            }
            if (bestD < ASSOC_DIST && frame.boxes[bestIdx].category.find("vehicle.") == 0)
                isVehicle = true;
        }

        visualization_msgs::Marker label;
        label.header.frame_id = GLOBAL_FRAME;
        label.header.stamp = stamp;
        label.ns = "track_ids";
        label.id = label_id++;
        label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::Marker::ADD;
        label.lifetime = ros::Duration(1.0);
        label.pose.position.x = targetPoints[i].x;
        label.pose.position.y = targetPoints[i].y;
        label.pose.position.z = 2.5;
        label.pose.orientation.w = 1.0;
        label.scale.z = 1.0;
        char buf[128];
        int tid = (i < trackIds.size()) ? trackIds[i] : (int)i;
        int modeIdx = -1;
        if (targetVandYaw[i].size() > 4)
            modeIdx = static_cast<int>(targetVandYaw[i][4]);
        const char* modeStr = (modeIdx == 0) ? "CV" : (modeIdx == 1) ? "CTRV" : (modeIdx == 2) ? "RM" : (modeIdx == 3) ? "CA" : "";
        // Only show motion model (CV/CTRV/RM/CA) for vehicles
        if (trackManage[i] >= 5 && !isStaticVec[i]) {
            if (isVehicle && modeStr[0]) {
                double accel = (targetVandYaw[i].size() > 3) ? targetVandYaw[i][3] : 0.0;
                if (modeIdx == 3)
                    snprintf(buf, sizeof(buf), "T%d v=%.1fm/s a=%.1f [CA]", tid, targetVandYaw[i][0], accel);
                else
                    snprintf(buf, sizeof(buf), "T%d v=%.1fm/s [%s]", tid, targetVandYaw[i][0], modeStr);
            } else {
                snprintf(buf, sizeof(buf), "T%d v=%.1fm/s", tid, targetVandYaw[i][0]);
            }
        } else {
            if (isVehicle && modeStr[0])
                snprintf(buf, sizeof(buf), "T%d [%s]", tid, modeStr);
            else
                snprintf(buf, sizeof(buf), "T%d", tid);
        }
        label.text = buf;
        if (isStaticVec[i])           { label.color.r=0.5; label.color.g=0.5; label.color.b=1.0; }
        else if (trackManage[i] >= 5) { label.color.r=0.2; label.color.g=1.0; label.color.b=0.2; }
        else                          { label.color.r=1.0; label.color.g=1.0; label.color.b=0.0; }
        label.color.a = 1.0;
        label_array.markers.push_back(label);
    }
    label_pub.publish(label_array);

    // --- Velocity arrows ---
    {
        visualization_msgs::Marker del_arrows;
        del_arrows.header.frame_id = GLOBAL_FRAME;
        del_arrows.header.stamp = stamp;
        del_arrows.ns = "arrows";
        del_arrows.action = visualization_msgs::Marker::DELETEALL;
        vis_pub2.publish(del_arrows);
    }
    for (size_t i = 0; i < targetPoints.size(); i++) {
        if (trackManage[i] == 0 || !isVisVec[i] || isStaticVec[i]) continue;
        geometry_msgs::Point p;
        p.x = targetPoints[i].x; p.y = targetPoints[i].y; p.z = 0;
        if (std::isnan(p.x) || std::isnan(p.y)) continue;
        visualization_msgs::Marker arrow;
        arrow.header.frame_id = GLOBAL_FRAME;
        arrow.header.stamp = stamp;
        arrow.ns = "arrows";
        arrow.action = visualization_msgs::Marker::ADD;
        arrow.type = visualization_msgs::Marker::ARROW;
        arrow.id = static_cast<int>(i);
        arrow.color.g = 1.0f; arrow.color.a = 1.0;
        arrow.pose.position = p;
        arrow.pose.orientation.z = sin(targetVandYaw[i][1] / 2.0);
        arrow.pose.orientation.w = cos(targetVandYaw[i][1] / 2.0);
        arrow.scale.x = std::max(targetVandYaw[i][0], 0.1);
        arrow.scale.y = 0.15; arrow.scale.z = 0.15;
        vis_pub2.publish(arrow);
    }

    // --- Tracked bounding box lines (green) ---
    visualization_msgs::Marker track_boxes;
    track_boxes.header.frame_id = GLOBAL_FRAME;
    track_boxes.header.stamp = stamp;
    track_boxes.ns = "track_boxes";
    track_boxes.id = 0;
    track_boxes.action = visualization_msgs::Marker::ADD;
    track_boxes.type = visualization_msgs::Marker::LINE_LIST;
    track_boxes.pose.orientation.w = 1.0;
    track_boxes.scale.x = 0.1;
    track_boxes.color.g = 1.0f; track_boxes.color.a = 1.0;
    track_boxes.lifetime = ros::Duration(1.0);

    for (size_t oi = 0; oi < visBBs.size(); oi++) {
        if (visBBs[oi].size() < 8) continue;
        for (int pi = 0; pi < 4; pi++) {
            if (std::isnan(visBBs[oi][pi].x) || std::isnan(visBBs[oi][(pi+1)%4].x) ||
                std::isnan(visBBs[oi][pi+4].x) || std::isnan(visBBs[oi][(pi+1)%4+4].x))
                continue;
            geometry_msgs::Point p;
            p.x=visBBs[oi][pi].x;       p.y=visBBs[oi][pi].y;       p.z=visBBs[oi][pi].z;       track_boxes.points.push_back(p);
            p.x=visBBs[oi][(pi+1)%4].x; p.y=visBBs[oi][(pi+1)%4].y; p.z=visBBs[oi][(pi+1)%4].z; track_boxes.points.push_back(p);
            p.x=visBBs[oi][pi].x;       p.y=visBBs[oi][pi].y;       p.z=visBBs[oi][pi].z;       track_boxes.points.push_back(p);
            p.x=visBBs[oi][pi+4].x;     p.y=visBBs[oi][pi+4].y;     p.z=visBBs[oi][pi+4].z;     track_boxes.points.push_back(p);
            p.x=visBBs[oi][pi+4].x;       p.y=visBBs[oi][pi+4].y;       p.z=visBBs[oi][pi+4].z;       track_boxes.points.push_back(p);
            p.x=visBBs[oi][(pi+1)%4+4].x; p.y=visBBs[oi][(pi+1)%4+4].y; p.z=visBBs[oi][(pi+1)%4+4].z; track_boxes.points.push_back(p);
        }
    }
    box_pub.publish(track_boxes);

    // --- Detection bounding boxes (blue) ---
    visualization_msgs::Marker det_boxes;
    det_boxes.header.frame_id = GLOBAL_FRAME;
    det_boxes.header.stamp = stamp;
    det_boxes.ns = "detections";
    det_boxes.id = 1;
    det_boxes.action = visualization_msgs::Marker::ADD;
    det_boxes.type = visualization_msgs::Marker::LINE_LIST;
    det_boxes.pose.orientation.w = 1.0;
    det_boxes.scale.x = 0.05;
    det_boxes.color.b = 1.0f; det_boxes.color.g = 0.5f; det_boxes.color.a = 0.5;
    det_boxes.lifetime = ros::Duration(1.0);
    for (const auto& box : frame.boxes) {
        for (int pi = 0; pi < 4; pi++) {
            geometry_msgs::Point p;
            p.x=box.bbox[pi].x;       p.y=box.bbox[pi].y;       p.z=box.bbox[pi].z;       det_boxes.points.push_back(p);
            p.x=box.bbox[(pi+1)%4].x; p.y=box.bbox[(pi+1)%4].y; p.z=box.bbox[(pi+1)%4].z; det_boxes.points.push_back(p);
            p.x=box.bbox[pi].x;       p.y=box.bbox[pi].y;       p.z=box.bbox[pi].z;       det_boxes.points.push_back(p);
            p.x=box.bbox[pi+4].x;     p.y=box.bbox[pi+4].y;     p.z=box.bbox[pi+4].z;     det_boxes.points.push_back(p);
            p.x=box.bbox[pi+4].x;       p.y=box.bbox[pi+4].y;       p.z=box.bbox[pi+4].z;       det_boxes.points.push_back(p);
            p.x=box.bbox[(pi+1)%4+4].x; p.y=box.bbox[(pi+1)%4+4].y; p.z=box.bbox[(pi+1)%4+4].z; det_boxes.points.push_back(p);
        }
    }
    box_pub.publish(det_boxes);

    // --- Ego vehicle marker ---
    visualization_msgs::Marker ego_marker;
    ego_marker.header.frame_id = GLOBAL_FRAME;
    ego_marker.header.stamp = stamp;
    ego_marker.ns = "ego_vehicle";
    ego_marker.id = 0;
    ego_marker.type = visualization_msgs::Marker::ARROW;
    ego_marker.action = visualization_msgs::Marker::ADD;
    ego_marker.lifetime = ros::Duration(1.0);
    ego_marker.pose.position.x = frame.ego_x;
    ego_marker.pose.position.y = frame.ego_y;
    ego_marker.pose.position.z = 0.5;
    ego_marker.pose.orientation.z = sin(frame.ego_yaw / 2.0);
    ego_marker.pose.orientation.w = cos(frame.ego_yaw / 2.0);
    ego_marker.scale.x = 4.0; ego_marker.scale.y = 1.5; ego_marker.scale.z = 1.0;
    ego_marker.color.r = 1.0f; ego_marker.color.a = 0.8;
    vis_pub.publish(ego_marker);

    // --- HUD text ---
    visualization_msgs::Marker hud;
    hud.header.frame_id = GLOBAL_FRAME;
    hud.header.stamp = stamp;
    hud.ns = "hud";
    hud.id = 0;
    hud.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    hud.action = visualization_msgs::Marker::ADD;
    hud.lifetime = ros::Duration(1.0);
    hud.pose.position.x = frame.ego_x - 70;
    hud.pose.position.y = frame.ego_y + 30;
    hud.pose.position.z = 5.0;
    hud.pose.orientation.w = 1.0;
    hud.scale.z = 2.0;
    hud.color.r = 1.0f; hud.color.g = 1.0f; hud.color.b = 1.0f; hud.color.a = 0.9;
    int active = 0;
    for (size_t i = 0; i < trackManage.size(); i++)
        if (trackManage[i] > 0) active++;
    double cur_rate;
    { std::lock_guard<std::mutex> lock(g_rate_mutex); cur_rate = g_playback_rate; }
    char hud_buf[256];
    snprintf(hud_buf, sizeof(hud_buf),
             "Frame %zu/%zu  Det:%zu  Tracks:%d  Rate:%.1fHz",
             frame_idx + 1, total_frames, frame.boxes.size(), active, cur_rate);
    hud.text = hud_buf;
    hud_pub.publish(hud);
}


// ─── main ─────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    ros::init(argc, argv, "nuscenes_tracking");
    ros::NodeHandle nh("~");

    string data_file, result_file, nuscenes_dataroot;
    double initial_rate;
    nh.param<string>("data_file", data_file, "");
    nh.param<string>("result_file", result_file, "tracking_results.txt");
    nh.param<double>("playback_rate", initial_rate, 2.0);
    nh.param<string>("nuscenes_dataroot", nuscenes_dataroot, "");

    { std::lock_guard<std::mutex> lock(g_rate_mutex); g_playback_rate = initial_rate; }

    if (data_file.empty()) {
        ROS_ERROR("Parameter 'data_file' not set!");
        return 1;
    }

    if (nuscenes_dataroot.empty()) {
        // Try to infer: data_file is typically in <project>/nuscenes_preprocessed/
        // and dataroot is <project>/v1.0-mini/
        size_t pos = data_file.rfind("/nuscenes_preprocessed");
        if (pos != string::npos) {
            nuscenes_dataroot = data_file.substr(0, pos) + "/v1.0-mini";
        }
    }

    ROS_INFO("=== nuScenes IMM-UKF-JPDA Tracking ===");
    ROS_INFO("Data file:     %s", data_file.c_str());
    ROS_INFO("Result file:   %s", result_file.c_str());
    ROS_INFO("Dataroot:      %s", nuscenes_dataroot.c_str());
    ROS_INFO("Playback rate: %.1f Hz (adjustable via /playback_rate topic)", initial_rate);

    vector<NuScenesFrame> frames = readFrames(data_file);
    ROS_INFO("Loaded %zu frames", frames.size());
    if (frames.empty()) { ROS_ERROR("No frames loaded"); return 1; }

    // Publishers
    ros::Publisher vis_pub    = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 100);
    ros::Publisher vis_pub2   = nh.advertise<visualization_msgs::Marker>("/visualization_marker2", 100);
    ros::Publisher lane_pub   = nh.advertise<visualization_msgs::Marker>("/visualization_marker_lane", 100);
    ros::Publisher box_pub    = nh.advertise<visualization_msgs::Marker>("/visualization_marker_boxes", 100);
    ros::Publisher hud_pub    = nh.advertise<visualization_msgs::Marker>("/visualization_marker_hud", 100);
    ros::Publisher label_pub  = nh.advertise<visualization_msgs::MarkerArray>("/track_labels", 100);
    ros::Publisher cloud_pub  = nh.advertise<sensor_msgs::PointCloud2>("/nuscenes/lidar", 1);

    tf::TransformBroadcaster tf_broadcaster;

    // Per-camera image publishers (created on demand)
    map<string, ros::Publisher> cam_pubs;
    for (const auto& frame : frames) {
        for (const auto& cam : frame.cameras) {
            if (cam_pubs.find(cam.channel) == cam_pubs.end()) {
                string topic = "/nuscenes/" + cam.channel;
                std::transform(topic.begin(), topic.end(), topic.begin(), ::tolower);
                cam_pubs[cam.channel] = nh.advertise<sensor_msgs::Image>(topic, 1);
            }
        }
    }

    ros::Subscriber rate_sub = nh.subscribe<std_msgs::Float64>(
        "/playback_rate", 1, playbackRateCallback);

    // Wait for RViz
    ROS_INFO("Waiting for RViz to connect (max 10s)...");
    ros::Time wait_start = ros::Time::now();
    while (vis_pub.getNumSubscribers() == 0 && ros::ok()) {
        ros::Duration(0.5).sleep();
        ros::spinOnce();
        if ((ros::Time::now() - wait_start).toSec() > 10.0) {
            ROS_WARN("No RViz subscriber detected, proceeding anyway.");
            break;
        }
    }
    if (vis_pub.getNumSubscribers() > 0) {
        ROS_INFO("RViz connected. Starting tracking...");
        ros::Duration(1.0).sleep();
    }

    ROS_INFO("  [Tip] Adjust speed:  rostopic pub /playback_rate std_msgs/Float64 \"data: 1.0\"");

    ofstream result_out(result_file);
    result_out << fixed << setprecision(6);
    result_out << "# nuScenes IMM-UKF-JPDA Tracking Results\n";
    result_out << "# FRAME frame_idx timestamp ego_x ego_y ego_yaw num_det num_track\n";
    result_out << "# TRACK track_id x y v yaw trackManage isStatic\n";
    result_out << "# DET instance_token category cx cy\n\n";

    vector<PointCloud<PointXYZI>> his_nearvisBBs;

    for (size_t fi = 0; fi < frames.size(); fi++) {
        if (!ros::ok()) break;

        NuScenesFrame& frame = frames[fi];

        vector<PointCloud<PointXYZ>> bBoxes;
        for (auto& box : frame.boxes) bBoxes.push_back(box.bbox);

        PointXYZ car_pos;
        car_pos.x = frame.ego_x; car_pos.y = frame.ego_y;

        PointCloud<PointXYZ> targetPoints;
        vector<vector<double>> targetVandYaw;
        vector<int> trackManage;
        vector<bool> isStaticVec, isVisVec;
        vector<PointCloud<PointXYZ>> visBBs;
        vector<int> trackIds;

        immUkfJpdaf(bBoxes, frame.timestamp, frame.ego_yaw,
                    targetPoints, targetVandYaw, trackManage,
                    isStaticVec, isVisVec, visBBs,
                    car_pos, his_nearvisBBs, &trackIds);

        int active_tracks = 0;
        for (size_t i = 0; i < trackManage.size(); i++)
            if (trackManage[i] > 0) active_tracks++;

        ROS_INFO("Frame %3zu/%zu | t=%.3f | det=%zu | tracks=%d",
                 fi + 1, frames.size(), frame.timestamp,
                 frame.boxes.size(), active_tracks);

        // Save results
        result_out << "FRAME " << fi << " " << frame.timestamp
                   << " " << frame.ego_x << " " << frame.ego_y
                   << " " << frame.ego_yaw
                   << " " << frame.boxes.size()
                   << " " << active_tracks << "\n";
        for (size_t i = 0; i < targetPoints.size(); i++) {
            if (trackManage[i] == 0) continue;
            if (std::isnan(targetPoints[i].x) || std::isnan(targetPoints[i].y)) continue;
            result_out << "TRACK " << i
                       << " " << targetPoints[i].x << " " << targetPoints[i].y
                       << " " << targetVandYaw[i][0] << " " << targetVandYaw[i][1]
                       << " " << trackManage[i] << " " << isStaticVec[i] << "\n";
        }
        for (const auto& box : frame.boxes) {
            VectorXd cp = getCpFromBbox(box.bbox);
            result_out << "DET " << box.instance_token << " " << box.category
                       << " " << cp(0) << " " << cp(1) << "\n";
        }
        result_out << "ENDFRAME\n\n";

        // Publish visualizations
        ros::Time stamp = ros::Time::now();

        // Broadcast ego vehicle TF so RViz can follow it
        tf::Transform ego_tf;
        ego_tf.setOrigin(tf::Vector3(frame.ego_x, frame.ego_y, 0.0));
        tf::Quaternion ego_q;
        ego_q.setRPY(0, 0, frame.ego_yaw);
        ego_tf.setRotation(ego_q);
        tf_broadcaster.sendTransform(
            tf::StampedTransform(ego_tf, stamp, GLOBAL_FRAME, "ego_vehicle"));

        publishVisualization(vis_pub, vis_pub2, lane_pub, box_pub, hud_pub, label_pub, frame,
                             targetPoints, targetVandYaw,
                             trackManage, isStaticVec, isVisVec, visBBs,
                             trackIds,
                             stamp, fi, frames.size());

        // Publish LiDAR point cloud
        if (frame.has_lidar && !nuscenes_dataroot.empty()) {
            string lidar_abs = nuscenes_dataroot + "/" + frame.lidar_filepath;
            sensor_msgs::PointCloud2 cloud =
                loadLidarCloud(lidar_abs, frame.lidar_to_global, stamp);
            if (cloud.width > 0)
                cloud_pub.publish(cloud);
        }

        // Publish camera images with projected boxes
        if (!nuscenes_dataroot.empty()) {
            for (const auto& cam : frame.cameras) {
                auto it = cam_pubs.find(cam.channel);
                if (it == cam_pubs.end()) continue;
                if (it->second.getNumSubscribers() == 0) continue;

                string img_abs = nuscenes_dataroot + "/" + cam.filepath;
                sensor_msgs::ImagePtr img_msg = loadCameraImage(
                    img_abs, cam, frame.boxes,
                    targetPoints, trackManage, targetVandYaw, visBBs, trackIds, stamp);
                if (img_msg)
                    it->second.publish(img_msg);
            }
        }

        ros::spinOnce();
        double cur_rate;
        { std::lock_guard<std::mutex> lock(g_rate_mutex); cur_rate = g_playback_rate; }
        ros::Rate(cur_rate).sleep();
    }

    result_out.close();
    ROS_INFO("=== Tracking complete ===");
    ROS_INFO("Results saved to: %s", result_file.c_str());
    return 0;
}
