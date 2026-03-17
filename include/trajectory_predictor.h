#pragma once

#include <pcl/point_types.h>

#include <string>
#include <vector>

struct LanePolyline {
    struct Point {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
        double kappa = 0.0;
    };

    std::string token;
    double length_m = 0.0;
    std::vector<Point> points;
};

struct RoadPolygon {
    std::string token;
    std::string layer;  // road_segment / road_block
    bool is_intersection = false;  // meaningful for road_segment
    std::vector<pcl::PointXYZ> points;
};

struct DetCategoryPoint {
    double x = 0.0;
    double y = 0.0;
    std::string category;
};

struct TrackKinematicState {
    int track_id = -1;
    int mode_idx = -1;      // 0=CV, 1=CTRV, 2=RM, 3=CA
    int track_manage = 0;
    bool is_static = false;
    bool valid = false;

    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double v = 0.0;
    double accel = 0.0;
    double yaw_rate = 0.0;
};

struct PredTrajectory {
    int track_id = -1;
    int mode_idx = -1;
    std::vector<std::string> selected_lane_tokens;
    std::vector<pcl::PointXYZ> points;
};

struct PredictionConfig {
    double horizon_s = 5.0;
    double step_s = 0.1;
    double min_ref_length_m = 50.0;
    double stitch_max_gap_m = 1.0;
};

std::vector<PredTrajectory> buildVehiclePredictions(
    const std::vector<LanePolyline>& lanes,
    const std::vector<RoadPolygon>& road_polygons,
    const std::vector<DetCategoryPoint>& detections,
    const std::vector<TrackKinematicState>& tracks,
    const PredictionConfig& cfg);
