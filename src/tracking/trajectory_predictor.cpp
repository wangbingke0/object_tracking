#include "trajectory_predictor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <ros/ros.h>

namespace {

struct LaneProjectionResult {
    bool valid = false;
    double dist = std::numeric_limits<double>::infinity();
    double nearest_x = 0.0;
    double nearest_y = 0.0;
    int seg_idx = 0;
    double seg_t = 0.0;
    double lateral_sign = 0.0;  // >0: left side of heading, <0: right
    double tangent_yaw = 0.0;
};

struct LaneChoiceResult {
    const LanePolyline* lane = nullptr;
    double nearest_valid_dist = std::numeric_limits<double>::infinity();
};

double normalizeAngle(double a)
{
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

double clampValue(double x, double lo, double hi)
{
    return std::max(lo, std::min(x, hi));
}

bool isVehicleCategory(const std::string& category)
{
    return category.find("vehicle.") == 0;
}

LaneProjectionResult projectPointToLane(
    const LanePolyline& lane, double x, double y, double heading_yaw)
{
    LaneProjectionResult best;
    if (lane.points.size() < 2) return best;

    for (size_t i = 0; i + 1 < lane.points.size(); i++) {
        const auto& p0 = lane.points[i];
        const auto& p1 = lane.points[i + 1];
        double dx = p1.x - p0.x;
        double dy = p1.y - p0.y;
        double len2 = dx * dx + dy * dy;
        if (len2 < 1e-6) continue;

        double t = ((x - p0.x) * dx + (y - p0.y) * dy) / len2;
        t = clampValue(t, 0.0, 1.0);

        double nx = p0.x + t * dx;
        double ny = p0.y + t * dy;
        double dist = hypot(nx - x, ny - y);
        if (dist < best.dist) {
            best.valid = true;
            best.dist = dist;
            best.nearest_x = nx;
            best.nearest_y = ny;
            best.seg_idx = static_cast<int>(i);
            best.seg_t = t;
            const double yaw0 = p0.yaw;
            const double yaw1 = p1.yaw;
            best.tangent_yaw = normalizeAngle(yaw0 + t * normalizeAngle(yaw1 - yaw0));
            if (!std::isfinite(best.tangent_yaw)) {
                best.tangent_yaw = atan2(dy, dx);
            }
        }
    }

    if (best.valid) {
        double hx = cos(heading_yaw);
        double hy = sin(heading_yaw);
        double vx = best.nearest_x - x;
        double vy = best.nearest_y - y;
        best.lateral_sign = hx * vy - hy * vx;
    }
    return best;
}

bool findLookaheadPoint(const LanePolyline& lane,
                        const LaneProjectionResult& proj,
                        double lookahead,
                        double& lx,
                        double& ly)
{
    if (!proj.valid || lane.points.size() < 2) return false;

    int i = proj.seg_idx;
    if (i < 0 || i + 1 >= static_cast<int>(lane.points.size())) return false;

    double sx = proj.nearest_x;
    double sy = proj.nearest_y;
    double remain = std::max(lookahead, 0.0);

    while (i + 1 < static_cast<int>(lane.points.size())) {
        double ex = lane.points[i + 1].x;
        double ey = lane.points[i + 1].y;
        double seg_dx = ex - sx;
        double seg_dy = ey - sy;
        double seg_len = hypot(seg_dx, seg_dy);

        if (seg_len < 1e-6) {
            i++;
            if (i + 1 >= static_cast<int>(lane.points.size())) break;
            sx = lane.points[i].x;
            sy = lane.points[i].y;
            continue;
        }

        if (remain <= seg_len) {
            double r = remain / seg_len;
            lx = sx + r * seg_dx;
            ly = sy + r * seg_dy;
            return true;
        }

        remain -= seg_len;
        i++;
        if (i + 1 >= static_cast<int>(lane.points.size())) break;
        sx = lane.points[i].x;
        sy = lane.points[i].y;
    }

    lx = lane.points.back().x;
    ly = lane.points.back().y;
    return true;
}

double computeRemainingDistanceToLaneEnd(
    const LanePolyline& lane,
    const LaneProjectionResult& proj)
{
    if (!proj.valid || lane.points.size() < 2) {
        return std::numeric_limits<double>::infinity();
    }
    int i = proj.seg_idx;
    if (i < 0 || i + 1 >= static_cast<int>(lane.points.size())) {
        return std::numeric_limits<double>::infinity();
    }

    double remain = 0.0;
    double sx = proj.nearest_x;
    double sy = proj.nearest_y;
    while (i + 1 < static_cast<int>(lane.points.size())) {
        double ex = lane.points[i + 1].x;
        double ey = lane.points[i + 1].y;
        remain += std::hypot(ex - sx, ey - sy);
        i++;
        if (i + 1 >= static_cast<int>(lane.points.size())) break;
        sx = lane.points[i].x;
        sy = lane.points[i].y;
    }
    return remain;
}

double computeLocalAvgAbsKappa(
    const LanePolyline& lane,
    const LaneProjectionResult& proj,
    double window_m)
{
    if (!proj.valid || lane.points.size() < 2 || window_m <= 0.0) return 0.0;
    if (proj.seg_idx < 0 || proj.seg_idx + 1 >= static_cast<int>(lane.points.size())) return 0.0;

    const auto& p0 = lane.points[proj.seg_idx];
    const auto& p1 = lane.points[proj.seg_idx + 1];
    const double kappa_proj = p0.kappa + proj.seg_t * (p1.kappa - p0.kappa);
    auto avgAbsKappaBackward = [&]() {
        double sum_abs_kappa = std::fabs(kappa_proj);
        int cnt = 1;
        double rem = window_m;
        double cx = proj.nearest_x;
        double cy = proj.nearest_y;
        for (int i = proj.seg_idx; i >= 0 && rem > 0.0; --i) {
            const double nx = lane.points[i].x;
            const double ny = lane.points[i].y;
            const double ds = std::hypot(nx - cx, ny - cy);
            if (ds < 1e-6) continue;
            rem -= ds;
            sum_abs_kappa += std::fabs(lane.points[i].kappa);
            cnt++;
            cx = nx;
            cy = ny;
        }
        return (cnt > 0) ? (sum_abs_kappa / static_cast<double>(cnt)) : 0.0;
    };

    auto avgAbsKappaForward = [&]() {
        double sum_abs_kappa = std::fabs(kappa_proj);
        int cnt = 1;
        double rem = window_m;
        double cx = proj.nearest_x;
        double cy = proj.nearest_y;
        for (int i = proj.seg_idx + 1; i < static_cast<int>(lane.points.size()) && rem > 0.0; ++i) {
            const double nx = lane.points[i].x;
            const double ny = lane.points[i].y;
            const double ds = std::hypot(nx - cx, ny - cy);
            if (ds < 1e-6) continue;
            rem -= ds;
            sum_abs_kappa += std::fabs(lane.points[i].kappa);
            cnt++;
            cx = nx;
            cy = ny;
        }
        return (cnt > 0) ? (sum_abs_kappa / static_cast<double>(cnt)) : 0.0;
    };

    const double avg_back = avgAbsKappaBackward();
    const double avg_front = avgAbsKappaForward();
    return std::max(avg_back, avg_front);
}

LaneChoiceResult chooseReferenceLane(
    const std::vector<LanePolyline>& lanes,
    double x,
    double y,
    double yaw,
    int mode_idx,
    double yaw_rate)
{
    LaneChoiceResult res;
    if (lanes.empty()) return res;

    const LanePolyline* nearest_lane = nullptr;
    double nearest_score = std::numeric_limits<double>::infinity();
    const double MAX_DIST = 8.0;
    const double MAX_HEADING_DIFF = M_PI * 0.25;  // 45 deg

    const LanePolyline* left_lane = nullptr;
    double left_dist = std::numeric_limits<double>::infinity();
    const LanePolyline* right_lane = nullptr;
    double right_dist = std::numeric_limits<double>::infinity();
    bool want_left = yaw_rate >= 0.0;

    for (const auto& lane : lanes) {
        LaneProjectionResult proj = projectPointToLane(lane, x, y, yaw);
        if (!proj.valid) continue;
        if (proj.dist > MAX_DIST) continue;

        double heading_diff = std::fabs(normalizeAngle(yaw - proj.tangent_yaw));
        if (heading_diff > MAX_HEADING_DIFF) continue;
        double score = proj.dist + 2.0 * heading_diff;

        if (score < nearest_score) {
            nearest_score = score;
            nearest_lane = &lane;
            res.nearest_valid_dist = proj.dist;
        }

        if (mode_idx == 1) {  // CTRV: side preference first.
            if (proj.lateral_sign > 0.1 && proj.dist < left_dist) {
                left_dist = proj.dist;
                left_lane = &lane;
            } else if (proj.lateral_sign < -0.1 && proj.dist < right_dist) {
                right_dist = proj.dist;
                right_lane = &lane;
            }
        }
    }

    if (mode_idx == 1) {
        // Preferred side, then opposite side (avoid reverse-direction lanes).
        if (want_left && left_lane != nullptr) { res.lane = left_lane; return res; }
        if (!want_left && right_lane != nullptr) { res.lane = right_lane; return res; }
        if (want_left && right_lane != nullptr) { res.lane = right_lane; return res; }
        if (!want_left && left_lane != nullptr) { res.lane = left_lane; return res; }
    }
    if (nearest_lane != nullptr) {
        res.lane = nearest_lane;
        return res;
    }

    // No same-direction candidate -> no reference line for this step.
    return res;
}

bool inferVehicleTrackByNearestDetection(
    const std::vector<DetCategoryPoint>& detections,
    double tx,
    double ty,
    double assoc_dist = 3.5)
{
    if (detections.empty()) return false;

    double best_d = assoc_dist + 1.0;
    int best_idx = -1;
    for (size_t i = 0; i < detections.size(); i++) {
        double d = hypot(tx - detections[i].x, ty - detections[i].y);
        if (d < best_d) {
            best_d = d;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx < 0 || best_d >= assoc_dist) return false;
    return isVehicleCategory(detections[best_idx].category);
}

bool pointOnSegment2D(double px, double py,
                      double ax, double ay,
                      double bx, double by,
                      double eps = 1e-4)
{
    double abx = bx - ax;
    double aby = by - ay;
    double apx = px - ax;
    double apy = py - ay;
    double cross = std::fabs(abx * apy - aby * apx);
    if (cross > eps) return false;

    double dot = apx * abx + apy * aby;
    if (dot < -eps) return false;
    double len2 = abx * abx + aby * aby;
    if (dot > len2 + eps) return false;
    return true;
}

bool pointInPolygon2D(const std::vector<pcl::PointXYZ>& poly, double x, double y)
{
    if (poly.size() < 3) return false;

    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly[i].x;
        double yi = poly[i].y;
        double xj = poly[j].x;
        double yj = poly[j].y;

        if (pointOnSegment2D(x, y, xj, yj, xi, yi)) return true;

        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

bool isOnRoad(const std::vector<RoadPolygon>& road_polygons, double x, double y)
{
    for (const auto& poly : road_polygons) {
        if (poly.points.size() < 3) continue;
        if (pointInPolygon2D(poly.points, x, y)) return true;
    }
    return false;
}

double computePolylineLength(const LanePolyline& lane)
{
    if (lane.length_m > 1e-6) return lane.length_m;
    if (lane.points.size() < 2) return 0.0;
    double len = 0.0;
    for (size_t i = 1; i < lane.points.size(); i++) {
        double dx = lane.points[i].x - lane.points[i - 1].x;
        double dy = lane.points[i].y - lane.points[i - 1].y;
        len += std::hypot(dx, dy);
    }
    return len;
}

LanePolyline stitchLaneByNearestEndpoint(
    const LanePolyline& base_lane,
    const std::vector<LanePolyline>& lanes,
    double min_ref_length_m,
    double max_join_dist_m,
    std::vector<std::string>* stitched_tokens = nullptr,
    std::vector<double>* stitch_join_dists_m = nullptr,
    std::vector<int>* stitch_joint_indices = nullptr)
{
    LanePolyline merged = base_lane;
    merged.length_m = computePolylineLength(merged);
    if (stitched_tokens != nullptr) {
        stitched_tokens->clear();
        stitched_tokens->push_back(base_lane.token);
    }
    if (stitch_join_dists_m != nullptr) {
        stitch_join_dists_m->clear();
    }
    if (stitch_joint_indices != nullptr) {
        stitch_joint_indices->clear();
    }
    if (merged.length_m >= min_ref_length_m || merged.points.size() < 2) {
        return merged;
    }

    std::unordered_set<std::string> used_tokens;
    used_tokens.insert(base_lane.token);

    while (merged.length_m < min_ref_length_m) {
        const LanePolyline* best_next = nullptr;
        double best_dist = std::numeric_limits<double>::infinity();
        if (merged.points.empty()) break;
        const auto& end_pt = merged.points.back();

        for (const auto& cand : lanes) {
            if (cand.points.size() < 2) continue;
            if (used_tokens.find(cand.token) != used_tokens.end()) continue;
            double d = std::hypot(cand.points.front().x - end_pt.x,
                                  cand.points.front().y - end_pt.y);
            if (d < best_dist) {
                best_dist = d;
                best_next = &cand;
            }
        }

        if (best_next == nullptr || best_dist > max_join_dist_m) break;
        used_tokens.insert(best_next->token);
        if (stitched_tokens != nullptr) stitched_tokens->push_back(best_next->token);
        if (stitch_join_dists_m != nullptr) stitch_join_dists_m->push_back(best_dist);
        if (stitch_joint_indices != nullptr) {
            stitch_joint_indices->push_back(static_cast<int>(merged.points.size()) - 1);
        }

        //在拼接相邻车道时，如果首尾重叠，就跳过重复的第一个点。
        size_t start_idx = 0;
        if (std::hypot(best_next->points.front().x - end_pt.x,
                       best_next->points.front().y - end_pt.y) < 1e-3) {
            start_idx = 1;
        }
        for (size_t i = start_idx; i < best_next->points.size(); i++) {
            merged.points.push_back(best_next->points[i]);
        }
        merged.length_m = computePolylineLength(merged);
    }

    return merged;
}

void smoothLaneYawKappaAtJoints(
    LanePolyline& lane,
    const std::vector<int>& stitch_joint_indices,
    int half_window_points = 10)
{
    if (lane.points.size() < 3 || stitch_joint_indices.empty()) return;
    const int n = static_cast<int>(lane.points.size());
    const int w = std::max(1, half_window_points);

    for (int j : stitch_joint_indices) {
        if (j < 0 || j >= n) continue;
        const int l = std::max(0, j - w);
        const int r = std::min(n - 1, j + w);
        if (r - l < 2) continue;

        const double yaw_l = lane.points[l].yaw;
        const double yaw_r = lane.points[r].yaw;
        const double k_l = lane.points[l].kappa;
        const double k_r = lane.points[r].kappa;

        for (int i = l; i <= r; i++) {
            const double t = static_cast<double>(i - l) /
                             static_cast<double>(r - l);
            lane.points[i].yaw = normalizeAngle(
                yaw_l + t * normalizeAngle(yaw_r - yaw_l));
            lane.points[i].kappa = (1.0 - t) * k_l + t * k_r;
        }
    }
}

LanePolyline buildLaneFromProjection(
    const LanePolyline& lane,
    double x,
    double y,
    double yaw)
{
    LanePolyline out;
    out.token = lane.token;
    if (lane.points.size() < 2) return out;

    LaneProjectionResult proj = projectPointToLane(lane, x, y, yaw);
    if (!proj.valid) return out;
    if (proj.seg_idx < 0 ||
        proj.seg_idx + 1 >= static_cast<int>(lane.points.size())) return out;

    const auto& p0 = lane.points[proj.seg_idx];
    const auto& p1 = lane.points[proj.seg_idx + 1];
    LanePolyline::Point start;
    start.x = proj.nearest_x;
    start.y = proj.nearest_y;
    start.yaw = normalizeAngle(
        p0.yaw + proj.seg_t * normalizeAngle(p1.yaw - p0.yaw));
    if (!std::isfinite(start.yaw)) {
        start.yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);
    }
    start.kappa = p0.kappa + proj.seg_t * (p1.kappa - p0.kappa);
    out.points.push_back(start);

    for (size_t i = static_cast<size_t>(proj.seg_idx) + 1;
         i < lane.points.size(); i++) {
        out.points.push_back(lane.points[i]);
    }

    if (out.points.size() >= 2) {
        const double d01 = std::hypot(out.points[1].x - out.points[0].x,
                                      out.points[1].y - out.points[0].y);
        if (d01 < 1e-4) {
            out.points.erase(out.points.begin() + 1);
        }
    }
    out.length_m = computePolylineLength(out);
    return out;
}

PredTrajectory predictSingleTrack(const TrackKinematicState& track,
                                  const std::vector<LanePolyline>& lanes,
                                  const PredictionConfig& cfg)
{
    PredTrajectory out;
    out.track_id = track.track_id;
    out.mode_idx = track.mode_idx;

    double dt = std::max(cfg.step_s, 1e-3);
    double horizon = std::max(cfg.horizon_s, dt);
    int n_steps = std::max(1, static_cast<int>(std::round(horizon / dt)));

    double x = track.x;
    double y = track.y;
    double yaw = normalizeAngle(track.yaw);
    double v = std::max(0.0, track.v);
    double a_track = clampValue(track.accel, -6.0, 3.5);
    double yaw_rate_track = clampValue(track.yaw_rate, -1.2, 1.2);
    const char* model_str = (track.mode_idx == 0) ? "CV" :
                            (track.mode_idx == 1) ? "CTRV" :
                            (track.mode_idx == 3) ? "CA" :
                            (track.mode_idx == 2) ? "RM" : "UNK";

    pcl::PointXYZ p0;
    p0.x = static_cast<float>(x);
    p0.y = static_cast<float>(y);
    p0.z = 0.15f;
    out.points.push_back(p0);

    // Fixed reference line: choose at t0 and keep it for the full horizon.
    LaneChoiceResult init_choice = chooseReferenceLane(
        lanes, x, y, yaw, track.mode_idx, yaw_rate_track);
    if (init_choice.lane == nullptr) {
        ROS_INFO_STREAM("[PredLane] track=" << track.track_id
                        << " model=" << model_str
                        << " selected_lane_token=<none>");
        return out;
    }

    ROS_INFO_STREAM("[PredLane] track=" << track.track_id
                    << " model=" << model_str
                    << " selected_lane_token="
                    << init_choice.lane->token);

    LanePolyline fixed_lane = buildLaneFromProjection(*init_choice.lane, x, y, yaw);
    if (fixed_lane.points.size() < 2) return out;
    const bool short_lane_mode = (fixed_lane.length_m + 1e-6 < cfg.min_ref_length_m);
    const double end_stop_margin_m = 0.8;
    const double end_target_buffer_m = 0.3;
    const double kappa_avg_window_m = 8.0;
    const double kappa_lookahead_gain = 12.0;

    std::vector<std::string> stitched_tokens{fixed_lane.token};
    std::vector<double> stitch_join_dists_m;
    std::vector<int> stitch_joint_indices;
    if (fixed_lane.length_m < cfg.min_ref_length_m) {
        fixed_lane = stitchLaneByNearestEndpoint(
            fixed_lane, lanes, cfg.min_ref_length_m, cfg.stitch_max_gap_m,
            &stitched_tokens, &stitch_join_dists_m, &stitch_joint_indices);
        smoothLaneYawKappaAtJoints(fixed_lane, stitch_joint_indices, 10);
        fixed_lane.length_m = computePolylineLength(fixed_lane);
    }
    if (fixed_lane.points.size() < 2) return out;

    out.selected_lane_tokens.clear();
    for (const auto& tok : stitched_tokens) {
        if (!tok.empty() &&
            std::find(out.selected_lane_tokens.begin(),
                      out.selected_lane_tokens.end(),
                      tok) == out.selected_lane_tokens.end()) {
            out.selected_lane_tokens.push_back(tok);
        }
    }

    std::ostringstream chain_ss;
    for (size_t i = 0; i < stitched_tokens.size(); i++) {
        if (i > 0) chain_ss << " -> ";
        chain_ss << stitched_tokens[i];
    }
    std::ostringstream dist_ss;
    for (size_t i = 0; i < stitch_join_dists_m.size(); i++) {
        if (i > 0) dist_ss << ", ";
        dist_ss << stitch_join_dists_m[i];
    }
    ROS_INFO_STREAM("[PredLane] track=" << track.track_id
                    << " model=" << model_str
                    << " stitch_chain=" << chain_ss.str()
                    << " stitch_join_dist_m=[" << dist_ss.str() << "]"
                    << " merged_len_m=" << fixed_lane.length_m);

    for (int k = 0; k < n_steps; k++) {
        LaneProjectionResult proj = projectPointToLane(fixed_lane, x, y, yaw);
        if (!proj.valid) break;

        const double remain_to_end = computeRemainingDistanceToLaneEnd(fixed_lane, proj);
        if (short_lane_mode && remain_to_end <= end_stop_margin_m) {
            pcl::PointXYZ pend;
            pend.x = static_cast<float>(fixed_lane.points.back().x);
            pend.y = static_cast<float>(fixed_lane.points.back().y);
            pend.z = 0.15f;
            if (out.points.empty() ||
                std::hypot(out.points.back().x - pend.x, out.points.back().y - pend.y) > 0.1) {
                out.points.push_back(pend);
            }
            break;
        }

        const double avg_abs_kappa = computeLocalAvgAbsKappa(
            fixed_lane, proj, kappa_avg_window_m);
        const double lookahead_raw = 2.5 + 0.5 * v - kappa_lookahead_gain * avg_abs_kappa;
        double lookahead = clampValue(lookahead_raw, 2.0, 12.0);
        if (short_lane_mode && std::isfinite(remain_to_end)) {
            lookahead = std::min(lookahead, std::max(remain_to_end - end_target_buffer_m, 0.3));
        }
        double lx = x, ly = y;
        if (!findLookaheadPoint(fixed_lane, proj, lookahead, lx, ly)) break;

        // Lateral pure pursuit, blended with tracked yaw-rate.
        double heading_to_ref = atan2(ly - y, lx - x);
        double alpha = normalizeAngle(heading_to_ref - yaw);
        double yaw_rate_pp = 2.0 * std::max(v, 0.1) * sin(alpha) /
                             std::max(lookahead, 1.0);
        double cte = proj.lateral_sign * proj.dist;
        double yaw_rate_cte = clampValue(0.45 * cte, -0.6, 0.6);
        double yaw_rate_lane = yaw_rate_pp + yaw_rate_cte;
        // Keep CTRV lane-following aggressiveness consistent with CV.
        double w_track = 0.25;
        if (std::fabs(cte) > 1.5) {
            w_track *= 0.5;
        }
        double yaw_rate_cmd = w_track * yaw_rate_track + (1.0 - w_track) * yaw_rate_lane;
        yaw_rate_cmd = clampValue(yaw_rate_cmd, -1.2, 1.2);

        // Longitudinal IDM free-road term, blended with tracked acceleration.
        double v_des = std::max(8.0, track.v + 2.0);
        double a_max = clampValue(std::fabs(a_track) + 0.8, 0.8, 3.0);
        double ratio = std::max(0.0, v / std::max(v_des, 0.1));
        double a_idm = a_max * (1.0 - std::pow(ratio, 4.0));
        double a_cmd = 0.85 * a_track + 0.15 * a_idm;
        a_cmd = clampValue(a_cmd, -6.0, 3.5);

        v = std::max(0.0, v + a_cmd * dt);
        yaw = normalizeAngle(yaw + yaw_rate_cmd * dt);
        x += v * cos(yaw) * dt;
        y += v * sin(yaw) * dt;

        pcl::PointXYZ p;
        p.x = static_cast<float>(x);
        p.y = static_cast<float>(y);
        p.z = 0.15f;
        out.points.push_back(p);
    }

    return out;
}

}  // namespace

std::vector<PredTrajectory> buildVehiclePredictions(
    const std::vector<LanePolyline>& lanes,
    const std::vector<RoadPolygon>& road_polygons,
    const std::vector<DetCategoryPoint>& detections,
    const std::vector<TrackKinematicState>& tracks,
    const PredictionConfig& cfg)
{
    std::vector<PredTrajectory> preds;
    if (lanes.empty()) return preds;

    for (const auto& track : tracks) {
        if (!track.valid || track.track_manage <= 0 || track.is_static) continue;
        if (!(track.mode_idx == 0 || track.mode_idx == 1 || track.mode_idx == 3)) continue;

        if (!inferVehicleTrackByNearestDetection(detections, track.x, track.y)) continue;
        if (!road_polygons.empty() && !isOnRoad(road_polygons, track.x, track.y)) continue;
        LaneChoiceResult init_choice = chooseReferenceLane(
            lanes, track.x, track.y, track.yaw, track.mode_idx, track.yaw_rate);
        if (init_choice.lane == nullptr) continue;
        if (init_choice.nearest_valid_dist > 4.0) continue;

        PredTrajectory pred = predictSingleTrack(track, lanes, cfg);
        if (pred.points.size() >= 2) {
            preds.push_back(pred);
        }
    }

    return preds;
}
