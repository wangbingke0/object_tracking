#ifndef MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H
#define MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace pcl;

Eigen::VectorXd getCpFromBbox_(PointCloud<PointXYZI> bBox);
Eigen::VectorXd getCpFromBbox(PointCloud<PointXYZ> bBox);
double getBboxArea_(PointCloud<PointXYZI> bBox);
double getBboxArea(PointCloud<PointXYZ> bBox);
void immUkfJpdaf(vector<PointCloud<PointXYZ>> bBoxes, double timestamp,double car_yaw,
                     PointCloud<PointXYZ>& targetPoints, vector<vector<double>>& targetVandYaw,
                     vector<int>& trackManage, vector<bool>& isStaticVec,
                     vector<bool>& isVisVec, vector<PointCloud<PointXYZ>>& visBBs ,PointXYZ &car_pos , vector<PointCloud<PointXYZI>> &his_nearvisBBs,
                     vector<int>* trackIds = nullptr);




#endif /* MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H */
