
#ifndef MY_PCL_TUTORIAL_COMPONENT_CLUSTERING_H
#define MY_PCL_TUTORIAL_COMPONENT_CLUSTERING_H

#include <array>
#include <pcl/io/pcd_io.h>
#include <nav_msgs/OccupancyGrid.h>
#include <planning_msgs/Obstacle_tracking.h>
#include <planning_msgs/ObstacleList_tracking.h>
using namespace std;
using namespace pcl;
const float grid_size = 0.1;  

extern int numGrid_x;
extern int numGrid_y;

extern int min_x;
extern int min_y;
extern int max_x;
extern int max_y;

const int kernelSize = 3;

// 下面什么参数？
const double g_resolution = 1.0;
const int g_cell_width =60;  // 宽 
const int g_cell_height=60;
const double g_offset_x=0;
const double g_offset_y = 30;
const double g_offset_z = -0.8;

const double HEIGHT_LIMIT = 0.8;  // from sensor
const double CAR_LENGTH = 4.5;  // 车的宽高
const double CAR_WIDTH = 2;
//costmap paramter

void setNumGrid_x(int value);
void setNumGrid_y(int value);
void setMin_x(int value);
void setMin_y(int value);
void setMax_x(int value);
void setMax_y(int value);

void componentClustering(PointCloud<pcl::PointXYZ>::Ptr elevatedCloud,
                         std::vector<std::vector<int>> &cartesianData,
                         int & numCluster);

void mapCartesianGrid(PointCloud<PointXYZ>::Ptr elevatedCloud,
                            std::vector<std::vector<int>> &cartesianData);

void makeClusteredCloud(PointCloud<pcl::PointXYZ>::Ptr& elevatedCloud,
                        std::vector<std::vector<int>> &cartesianData,
                        PointCloud<pcl::PointXYZ>::Ptr& clusterCloud);

void setOccupancyGrid(nav_msgs::OccupancyGrid *og);

std::vector<int> createCostMap(const pcl::PointCloud<pcl::PointXYZ> &scan);

//void makeClusterVector(PointCloud<pcl::PointXYZ>::Ptr& elevatedCloud,
//                       array<array<int, numGrid>, numGrid> cartesianData,
//                       vector<PointCloud<pcl::PointXYZ>>& clusteredObjects);

#endif //TEST1_COMPONENT_CLUSTERING_H
