//xugong5 problem:
//1 tracking miss
//2 tf_old_data and process die
//xugong6 solve tf_old_data problem , tf::TransformListener lr(ros::Duration(100));
// changshu_2 base on xugong6 ,change parameter  related with vehicle ,Lidar height

#include <ros/ros.h>

#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Imu.h"
#include <visualization_msgs/Marker.h>

// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include "pcl_ros/impl/transforms.hpp"
// #include <pcl_ros/transforms.h>
#include <ros/package.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>

#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "tf/transform_datatypes.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include "geometry_msgs/TransformStamped.h"
// #include <tf2_ros/buffer.h>
// #include <tf2/transform_datatypes.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>

// #include <message_filters/subscriber.h>  
// #include <message_filters/time_synchronizer.h>  
// #include <message_filters/sync_policies/approximate_time.h>

#include "std_msgs/Float32MultiArray.h"
#include "imm_ukf_jpda.h"
#include "planning_msgs/Obstacle_tracking.h"
#include "planning_msgs/ObstacleList_tracking.h"
#include "planning_msgs/point.h"
#include "planning_msgs/Obstacle.h"
#include "planning_msgs/ObstacleList.h"

using namespace std;
using namespace Eigen;
using namespace pcl;
// using namespace message_filters;

string global_link = "velodyne";
string car_link = "car";

ros::Publisher pub;
ros::Publisher pub_obstacleList;
ros::Publisher vis_pub;
ros::Publisher vis_pub2;
tf::TransformListener* tran;
// tf::TransformListener* tran2;
vector<int> move_id;
vector<PointCloud<PointXYZ>> track_trajectory_total;
vector<bool> isnewPoint;
vector<PointCloud<PointXYZI>> His_nearvisBBs;
vector<PointCloud<PointXYZ>> map_visBBs;
Eigen::Matrix4f car_matrix;
double map_dis = 7;
double car_roll;
double car_pitch;
double car_yaw;

// struct Point {
//     double x, y;
// };

// // Function to compute the area of a polygon using the Shoelace formula
// double polygonArea(Point vertices[], int n) {
//     double area = 0.0;
//     int j = n - 1;  // Previous vertex index

//     for (int i = 0; i < n; i++) {
//         area += (vertices[j].x + vertices[i].x) * (vertices[j].y - vertices[i].y);
//         j = i;  // Update previous vertex index
//     }

//     return abs(area) / 2.0;
// }

// // Function to check if a point lies inside a rectangle defined by its vertices
// bool pointInsideRectangle(Point p, Point rect[4]) {
//     // Using cross product to determine if point p is inside rectangle rect
//     // Assume rect vertices are in counterclockwise order

//     // Check cross products for each edge of the rectangle
//     bool inside = true;
//     for (int i = 0; i < 4; ++i) {
//         Point v1 = rect[i];
//         Point v2 = rect[(i + 1) % 4];

//         // Calculate cross product (v2 - v1) x (p - v1)
//         double cross_product = (v2.x - v1.x) * (p.y - v1.y) - (v2.y - v1.y) * (p.x - v1.x);

//         if (cross_product < 0) {
//             inside = false;
//             break;
//         }
//     }

//     return inside;
// }

// // Function to compute intersection area of two parallelograms
// double intersectionArea(PointCloud<PointXYZ> par1, PointCloud<PointXYZ> par2) {
//     Point vertices1[4];
//     Point vertices2[4];

//     // Get vertices of par1
//     for (int i = 0; i < 4; ++i) {
//         vertices1[i].x = par1.points[i].x;
//         vertices1[i].y = par1.points[i].y;
//     }

//     // Get vertices of par2
//     for (int i = 0; i < 4; ++i) {
//         vertices2[i].x = par2.points[i].x;
//         vertices2[i].y = par2.points[i].y;
//     }

//     // Calculate intersection vertices (vertices3)
//     Point vertices3[4]; // Intersection vertices

//     // Calculate intersection area using polygon area calculation
//     double area = polygonArea(vertices3, 4);

//     return area;
// }

double intersectionArea(PointCloud<PointXYZ> quad1, PointCloud<PointXYZ> quad2) {
    // 计算四边形1的最小外包矩形坐标

    double x1_min = std::min({quad1[0].x, quad1[1].x, quad1[2].x, quad1[3].x});
    double y1_min = std::min({quad1[0].y, quad1[1].y, quad1[2].y, quad1[3].y});
    double x1_max = std::max({quad1[0].x, quad1[1].x, quad1[2].x, quad1[3].x});
    double y1_max = std::max({quad1[0].y, quad1[1].y, quad1[2].y, quad1[3].y});
    // 计算四边形2的最小外包矩形坐标
    double x2_min = std::min({quad2[0].x, quad2[1].x, quad2[2].x, quad2[3].x});
    double y2_min = std::min({quad2[0].y, quad2[1].y, quad2[2].y, quad2[3].y});
    double x2_max = std::max({quad2[0].x, quad2[1].x, quad2[2].x, quad2[3].x});
    double y2_max = std::max({quad2[0].y, quad2[1].y, quad2[2].y, quad2[3].y});

    // 计算交集的左下角和右上角坐标
    double inter_x1 = std::max(x1_min, x2_min);
    double inter_y1 = std::max(y1_min, y2_min);
    double inter_x2 = std::min(x1_max, x2_max);
    double inter_y2 = std::min(y1_max, y2_max);

    // 计算交集的宽度和高度
    double inter_width = inter_x2 - inter_x1;
    double inter_height = inter_y2 - inter_y1;

    // 如果交集为非正，则返回0
    if (inter_width <= 0 || inter_height <= 0) {
        return 0.0;
    }

    // 计算交集面积
    double interArea = inter_width * inter_height;
    return interArea;
}

// 计算两个四边形的并集面积
double unionArea(PointCloud<PointXYZ> quad1, PointCloud<PointXYZ> quad2) {
    // 计算四边形1的面积
    double area1 = 0.5 * std::abs((quad1[0].x * (quad1[1].y - quad1[3].y)) + 
                                  (quad1[1].x * (quad1[3].y - quad1[0].y)) + 
                                  (quad1[3].x * (quad1[0].y - quad1[1].y)));

    // 计算四边形2的面积
    double area2 = 0.5 * std::abs((quad2[0].x * (quad2[1].y - quad2[3].y)) + 
                                  (quad2[1].x * (quad2[3].y - quad2[0].y)) + 
                                  (quad2[3].x * (quad2[0].y - quad2[1].y)));

    // // 计算并集面积
    double unionArea = area1 + area2 - intersectionArea(quad1, quad2);
    return unionArea;
    //     double area = 0.5 * abs(quad1[0].x * quad1[1].y + quad1[1].x * quad1[2].y + quad1[2].x * quad1[3].y + quad1[3].x *
    //      quad1[0].y- (quad1[0].y * quad1[1].x + quad1[1].y * quad1[2].x + quad1[2].y * quad1[3].x + quad1[3].y * quad1[0].x));
    // return area;

}

// 计算两个四边形的相交比（IoU）
double intersectionOverUnion(PointCloud<PointXYZ> quad1, PointCloud<PointXYZ> quad2) {
    double interArea = intersectionArea(quad1, quad2);

    double unionAreaVal = getBboxArea(quad1);

   
    if (unionAreaVal == 0.0) {
        return 0.0;  // 避免除以零的情况
    }

    double iou = interArea / unionAreaVal;
    return iou;
}


void map_obstacle_load()
{
  std::string object_track_path = ros::package::getPath("object_tracking");
  //加载txt文件
  std::string map_cluster_flie = object_track_path+"/file/map_cluster.txt";
  string line;
  fstream file_;
  file_.open(map_cluster_flie,ios::in|ios::out);
  while (getline(file_, line))
  {
    pcl::PointXYZ points1;
    pcl::PointXYZ points2;
    pcl::PointXYZ points3;
    pcl::PointXYZ points4;
    float Z;
    std::istringstream iss(line);
    if (!(iss >> points1.x >> points1.y >> points2.x >> points2.y >> points3.x >> points3.y >> points4.x >> points4.y >> Z))
    {
        std::cout << "Error reading line: " << line << std::endl;
        continue;
    }
    pcl::PointCloud<pcl::PointXYZ> obstacle_single;
    points1.z = points2.z = points3.z = points4.z =-0.8;
    obstacle_single.points.push_back(points1);
    obstacle_single.points.push_back(points2);
    obstacle_single.points.push_back(points3);
    obstacle_single.points.push_back(points4);
    points1.z = points2.z = points3.z = points4.z =Z-0.8;
    obstacle_single.points.push_back(points1);
    obstacle_single.points.push_back(points2);
    obstacle_single.points.push_back(points3);
    obstacle_single.points.push_back(points4);
    map_visBBs.push_back(obstacle_single);
  }
  cout << "成功导入地图障碍物" << endl;
  cout << "地图障碍物的数量为"<<map_visBBs.size()<<endl;
}

void car_matrix_cb(const std_msgs::Float32MultiArrayConstPtr &array)
{
  for(int i = 0; i < car_matrix.rows(); ++i) 
  {
    for (int j = 0; j < car_matrix.cols(); ++j)
    {
      car_matrix(i, j) = array->data[i * 4 + j];
    }
  }
  tf::Matrix3x3 mat_l;
  mat_l.setValue(static_cast<double>(car_matrix(0, 0)), static_cast<double>(car_matrix(0, 1)), static_cast<double>(car_matrix(0, 2)),
                 static_cast<double>(car_matrix(1, 0)), static_cast<double>(car_matrix(1, 1)), static_cast<double>(car_matrix(1, 2)),
                 static_cast<double>(car_matrix(2, 0)), static_cast<double>(car_matrix(2, 1)), static_cast<double>(car_matrix(2, 2)));
  mat_l.getRPY(car_roll,car_pitch,car_yaw);

  // cout<<"car_roll ,car_pitch , car_yaw =  "<<car_roll<<" "<<car_pitch<<" "<<car_yaw<<endl;
}

// 回调函数-- 话题 track_box
void  cloud_cb (const planning_msgs::ObstacleList_tracking& input){

ros::Time start_time = ros::Time::now();
  static int count = 0;
  if(count<100)  count++;

  // convert local to global-------------------------
  double timestamp = input.header.stamp.toSec(); 
  ros::Time input_time = input.header.stamp;
  planning_msgs::ObstacleList ObstacleList;
  int Obstacle_id = 0;
  //发布车身和世界坐标系的TF变换
  static tf::TransformBroadcaster br;
  geometry_msgs::TransformStamped tfs;
    //  |----头设置
    tfs.header.frame_id = global_link;
    tfs.header.stamp = input_time;
    // cout<<"stamp:"<<input_time<<endl;
    //  |----坐标系 ID
    tfs.child_frame_id = car_link;

    //  |----坐标系相对信息设置
    tfs.transform.translation.x = car_matrix(0,3);
    tfs.transform.translation.y = car_matrix(1,3);
    tfs.transform.translation.z = 0.0; 
    //  |--------- 四元数设置
    tf2::Quaternion qtn;
    qtn.setRPY(0,0,car_yaw);
    tfs.transform.rotation.x = qtn.getX();
    tfs.transform.rotation.y = qtn.getY();
    tfs.transform.rotation.z = qtn.getZ();
    tfs.transform.rotation.w = qtn.getW();
    //  5-3.广播器发布数据
    br.sendTransform(tfs);

  PointXYZ car_pos;
  car_pos.x = car_matrix(0,3);
  car_pos.y = car_matrix(1,3);

  int box_num = input.num;

  //设置障碍物框的高度
  vector<PointCloud<PointXYZ>> bBoxes;
  PointCloud<PointXYZ> oneBbox;
  for(int box_i = 0;box_i < box_num; box_i++)
  {
      PointXYZ o;
      o.x = input.obstacle[box_i].x1;
      o.y = input.obstacle[box_i].y1;
      o.z = -0.8;
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x2;
      o.y = input.obstacle[box_i].y2;
      o.z = -0.8;   
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x3;
      o.y = input.obstacle[box_i].y3;
      o.z = -0.8;   
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x4;
      o.y = input.obstacle[box_i].y4;
      o.z = -0.8;
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x1;
      o.y = input.obstacle[box_i].y1;
      o.z = input.obstacle[box_i].maxZ;
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x2;
      o.y = input.obstacle[box_i].y2;
      o.z = input.obstacle[box_i].maxZ;
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x3;
      o.y = input.obstacle[box_i].y3;
      o.z = input.obstacle[box_i].maxZ;
      oneBbox.push_back(o);
      o.x = input.obstacle[box_i].x4;
      o.y = input.obstacle[box_i].y4;
      o.z = input.obstacle[box_i].maxZ;
      oneBbox.push_back(o);
      bBoxes.push_back(oneBbox);
      oneBbox.clear();
  }
  
  //由车身坐标系转为世界坐标系
  PointCloud<PointXYZ> newBox;
  for(int i = 0; i < bBoxes.size(); i++ ){
   bBoxes[i].header.frame_id = car_link;

   tran->waitForTransform(global_link, car_link, input_time, ros::Duration(0.1));
    pcl_ros::transformPointCloud(global_link, bBoxes[i], newBox, *tran);
    bBoxes[i] = newBox;
  }
  //得到为处理的数据
  PointCloud<PointXYZ> targetPoints_temp;
  vector<vector<double>> targetVandYaw_temp;
  vector<int> trackManage_temp;  // trackManage???  大量具有相关不确定性的跟踪对象需要有效地实施跟踪管理。跟踪管理的主要目的是动态限制虚假跟踪列表的数量（从而防止错误的数据关联），并在丢失检测的情况下保持对象跟踪
  vector<bool> isStaticVec_temp;
  vector<bool> isVisVec_temp;
  vector<PointCloud<PointXYZ>> visBBs_temp;
  ros::Time start_time1 = ros::Time::now();
  immUkfJpdaf(bBoxes, timestamp, car_yaw, targetPoints_temp, targetVandYaw_temp, trackManage_temp, isStaticVec_temp, isVisVec_temp, visBBs_temp,car_pos, His_nearvisBBs);
      ros::Time end_time1 = ros::Time::now();
    ros::Duration duration1 = end_time1 - start_time1;

    // 打印程序运行时间
    cout<<"程序运行时间1:"<<duration1.toSec()<<endl;
  // assert(targetPoints.size()== targetVandYaw.size());
  // assert(targetPoints.size()== trackManage.size());
  // assert(targetPoints.size()== isStaticVec.size());
  // assert(targetPoints.size()== isVisVec.size());
  // assert(targetPoints.size()== visBBs.size());
  vector<PointCloud<PointXYZ>> map_visBBs_;
  //找到地图上的障碍物
  for (size_t i = 0; i < map_visBBs.size(); i++)
  {
    VectorXd cp = getCpFromBbox(map_visBBs[i]);
    float dis = sqrt((cp(0)-car_matrix(0,3))*(cp(0)-car_matrix(0,3))+(cp(1)-car_matrix(1,3))*(cp(1)-car_matrix(1,3)));
    if(dis<map_dis)
    {
       planning_msgs::Obstacle Obstacle;
      for (size_t j = 0; j < 4; j++)
      {
        Obstacle.bounding_boxs[j].x = map_visBBs[i].points[j].x;
        Obstacle.bounding_boxs[j].y = map_visBBs[i].points[j].y;
      }
      Obstacle.number = Obstacle_id;
      Obstacle.isMapObs = true;
      Obstacle_id++;
      ObstacleList.obstacles.push_back(Obstacle);//发布地图上的障碍物
      map_visBBs_.push_back(map_visBBs[i]);
    }
  }
  //删去重复的障碍物
  PointCloud<PointXYZ> targetPoints;
  vector<vector<double>> targetVandYaw;
  vector<int> trackManage;  // trackManage???  大量具有相关不确定性的跟踪对象需要有效地实施跟踪管理。跟踪管理的主要目的是动态限制虚假跟踪列表的数量（从而防止错误的数据关联），并在丢失检测的情况下保持对象跟踪
  vector<bool> isStaticVec;
  vector<bool> isVisVec;
  vector<PointCloud<PointXYZ>> visBBs;
  
    for (size_t i = 0; i < visBBs_temp.size(); i++)
    {
      if(visBBs_temp[i].points.size()!=8) 
      {
        continue;
      }
      float minD = std::numeric_limits<double>::max();
      int id = 0;
      VectorXd cp = getCpFromBbox(visBBs_temp[i]);

      for (size_t j = 0; j < map_visBBs_.size(); j++)
      {
        VectorXd cp1 = getCpFromBbox(map_visBBs_[j]);
        float dis = sqrt((cp(0)-cp1(0))*(cp(0)-cp1(0))+(cp(1)-cp1(1))*(cp(1)-cp1(1)));
        if(dis<minD)
        {
          minD = dis;
          id = j;
        }
      }
      double intersection = 0;

      if (minD<5&&map_visBBs_.size())
      {
        intersection = intersectionOverUnion(visBBs_temp[i],map_visBBs_[id]);
      }
      if(intersection == 0)
      {
        targetPoints.push_back(targetPoints_temp[i]);
        targetVandYaw.push_back(targetVandYaw_temp[i]);
        trackManage.push_back(trackManage_temp[i]);
        isStaticVec.push_back(isStaticVec_temp[i]);
        isVisVec.push_back(isVisVec_temp[i]);
        visBBs.push_back(visBBs_temp[i]);
      }
    }

  cout<<"His_nearvisBBs.size():"<<His_nearvisBBs.size()<<endl;
  vector<bool> isReserveHis_nearvisBBs;
  for (size_t i = 0; i < His_nearvisBBs.size(); i++)
  {
    PointCloud<PointXYZ> His_nearvisBBs_single;
    for(int k = 0 ; k < 4; k++)
    {
      PointXYZ point;
      point.x = His_nearvisBBs[i][k].x;
      point.y = His_nearvisBBs[i][k].y;
      point.z = His_nearvisBBs[i][k].z;
      His_nearvisBBs_single.push_back(point);
    }
    float minD = std::numeric_limits<double>::max();
    float minD1 = std::numeric_limits<double>::max();
    int id = 0;
    VectorXd cp = getCpFromBbox(His_nearvisBBs_single);
cout<<"1"<<endl;
    for (size_t j = 0; j < map_visBBs_.size(); j++)
    {
      VectorXd cp1 = getCpFromBbox(map_visBBs_[j]);
      float dis = sqrt((cp(0)-cp1(0))*(cp(0)-cp1(0))+(cp(1)-cp1(1))*(cp(1)-cp1(1)));
      if(dis<minD)
      {
        minD = dis;
        id = j;
      }
    }
    for (size_t j = 0; j < His_nearvisBBs.size()&&j!=i; j++)
    {
      VectorXd cp1 = getCpFromBbox_(His_nearvisBBs[j]);
      float dis = sqrt((cp(0)-cp1(0))*(cp(0)-cp1(0))+(cp(1)-cp1(1))*(cp(1)-cp1(1)));
      if(dis<minD1)
      {
        minD1 = dis;
      }
    }
    
    double intersection = 0;

      if (minD<5&&map_visBBs_.size())
      {
        intersection = intersectionOverUnion(His_nearvisBBs_single,map_visBBs_[id]);
      }
      if(intersection == 0&&minD1>0.2)
      {
        cout<<"His_nearvisBBs intersection:"<<intersection<<endl;
        isReserveHis_nearvisBBs.push_back(1);
      }else{
        isReserveHis_nearvisBBs.push_back(0);
      }

  }
    cout<<"222"<<endl;
  //用于显示
  vector<PointCloud<PointXYZ>> visBBs_;
  visBBs_.insert(visBBs_.end(), map_visBBs.begin(), map_visBBs.end());
  visBBs_.insert(visBBs_.end(), visBBs.begin(), visBBs.end());
  /*********************追踪点显示开始*************************/
  visualization_msgs::Marker pointsY, pointsG, pointsR, pointsB;
  // pointsY.header.frame_id = pointsG.header.frame_id = pointsR.header.frame_id = pointsB.header.frame_id = "velodyne";
  pointsY.header.frame_id = pointsG.header.frame_id = pointsR.header.frame_id = pointsB.header.frame_id = global_link;
  
  pointsY.header.stamp= pointsG.header.stamp= pointsR.header.stamp =pointsB.header.stamp = input_time;
  pointsY.ns= pointsG.ns = pointsR.ns =pointsB.ns=  "points";
  pointsY.action = pointsG.action = pointsR.action = pointsB.action = visualization_msgs::Marker::ADD;
  pointsY.pose.orientation.w = pointsG.pose.orientation.w  = pointsR.pose.orientation.w =pointsB.pose.orientation.w= 1.0;
  pointsY.pose.orientation.x = pointsG.pose.orientation.x  = pointsR.pose.orientation.x =pointsB.pose.orientation.x= 0.0;
  pointsY.pose.orientation.y = pointsG.pose.orientation.y  = pointsR.pose.orientation.y =pointsB.pose.orientation.y= 0.0;
  pointsY.pose.orientation.z = pointsG.pose.orientation.z  = pointsR.pose.orientation.z =pointsB.pose.orientation.z= 0.0;

  pointsY.id = 1;
  pointsG.id = 2;
  pointsR.id = 3;
  pointsB.id = 4;
  pointsY.type = pointsG.type = pointsR.type = pointsB.type = visualization_msgs::Marker::POINTS;

  // POINTS markers use x and y scale for width/height respectively
  pointsY.scale.x =pointsG.scale.x =pointsR.scale.x = pointsB.scale.x=0.5;
  pointsY.scale.y =pointsG.scale.y =pointsR.scale.y = pointsB.scale.y = 0.5;

  // yellow（红绿蓝混合为黄）
  pointsY.color.r = 1.0f;
  pointsY.color.g = 1.0f;
  pointsY.color.b = 0.0f;
  pointsY.color.a = 1.0;

  // green
  pointsG.color.g = 1.0f;
  pointsG.color.a = 1.0;

  // red
  pointsR.color.r = 1.0;
  pointsR.color.a = 1.0;

  // blue 
  pointsB.color.b = 1.0;
  pointsB.color.a = 1.0;

//  cout << "targetPoints.size() is --=------" << targetPoints.size() <<endl;
 
  for(int i = 0; i < targetPoints.size(); i++){
      if(trackManage[i] == 0) continue;
      geometry_msgs::Point p;
      p.x = targetPoints[i].x;
      p.y = targetPoints[i].y;
      p.z = 0;
      
      if (std::isnan(p.x) || std::isnan(p.y)) 
      {
        continue;
      }
      // 可以在这里进行处理，比如赋予默认值或者忽略这个点

     
        if(isStaticVec[i] == true){   // isStaticVec???
          pointsB.points.push_back(p);    // 蓝点
        }
        else if(trackManage[i] < 5 ){  // 小于5为黄点
          pointsY.points.push_back(p);
        }
        else if(trackManage[i] == 5){  // 等于5为绿点
          pointsG.points.push_back(p);
        }
        else if(trackManage[i] > 5){
          pointsR.points.push_back(p);    // 大于5为红点

    }
  
  }

  vis_pub.publish(pointsY);   // 发布
  vis_pub.publish(pointsG);  // 发布
  vis_pub.publish(pointsR);  // 发布
  vis_pub.publish(pointsB);  // 发布
  /*********************追踪点显示结束*************************/

  
  /*********************箭头显示开始*************************/
  for(int i = 0; i < targetPoints.size(); i++){
    visualization_msgs::Marker arrowsG;
    arrowsG.lifetime = ros::Duration(0.1);
    
    if(trackManage[i] == 0 ) {
      continue;
    }
    if(isVisVec[i] == false ) {
      continue;
    }
    if(isStaticVec[i] == true){
     
        planning_msgs::Obstacle Obstacle;
        for (size_t j = 0; j < 4; j++)
        {
          Obstacle.bounding_boxs[j].x = visBBs[i].points[j].x;
          Obstacle.bounding_boxs[j].y = visBBs[i].points[j].y;
        }
        Obstacle.number = Obstacle_id;
        Obstacle_id++;
        ObstacleList.obstacles.push_back(Obstacle);//发布实时检测的静态障碍物

      continue;
    }

    arrowsG.header.frame_id = global_link;
    arrowsG.header.stamp= input_time;
    arrowsG.ns = "arrows";
    arrowsG.action = visualization_msgs::Marker::ADD;
    arrowsG.type =  visualization_msgs::Marker::ARROW;
    // green  设置颜色
    arrowsG.color.g = 1.0f; // 绿色
    // arrowsG.color.r = 1.0f; // 红色
    arrowsG.color.a = 1.0;  
    arrowsG.id = i;
    geometry_msgs::Point p;
    // assert(targetPoints[i].size()==4);
    p.x = targetPoints[i].x;
    p.y = targetPoints[i].y;
    p.z = 0;  
    if (std::isnan(p.x) || std::isnan(p.y)) 
    {
       continue;
    }
    double tv   = targetVandYaw[i][0];
    double tyaw = targetVandYaw[i][1];

    planning_msgs::Obstacle Obstacle;
    for (size_t j = 0; j < 4; j++)
    {
      Obstacle.bounding_boxs[j].x = visBBs[i].points[j].x;
      Obstacle.bounding_boxs[j].y = visBBs[i].points[j].y;
    }
    Obstacle.number = Obstacle_id;
    Obstacle_id++;
    ObstacleList.obstacles.push_back(Obstacle);//发布实时检测的动态障碍物

    arrowsG.pose.position.x = p.x;
    arrowsG.pose.position.y = p.y;
    arrowsG.pose.position.z = p.z;

    // convert from 3 angles to quartenion
    tf::Matrix3x3 obs_mat;
    obs_mat.setEulerYPR(tyaw, 0, 0); // yaw, pitch, roll
    tf::Quaternion q_tf;
    obs_mat.getRotation(q_tf);
    arrowsG.pose.orientation.x = q_tf.getX();
    arrowsG.pose.orientation.y = q_tf.getY();
    arrowsG.pose.orientation.z = q_tf.getZ();
    arrowsG.pose.orientation.w = q_tf.getW();
    // cout<<"arrowsG.pose.orientation.w = q_tf.getW() = "<<arrowsG.pose.orientation.w<<endl;
    // Set the scale of the arrowsG -- 1x1x1 here means 1m on a side
    arrowsG.scale.x = tv;
    // arrowsG.scale.x = 0.1;
    arrowsG.scale.y = 0.1;
    arrowsG.scale.z = 0.1;
    vis_pub2.publish(arrowsG);  // 发布箭头消息
  }

  /*********************箭头显示结束*************************/

  /*********************聚类框显示开始*************************/
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = global_link;
  
  line_list.header.stamp = input_time;
  line_list.ns =  "boxes";
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;

  line_list.id = 0;
  line_list.type = visualization_msgs::Marker::LINE_LIST;

  //LINE_LIST markers use only the x component of scale, for the line width
  line_list.scale.x = 0.1;
  // Points are green
  line_list.color.g = 1.0f;
  line_list.color.a = 1.0;

  int id = 0;string ids;
  static int clear_num =0;
  static bool clear_flag = false;
  for (size_t i = His_nearvisBBs.size(); i > 0; )
  {
    --i;
    //float S = getBboxArea_(His_nearvisBBs[i]);
    // cout<<"S = "<<S<<endl;
    // if(!isReserveHis_nearvisBBs[i]) continue;

    if (His_nearvisBBs[i][0].intensity >= 50)
    {
        His_nearvisBBs.erase(His_nearvisBBs.begin() + i);
    }else if(His_nearvisBBs[i][0].intensity<50 && isReserveHis_nearvisBBs[i]){
      clear_flag = true;
      // cout<<"id: "<<i<<endl;
      planning_msgs::Obstacle Obstacle;
      geometry_msgs::Point p;
      for(int j = 0 ;j<4;j++)
      {
        
        Obstacle.bounding_boxs[j].x = His_nearvisBBs[i].points[j].x;
        Obstacle.bounding_boxs[j].y = His_nearvisBBs[i].points[j].y;
        p.x = His_nearvisBBs[i][j].x;
        p.y = His_nearvisBBs[i][j].y;
        p.z = 0;
        line_list.points.push_back(p);
        p.x = His_nearvisBBs[i][(j+1)%4].x;
        p.y = His_nearvisBBs[i][(j+1)%4].y;
        p.z = 0;
        line_list.points.push_back(p);
      }
      His_nearvisBBs[i].points[0].intensity++; 
      Obstacle.number = Obstacle_id;
      Obstacle_id++;
      ObstacleList.obstacles.push_back(Obstacle);//发布离车较近用于保留一秒的障碍物
    }
  }
  // //设置清空标志位，每十秒清空一次
  // if(clear_flag == false)
  // {
  //   clear_num++;
  // }
  // if(clear_num>1000)
  // {
  //   His_nearvisBBs.clear();
  //   clear_num = 0;
  // }

  ObstacleList.header = input.header;
  pub_obstacleList.publish(ObstacleList);

  for(int objectI = 0; objectI < visBBs_.size(); objectI ++){
    if(visBBs_[objectI].size()<4) continue; 
    for(int pointI = 0; pointI < 4; pointI++){
      assert((pointI+1)%4 < visBBs_[objectI].size());
      assert((pointI+4) < visBBs_[objectI].size());
      assert((pointI+1)%4+4 < visBBs_[objectI].size());

      id ++; ids = to_string(id);

      if (std::isnan(visBBs_[objectI][pointI].x)        ||std::isnan(visBBs_[objectI][pointI].y)||
          std::isnan(visBBs_[objectI][(pointI+1)%4].x)  ||std::isnan(visBBs_[objectI][(pointI+1)%4].y)||
          std::isnan(visBBs_[objectI][pointI+4].x)      ||std::isnan(visBBs_[objectI][pointI+4].y)||
          std::isnan(visBBs_[objectI][(pointI+1)%4+4].x)||std::isnan(visBBs_[objectI][(pointI+1)%4+4].y)) 
          {
            continue;
          }

      geometry_msgs::Point p;
      p.x = visBBs_[objectI][pointI].x;
      p.y = visBBs_[objectI][pointI].y;
      p.z = visBBs_[objectI][pointI].z;
      line_list.points.push_back(p);
      p.x = visBBs_[objectI][(pointI+1)%4].x;
      p.y = visBBs_[objectI][(pointI+1)%4].y;
      p.z = visBBs_[objectI][(pointI+1)%4].z;
      line_list.points.push_back(p);

      p.x = visBBs_[objectI][pointI].x;
      p.y = visBBs_[objectI][pointI].y;
      p.z = visBBs_[objectI][pointI].z;
      line_list.points.push_back(p);
      p.x = visBBs_[objectI][pointI+4].x;
      p.y = visBBs_[objectI][pointI+4].y;
      p.z = visBBs_[objectI][pointI+4].z;
      line_list.points.push_back(p);

      p.x = visBBs_[objectI][pointI+4].x;
      p.y = visBBs_[objectI][pointI+4].y;
      p.z = visBBs_[objectI][pointI+4].z;
      line_list.points.push_back(p);
      p.x = visBBs_[objectI][(pointI+1)%4+4].x;
      p.y = visBBs_[objectI][(pointI+1)%4+4].y;
      p.z = visBBs_[objectI][(pointI+1)%4+4].z;
      line_list.points.push_back(p);

    }
  }
  vis_pub.publish(line_list);
    ros::Time end_time = ros::Time::now();
    ros::Duration duration = end_time - start_time;
    static double time_max = 0;
    // 打印程序运行时间
    if(duration.toSec()>time_max)
    {
       time_max =  duration.toSec();
    }
    cout<<"程序运行时间:"<<duration.toSec()<<" "<<"最大时间："<<time_max<<endl;
  /*********************聚类框显示开始*************************/

}

int main (int argc, char** argv){
  // Initialize ROS
  ros::init (argc, argv, "obj_track"); // obj_track--节点
  ros::NodeHandle nh;

  tf::TransformListener lr(ros::Duration(10));         //(How long to store transform information)
  tran=&lr;
  map_obstacle_load();
  // Create a ROS subscriber for the input point cloud

  ros::Subscriber sub = nh.subscribe ("/ObstacleList_no_tf", 500, cloud_cb);   //订阅者  track_box -- 话题topic名（订阅边框八个点）
  ros::Subscriber sub2 = nh.subscribe ("/car_matrix", 500, car_matrix_cb);   //订阅者  /gps/odom -- 话题topic名

  pub_obstacleList = nh.advertise<planning_msgs::ObstacleList> ("obstacleList_lidar", 20);  //发布者  visualization_marker -- 话题topic名

  vis_pub = nh.advertise<visualization_msgs::Marker>( "visualization_marker", 50 );    //发布者  visualization_marker -- 话题topic名
  vis_pub2 = nh.advertise<visualization_msgs::Marker>( "visualization_marker2", 50 );  //发布者  visualization_marker2 -- 话题topic名

  ros::spin ();
}