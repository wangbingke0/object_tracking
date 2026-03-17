//
// Created by kosuke on 12/23/17.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <math.h>

#include "Ukf.h"
#include "imm_ukf_jpda.h"

using namespace std;
using namespace Eigen;
using namespace pcl;


bool init_= false;
double timestamp_ ;
double egoVelo_;
double egoYaw_;
double egoPreYaw_;
VectorXd preMeas_;

const double gammaG_ = 9.22; // 99%

//const double gammaG_ = 9.22; // 99%

const double pG_ = 0.99;//门概率
// const double gammaG_ = 5.99; // 99%
// const double pG_ = 0.95;
// const double gammaG_ = 15.22; // 99%
const double pD_ = 0.9; //检测概率

//bbox association param
//const double distanceThres_ = 0.25;
const double distanceThres_ = 2;
//const double distanceThres_ = 1;

const int lifeTimeThres_ = 0;
//const int lifeTimeThres_ = 8;

//bbox update params
const double bbYawChangeThres_  = 0.2; 
// const double bbYawChangeThres_  = 0.2; 
// const double bbAreaChangeThres_ = 0.2;
// 角速度(yaw_rate)超过此阈值时视为旋转，强制使用CTRV
const double YAW_RATE_THRESH_ = 0.2;  // rad/s, ~11.5 deg/s
const double bbAreaChangeThres_ = 0.5;
const double bbVelThres_        = 0.05;
const double bbDeltaVel_        = 0.01;
const int    nStepBack_         = 3;

vector<UKF> targets_;
vector<int> trackNumVec_;
int nextTrackId_ = 0;
double alpha = 0.35;

double filter(double x , double yaw_prev_ , double alpha) 
{
    double yaw_prev = (1 - alpha) * x + alpha * yaw_prev_;
    return yaw_prev;
}

// 计算数组的均值
double mean(const std::vector<double>& data) {
    double sum = 0.0;
    for (auto& num : data) {
        sum += num;
    }
    return sum / data.size();
}

// 计算数组的方差
double variance(const std::vector<double>& data) {
    double mean_value = mean(data);
    double var = 0.0;
    for (auto& num : data) {
        var += (num - mean_value) * (num - mean_value);
    }
    return var / data.size();
}

void findMaxZandS(UKF target, VectorXd& maxDetZ, MatrixXd& maxDetS){
    double cv_det   = target.lS_cv_.determinant();
    double ctrv_det = target.lS_ctrv_.determinant();
    double rm_det   = target.lS_rm_.determinant();
    double ca_det   = target.lS_ca_.determinant();

    double maxDet = cv_det;
    maxDetZ = target.zPredCVl_;
    maxDetS = target.lS_cv_;

    if(ctrv_det > maxDet) {
        maxDet = ctrv_det;
        maxDetZ = target.zPredCTRVl_;
        maxDetS = target.lS_ctrv_;
    }
    if(rm_det > maxDet) {
        maxDet = rm_det;
        maxDetZ = target.zPredRMl_;
        maxDetS = target.lS_rm_;
    }
    if(ca_det > maxDet) {
        maxDet = ca_det;
        maxDetZ = target.zPredCAl_;
        maxDetS = target.lS_ca_;
    }


}
//计算trackPoints的可行性，如果观测的点行的话，matchingVec = 1，并将该点加入到measVec的（x,y）中
void measurementValidation(vector<vector<double>> trackPoints, UKF& target, bool secondInit, VectorXd maxDetZ, MatrixXd maxDetS,
     vector<VectorXd>& measVec, vector<VectorXd>& bboxVec, vector<int>& matchingVec){
   
    int count = 0;
    bool secondInitDone = false;
    double smallestNIS = 999;
    VectorXd smallestMeas = VectorXd(2);
    for(int i = 0; i < trackPoints.size(); i++){
        double x = trackPoints[i][0];
        double y = trackPoints[i][1];
        VectorXd meas = VectorXd(2);
        meas << x, y;

        VectorXd bbox = VectorXd(11);
        bbox << x, y, 
                trackPoints[i][2], trackPoints[i][3],
                trackPoints[i][4], trackPoints[i][5], 
                trackPoints[i][6], trackPoints[i][7],
                trackPoints[i][8], trackPoints[i][9], trackPoints[i][10];

        VectorXd diff = meas - maxDetZ;//maxDet观测量的预测值
        double nis = diff.transpose()*maxDetS.inverse()*diff;

    //    cout << "size of nis is " << nis <<endl;
    //     cout <<"11111nis: " <<nis << endl;
        if(nis < gammaG_){ // x^2 99% range
            count ++;
            if(matchingVec[i] == 0) target.lifetime_ ++;

            // cout <<"nis: " <<nis << endl;
            // cout << "meas"<<endl<<meas << endl;

            // pick one meas with smallest nis
            if(secondInit){
                if(nis < smallestNIS)
                {
                    smallestNIS = nis;
                    smallestMeas = meas;
                    // measVec.push_back(meas);
                    matchingVec[i] = 1;
                    secondInitDone = true;
                }
            }
            else{
                measVec.push_back(meas);
                bboxVec.push_back(bbox);
                matchingVec[i] = 1;
            }
        }
        // cout << "meas"<<endl<<meas << endl;
    }
    if(secondInitDone) measVec.push_back(smallestMeas);
//     cout << "target.lifetime_ = "<<endl<<target.lifetime_ << endl;
//    cout << "size of bboxVec is " << bboxVec.size() <<endl;
}

void filterPDA(UKF& target, vector<VectorXd> measVec, vector<double>& lambdaVec){
    double numMeas = measVec.size();
    double b = 2*numMeas*(1-pD_*pG_)/(gammaG_*pD_);
    double eCVSum   = 0;
    double eCTRVSum = 0;
    double eRMSum   = 0;
    double eCASum   = 0;

    vector<double> eCvVec;
    vector<double> eCtrvVec;
    vector<double> eRmVec;
    vector<double> eCaVec;

    vector<VectorXd> diffCVVec;
    vector<VectorXd> diffCTRVVec;
    vector<VectorXd> diffRMVec;
    vector<VectorXd> diffCAVec;

    for(int i = 0; i < numMeas; i++){
        VectorXd diffCV   = measVec[i] - target.zPredCVl_;
        VectorXd diffCTRV = measVec[i] - target.zPredCTRVl_;
        VectorXd diffRM   = measVec[i] - target.zPredRMl_;
        VectorXd diffCA   = measVec[i] - target.zPredCAl_;

        diffCVVec.push_back(diffCV);
        diffCTRVVec.push_back(diffCTRV);
        diffRMVec.push_back(diffRM);
        diffCAVec.push_back(diffCA);

        double eCV   = exp(-0.5*diffCV.transpose()  *target.lS_cv_.inverse()  *diffCV);
        double eCTRV = exp(-0.5*diffCTRV.transpose()*target.lS_ctrv_.inverse()*diffCTRV);
        double eRM   = exp(-0.5*diffRM.transpose()  *target.lS_rm_.inverse()  *diffRM);
        double eCA   = exp(-0.5*diffCA.transpose()  *target.lS_ca_.inverse()  *diffCA);

        eCvVec.push_back(eCV);
        eCtrvVec.push_back(eCTRV);
        eRmVec.push_back(eRM);
        eCaVec.push_back(eCA);

        eCVSum   += eCV;
        eCTRVSum += eCTRV;
        eRMSum   += eRM;
        eCASum   += eCA;
    }
    double betaCVZero   = b/(b+eCVSum);
    double betaCTRVZero = b/(b+eCTRVSum);
    double betaRMZero   = b/(b+eRMSum);
    double betaCAZero   = b/(b+eCASum);

    vector<double> betaCV;
    vector<double> betaCTRV;
    vector<double> betaRM;
    vector<double> betaCA;

    for(int i = 0; i < numMeas; i++){
        betaCV.push_back(eCvVec[i]/(b+eCVSum));
        betaCTRV.push_back(eCtrvVec[i]/(b+eCTRVSum));
        betaRM.push_back(eRmVec[i]/(b+eRMSum));
        betaCA.push_back(eCaVec[i]/(b+eCASum));
    }

    VectorXd sigmaXcv, sigmaXctrv, sigmaXrm, sigmaXca;
    sigmaXcv.setZero(2);
    sigmaXctrv.setZero(2);
    sigmaXrm.setZero(2);
    sigmaXca.setZero(2);

    for(int i = 0; i < numMeas; i++){
        sigmaXcv   += betaCV[i]*diffCVVec[i];
        sigmaXctrv += betaCTRV[i]*diffCTRVVec[i];
        sigmaXrm   += betaRM[i]*diffRMVec[i];
        sigmaXca   += betaCA[i]*diffCAVec[i];
    }

    MatrixXd sigmaPcv, sigmaPctrv, sigmaPrm, sigmaPca;
    sigmaPcv.setZero(2,2);
    sigmaPctrv.setZero(2,2);
    sigmaPrm.setZero(2,2);
    sigmaPca.setZero(2,2);
    for(int i = 0; i < numMeas; i++){
        sigmaPcv   += (betaCV[i]  *diffCVVec[i]  *diffCVVec[i].transpose()   - sigmaXcv*sigmaXcv.transpose());
        sigmaPctrv += (betaCTRV[i]*diffCTRVVec[i]*diffCTRVVec[i].transpose() - sigmaXctrv*sigmaXctrv.transpose());
        sigmaPrm   += (betaRM[i]  *diffRMVec[i]  *diffRMVec[i].transpose()   - sigmaXrm*sigmaXrm.transpose());
        sigmaPca   += (betaCA[i]  *diffCAVec[i]  *diffCAVec[i].transpose()   - sigmaXca*sigmaXca.transpose());
    }

    target.x_cv_   = target.x_cv_   + target.K_cv_*sigmaXcv;
    target.x_ctrv_ = target.x_ctrv_ + target.K_ctrv_*sigmaXctrv;
    target.x_rm_   = target.x_rm_   + target.K_rm_*sigmaXrm;
    target.x_ca_   = target.x_ca_   + target.K_ca_*sigmaXca;
    while (target.x_cv_(3)> M_PI) target.x_cv_(3) -= 2.*M_PI;
    while (target.x_cv_(3)<-M_PI) target.x_cv_(3) += 2.*M_PI;
    while (target.x_ctrv_(3)> M_PI) target.x_ctrv_(3) -= 2.*M_PI;
    while (target.x_ctrv_(3)<-M_PI) target.x_ctrv_(3) += 2.*M_PI;
    while (target.x_rm_(3)> M_PI) target.x_rm_(3) -= 2.*M_PI;
    while (target.x_rm_(3)<-M_PI) target.x_rm_(3) += 2.*M_PI;
    while (target.x_ca_(3)> M_PI) target.x_ca_(3) -= 2.*M_PI;
    while (target.x_ca_(3)<-M_PI) target.x_ca_(3) += 2.*M_PI;

    if(numMeas != 0){
    	target.P_cv_   = betaCVZero*target.P_cv_ +
                  (1-betaCVZero)*(target.P_cv_ - target.K_cv_*target.lS_cv_*target.K_cv_.transpose()) +
                  target.K_cv_*sigmaPcv*target.K_cv_.transpose();
	    target.P_ctrv_ = betaCTRVZero*target.P_ctrv_ +
	                  (1-betaCTRVZero)*(target.P_ctrv_ - target.K_ctrv_*target.lS_ctrv_*target.K_ctrv_.transpose()) +
	                  target.K_ctrv_*sigmaPctrv*target.K_ctrv_.transpose();
	    target.P_rm_   = betaRMZero*target.P_rm_ +
	                  (1-betaRMZero)*(target.P_rm_ - target.K_rm_*target.lS_rm_*target.K_rm_.transpose()) +
	                  target.K_rm_*sigmaPrm*target.K_rm_.transpose();
	    target.P_ca_   = betaCAZero*target.P_ca_ +
	                  (1-betaCAZero)*(target.P_ca_ - target.K_ca_*target.lS_ca_*target.K_ca_.transpose()) +
	                  target.K_ca_*sigmaPca*target.K_ca_.transpose();
    }
    else{
    	target.P_cv_   = target.P_cv_   - target.K_cv_  *target.lS_cv_  *target.K_cv_.transpose();
	    target.P_ctrv_ = target.P_ctrv_ - target.K_ctrv_*target.lS_ctrv_*target.K_ctrv_.transpose();
	    target.P_rm_   = target.P_rm_   - target.K_rm_  *target.lS_rm_  *target.K_rm_.transpose();
	    target.P_ca_   = target.P_ca_   - target.K_ca_  *target.lS_ca_  *target.K_ca_.transpose();
    }

    VectorXd maxDetZ;
    MatrixXd maxDetS;
    findMaxZandS(target, maxDetZ, maxDetS);
    double Vk =  M_PI *sqrt(gammaG_ * maxDetS.determinant());

    double lambdaCV, lambdaCTRV, lambdaRM, lambdaCA;
    if(numMeas != 0){
    	lambdaCV   = (1 - pG_*pD_)/pow(Vk, numMeas) +
	                        pD_*pow(Vk, 1-numMeas)*eCVSum/(numMeas*sqrt(2*M_PI*target.lS_cv_.determinant()));
	    lambdaCTRV = (1 - pG_*pD_)/pow(Vk, numMeas) +
	                        pD_*pow(Vk, 1-numMeas)*eCTRVSum/(numMeas*sqrt(2*M_PI*target.lS_ctrv_.determinant()));
	    lambdaRM   = (1 - pG_*pD_)/pow(Vk, numMeas) +
	                        pD_*pow(Vk, 1-numMeas)*eRMSum/(numMeas*sqrt(2*M_PI*target.lS_rm_.determinant()));
	    lambdaCA   = (1 - pG_*pD_)/pow(Vk, numMeas) +
	                        pD_*pow(Vk, 1-numMeas)*eCASum/(numMeas*sqrt(2*M_PI*target.lS_ca_.determinant()));
    }
    else{
    	lambdaCV   = (1 - pG_*pD_)/pow(Vk, numMeas);
	    lambdaCTRV = (1 - pG_*pD_)/pow(Vk, numMeas);
	    lambdaRM   = (1 - pG_*pD_)/pow(Vk, numMeas);
	    lambdaCA   = (1 - pG_*pD_)/pow(Vk, numMeas);
    }
    lambdaVec.push_back(lambdaCV);
    lambdaVec.push_back(lambdaCTRV);
    lambdaVec.push_back(lambdaRM);
    lambdaVec.push_back(lambdaCA);
}

void getNearestEuclidBBox(UKF target, vector<VectorXd> bboxVec, vector<double>& Bbox, double& minDist){
    int minInd = 0;
    double px = target.x_merge_(0);
    double py = target.x_merge_(1);
    for (int i = 0; i < bboxVec.size(); i++){
        double measX = bboxVec[i](0);
        double measY = bboxVec[i](1);
        double dist = sqrt((px-measX)*(px-measX)+(py-measY)*(py-measY));
        if(dist < minDist){
            minDist = dist;
            minInd = i;
        }
    }
    assert(bboxVec[minInd].rows() == 11);
    for(int i = 0; i < 11; i++){
        Bbox.push_back(bboxVec[minInd](i));
    }
}


void associateBB(int trackNum, vector<VectorXd> bboxVec, UKF& target){
    //skip if no validated measurement

    // cout <<"bboxVec.size() is " << bboxVec.size() <<endl;
    // cout <<"target.lifetime_ is " << target.lifetime_ <<endl;
    // cout <<"trackNum_ is " << trackNum <<endl;
    if(bboxVec.size() == 0) {
        return;
    }

    
    if(trackNum == 5 && target.lifetime_ > lifeTimeThres_){
        vector<double> nearestBbox;
        double minDist = 999;
        getNearestEuclidBBox(target, bboxVec, nearestBbox, minDist);
        // cout <<"min dist is " << minDist <<endl;
        if(minDist < distanceThres_){
            PointCloud<PointXYZ> bbox;
            PointXYZ o;
            assert(nearestBbox.size() == 11);
            for(int i = 0; i < 2; i++){
                double height;
                if(i == 0) {height = -0.8;}
                else       {height = nearestBbox[10];}
                o.x = nearestBbox[2];
                o.y = nearestBbox[3];
                o.z = height;
                bbox.push_back(o);
                o.x = nearestBbox[4];
                o.y = nearestBbox[5];
                o.z = height;
                bbox.push_back(o);
                o.x = nearestBbox[6];
                o.y = nearestBbox[7];
                o.z = height;
                bbox.push_back(o);
                o.x = nearestBbox[8];
                o.y = nearestBbox[9];
                o.z = height;
                bbox.push_back(o);
            }
//            cout << " ********************************* "<< endl;
            target.isVisBB_ = true;
            target.BBox_    = bbox;
        }
    }
}

VectorXd getCpFromBbox(PointCloud<PointXYZ> bBox){
    PointXYZ p1 = bBox[0];
    PointXYZ p2 = bBox[1];
    PointXYZ p3 = bBox[2];
    PointXYZ p4 = bBox[3];

    double S1 = ((p4.x -p2.x)*(p1.y - p2.y) - (p4.y - p2.y)*(p1.x - p2.x))/2;
    double S2 = ((p4.x -p2.x)*(p2.y - p3.y) - (p4.y - p2.y)*(p2.x - p3.x))/2;
    double cx = p1.x + (p3.x-p1.x)*S1/(S1+S2);
    double cy = p1.y + (p3.y-p1.y)*S1/(S1+S2);

    VectorXd cp(2);
    cp << cx, cy;
    return cp;
}

VectorXd getCpFromBbox_(PointCloud<PointXYZI> bBox){
    PointXYZI p1 = bBox[0];
    PointXYZI p2 = bBox[1];
    PointXYZI p3 = bBox[2];
    PointXYZI p4 = bBox[3];

    double S1 = ((p4.x -p2.x)*(p1.y - p2.y) - (p4.y - p2.y)*(p1.x - p2.x))/2;
    double S2 = ((p4.x -p2.x)*(p2.y - p3.y) - (p4.y - p2.y)*(p2.x - p3.x))/2;
    double cx = p1.x + (p3.x-p1.x)*S1/(S1+S2);
    double cy = p1.y + (p3.y-p1.y)*S1/(S1+S2);

    VectorXd cp(2);
    cp << cx, cy;
    return cp;
}

// 得到候选框区域面积
double getBboxArea(PointCloud<PointXYZ> bBox){
    PointXYZ p1 = bBox[0];
    PointXYZ p2 = bBox[1];
    PointXYZ p3 = bBox[2];
    PointXYZ p4 = bBox[3];

    //S=tri(p1,p2,p3) + tri(p1, p3, p4)
    //s(triangle) = 1/2*|(x1−x3)(y2−y3)−(x2−x3)(y1−y3)|
    double tri1 = 0.5*abs((p1.x - p3.x)*(p2.y - p3.y) - (p2.x - p3.x)*(p1.y - p3.y));
    double tri2 = 0.5*abs((p1.x - p4.x)*(p3.y - p4.y) - (p3.x - p4.x)*(p1.y - p4.y)); 
    double S = tri1 + tri2;
    return S;
}
// 得到候选框区域面积
double getBboxArea_(PointCloud<PointXYZI> bBox){
    PointXYZI p1 = bBox[0];
    PointXYZI p2 = bBox[1];
    PointXYZI p3 = bBox[2];
    PointXYZI p4 = bBox[3];

    //S=tri(p1,p2,p3) + tri(p1, p3, p4)
    //s(triangle) = 1/2*|(x1−x3)(y2−y3)−(x2−x3)(y1−y3)|
    double tri1 = 0.5*abs((p1.x - p3.x)*(p2.y - p3.y) - (p2.x - p3.x)*(p1.y - p3.y));
    double tri2 = 0.5*abs((p1.x - p4.x)*(p3.y - p4.y) - (p3.x - p4.x)*(p1.y - p4.y)); 
    double S = tri1 + tri2;
    return S;
}

double getnearestdistance(PointCloud<PointXYZ> bBox , PointXYZ pos){
    double  minD = 999;
    for(int i = 0;i<bBox.size();i++)
    {
        double distance = sqrt((bBox[i].x-pos.x)*(bBox[i].x-pos.x) + (bBox[i].y-pos.y)*(bBox[i].y-pos.y));
        if(minD>distance)
        {
           minD = distance;
        }
    }
    return minD;
}

//保留实时框
void updateVisBoxArea(UKF& target, VectorXd dtCP){
    // cout << "calling area update"<<endl;   // 输出
    int lastInd = target.bb_yaw_history_.size()-1;
    // double diffYaw = target.bb_yaw_history_[lastInd] - target.bb_yaw_history_[lastInd-1];
    // cout << dtCP << endl;
    double area = getBboxArea(target.bestBBox_);
    for(int i = 0; i < target.BBox_.size(); i++){
        target.bestBBox_[i].x = target.bestBBox_[i].x + dtCP(0);
        target.bestBBox_[i].y = target.bestBBox_[i].y + dtCP(1);
    }

    // double postArea = getBboxArea(target.BBox_);
    // assert(abs(area - postArea)< 0.001);

}

// //依旧保留最好得到框，只是将其进行平移 ，给target.BBox_赋值
// void updateVisBoxArea(UKF& target, VectorXd dtCP){
//     // cout << "calling area update"<<endl;   // 输出
//     int lastInd = target.bb_yaw_history_.size()-1;
//     // double diffYaw = target.bb_yaw_history_[lastInd] - target.bb_yaw_history_[lastInd-1];
//     // cout << dtCP << endl;
//     double area = getBboxArea(target.bestBBox_);
//     for(int i = 0; i < target.BBox_.size(); i++){
//         target.BBox_[i].x = target.bestBBox_[i].x + dtCP(0);
//         target.BBox_[i].y = target.bestBBox_[i].y + dtCP(1);
//     }

//     double postArea = getBboxArea(target.BBox_);
//     assert(abs(area - postArea)< 0.001);

// }

void updateBoxYaw(UKF& target, VectorXd cp, double bestDiffYaw, bool isVis){
//    cout << "calling yaw update"<<endl;
    // cout << "before convert "<< target.BBox_[0].x << " "<<target.BBox_[0].y<<endl;
    for(int i = 0; i < target.BBox_.size(); i++){
        if(isVis){
            // rotate around cp
            double preX = target.BBox_[i].x;
            double preY = target.BBox_[i].y;
            target.BBox_[i].x = cos(bestDiffYaw)*(preX - cp(0)) - sin(bestDiffYaw)*(preY - cp(1)) + cp(0);
            target.BBox_[i].y = sin(bestDiffYaw)*(preX - cp(0)) + cos(bestDiffYaw)*(preY - cp(1)) + cp(1);
        }
        else{
            // rotate around cp
            double preX = target.bestBBox_[i].x;
            double preY = target.bestBBox_[i].y;
            target.bestBBox_[i].x = cos(bestDiffYaw)*(preX - cp(0)) - sin(bestDiffYaw)*(preY - cp(1)) + cp(0);
            target.bestBBox_[i].y = sin(bestDiffYaw)*(preX - cp(0)) + cos(bestDiffYaw)*(preY - cp(1)) + cp(1);   
        }
    }
    // cout << "after convert "<< target.BBox_[0].x << " "<<target.BBox_[0].y<<endl;
}

//计算target的BBox的yaw
double getBBoxYaw(UKF target){
    PointCloud<PointXYZ> bBox = target.BBox_;
    PointXYZ p1 = bBox[0];
    PointXYZ p2 = bBox[1];
    PointXYZ p3 = bBox[2];
    double dist1 = sqrt((p1.x- p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
    double dist2 = sqrt((p3.x- p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y));
    
    double yaw;
    // dist1 is length
    if(dist1>dist2){
        yaw = atan2(p1.y - p2.y, p1.x - p2.x);
    }
    else{
        yaw = atan2(p3.y - p2.y, p3.x - p2.x);   
    }

    double ukfYaw  = target.x_merge_(3);
    double diffYaw = abs(yaw - ukfYaw); 
    if(diffYaw < M_PI*0.5){
        return yaw;
    }
    else{
        yaw += M_PI;
        while (yaw> M_PI) yaw -= 2.*M_PI;
        while (yaw<-M_PI) yaw += 2.*M_PI;
        return yaw;
    }
}

//保留实时框
void updateBB(UKF& target){
    //to do: initialize target.BBox_ somewhere else
    // skip to prevent memory leak by accessing empty target.bbox_
    if(!target.isVisBB_){
        return;
    }
    // cout<<"start"<<endl;
    // 如果还没有最好的框，将第一个框赋给最好的框
    if(target.bestBBox_.empty()){
        target.bestBBox_ = target.BBox_;
        target.bestYaw_  = getBBoxYaw(target);
        return;
    }
    target.change_BBs_num++;
    if(target.change_BBs_num>10)
    {
        target.bestBBox_ = target.BBox_;
        target.bestYaw_  = getBBoxYaw(target);
        target.change_BBs_num = 0;
    }
    // cout<<"target.bestYaw_:"<<target.bestYaw_<<endl;
    // 计算两个框的质心位置的差值以及新的偏航角
    VectorXd cp         = getCpFromBbox(target.BBox_);
    VectorXd bestCP     = getCpFromBbox(target.bestBBox_);
    VectorXd dtCP       = cp - bestCP;
    double yaw = getBBoxYaw(target);
    // cout<<"target.Yaw_:"<<yaw<<endl;
    // bbox area
    // double area     = getBboxArea(target.BBox_);
    // double bestArea = getBboxArea(target.bestBBox_);

    // // start updating parameters
    // double deltaArea = area - bestArea;
    // cout << "area " <<area<<endl; 
    // cout << "bestbbox "<< bestArea<<endl;

    // 如果框较大，将这个框赋给最好的框
    // if( deltaArea < 0 ){ 
        // updateVisBoxArea(target, dtCP);
    // }
    // else if(deltaArea > 0){
    //     target.bestBBox_ = target.BBox_;
    //     // target.bestYaw_  = yaw;
    // }
    // cout <<"after transform " <<getBboxArea(target.BBox_)<<endl;


    // yaw = getBBoxYaw(target);
    double currentYaw  = getBBoxYaw(target);
    double DiffYaw = -(yaw - target.bestYaw_); 
    // double DiffYaw = currentYaw - yaw; 
    // double bestDiffYaw = target.bestYaw_ - yaw; 


    // cout << "box yaw "<< yaw<<endl;  // 输出
    // cout << "current yaw "<< currentYaw<<endl;
    // cout << "diff yaw "<< DiffYaw <<endl;



    // bestDiffYaw = -1 *bestDiffYaw;



    //when the diff yaw is out of range, keep the previous yaw
    // if(abs(DiffYaw) > bbYawChangeThres_){
    //     // updateVisBoxYaw(target, cp, bestDiffYaw);
    // }
    // //when the diff is acceptable, update best yaw
    // else if(abs(DiffYaw) < bbYawChangeThres_){
        bool isVis = true;
        updateBoxYaw(target, cp, DiffYaw, isVis);
        
        // isVis = false;
        // updateBoxYaw(target, cp, DiffYaw, isVis);

        // double afterYaw  = getBBoxYaw(target);
        // cout << "box yaw after "<< afterYaw<<endl;  // 输入  水平偏航角度yaw
        // assert(abs(yaw -getBBoxYaw(target)) < 0.01 );
    //     target.bestYaw_  = yaw;
    // // }
    // cout<<"end"<<endl;
}
//保留最大框
// void updateBB(UKF& target){
//     //to do: initialize target.BBox_ somewhere else
//     // skip to prevent memory leak by accessing empty target.bbox_
//     if(!target.isVisBB_){
//         return;
//     }
//     // skip the rest of process if the first bbox associaiton
//     if(target.bestBBox_.empty()){
//         target.bestBBox_ = target.BBox_;
//         target.bestYaw_  = getBBoxYaw(target);
//         return;
//     }

//     // calculate yaw
//     VectorXd cp         = getCpFromBbox(target.BBox_);
//     VectorXd bestCP     = getCpFromBbox(target.bestBBox_);
//     VectorXd dtCP       = cp - bestCP;
//     double yaw = getBBoxYaw(target);

//     // bbox area
//     double area     = getBboxArea(target.BBox_);
//     double bestArea = getBboxArea(target.bestBBox_);

//     // start updating parameters
//     double deltaArea = area - bestArea;
//     double ratioArea = area / bestArea;
//     // cout << "area " <<area<<endl; 
//     // cout << "bestbbox "<< bestArea<<endl;

//     // when the best area is bigger, keep best area
//     if( deltaArea < 0 || ratioArea>1.5){
//         updateVisBoxArea(target, dtCP);
//     }
//     else if(deltaArea > 0){
//         target.bestBBox_ = target.BBox_;
//         // target.bestYaw_  = yaw;
//     }
//     // yaw = getBBoxYaw(target);
//     double currentYaw  = getBBoxYaw(target);
//     double DiffYaw = yaw - currentYaw; 

//     //when the diff is acceptable, update best yaw
//     if(abs(DiffYaw) < bbYawChangeThres_){
//         bool isVis = true;
//         updateBoxYaw(target, cp, DiffYaw, isVis);
//         isVis = false;
//         updateBoxYaw(target, cp, DiffYaw, isVis);
//         target.bestYaw_  = yaw;
//     }

// }

// intersectM = ((p1.x - p2.x) * (p3.y - p1.y) + (p1.y - p2.y) * (p1.x - p3.x)) * _
//                  ((p1.x - p2.x) * (p4.y - p1.y) + (p1.y - p2.y) * (p1.x - p4.x))

double getIntersectCoef(double vec1x, double vec1y, double vec2x,double vec2y,
                        double px,double py, double cpx,double cpy){
    double intersectCoef = (((vec1x-vec2x)*(py - vec1y) + (vec1y - vec2y)*(vec1x - px)) *
        ((vec1x - vec2x)*(cpy - vec1y) + (vec1y - vec2y)*(vec1x - cpx)));
    return intersectCoef;

}

void mergeOverSegmentation(vector<UKF> targets){
    // cout << "mergeOverSegmentation"<<endl;
    for(int i = 0; i < targets.size(); i++){
        if(targets[i].isVisBB_ == true){
            double vec1x = targets[i].BBox_[0].x;
            double vec1y = targets[i].BBox_[0].y;
            double vec2x = targets[i].BBox_[1].x;
            double vec2y = targets[i].BBox_[1].y;
            double vec3x = targets[i].BBox_[2].x;
            double vec3y = targets[i].BBox_[2].y;
            double vec4x = targets[i].BBox_[3].x;
            double vec4y = targets[i].BBox_[3].y;
            double cp1x  = (vec1x+vec2x+vec3x)/3;
            double cp1y  = (vec1y+vec2y+vec3y)/3;
            double cp2x  = (vec1x+vec4x+vec3x)/3;
            double cp2y  = (vec1y+vec4y+vec3y)/3;
            for (int j = 0; j < targets.size(); j++){
                if(i == j) continue;
                double px = targets[j].x_merge_(0);
                double py = targets[j].x_merge_(1);
                double cross1 = getIntersectCoef(vec1x, vec1y, vec2x, vec2y, px, py, cp1x, cp1y);
                double cross2 = getIntersectCoef(vec1x, vec1y, vec3x, vec3y, px, py, cp1x, cp1y);
                double cross3 = getIntersectCoef(vec3x, vec3y, vec2x, vec2y, px, py, cp1x, cp1y);
                double cross4 = getIntersectCoef(vec1x, vec1y, vec4x, vec4y, px, py, cp2x, cp2y);
                double cross5 = getIntersectCoef(vec1x, vec1y, vec3x, vec3y, px, py, cp2x, cp2y);
                double cross6 = getIntersectCoef(vec3x, vec3y, vec4x, vec4y, px, py, cp2x, cp2y);
                if((cross1 > 0 && cross2>0&&cross3>0)||(cross4>0 && cross5 > 0 && cross6>0)){
                    // cout << "merrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrge"<<endl;
                    trackNumVec_[i] = 5;
                    trackNumVec_[j] = 0;
                }
            }
        }
    }
}

// void immUkfJpdaf(vector<PointCloud<PointXYZ>> bBoxes, double timestamp,
// 					 vector<PointXY>& targetPoints, vector<int>& trackManage, vector<bool>& isStaticVec){
void immUkfJpdaf(vector<PointCloud<PointXYZ>> bBoxes, double timestamp,double car_yaw,
                     PointCloud<PointXYZ>& targetPoints, vector<vector<double>>& targetVandYaw,
                     vector<int>& trackManage, vector<bool>& isStaticVec,
                     vector<bool>& isVisVec, vector<PointCloud<PointXYZ>>& visBBs ,PointXYZ &car_pos
                     ,vector<PointCloud<PointXYZI>> &his_nearvisBBs, vector<int>* trackIds){
    
    static bool change_flag = false;
    static int addNum = 0;

    egoYaw_ = car_yaw;
	// convert from bboxes to cx,cy
    // calculate cx, cy, http://imagingsolution.blog107.fc2.com/blog-entry-137.html
    vector<vector<double>> trackPoints;
    for(int i = 0; i < bBoxes.size(); i ++){
        PointXYZ p1 = bBoxes[i][0];
        PointXYZ p2 = bBoxes[i][1];
        PointXYZ p3 = bBoxes[i][2];
        PointXYZ p4 = bBoxes[i][3];
        double maxZ = bBoxes[i][4].z;
        VectorXd cp = getCpFromBbox(bBoxes[i]);

        // cout << "trackpoitns "<< cp(0) <<" "<<cp(1)<< endl;
        vector<double> point;
        point.push_back(cp(0));
        point.push_back(cp(1));
        point.push_back(p1.x);
        point.push_back(p1.y);
        point.push_back(p2.x);
        point.push_back(p2.y);
        point.push_back(p3.x);
        point.push_back(p3.y);
        point.push_back(p4.x);
        point.push_back(p4.y);
        point.push_back(maxZ);
        // cout<<"cp() = "<<cp(0)<<" "<<cp(1)<<endl;
        trackPoints.push_back(point);
    }  //求质心点

//    cout << trackPoints[0][0] << endl;
/***************************初始化******************************/
    if(!init_) {
        // cout << trackPoints.size()<< endl;
    	// for(int i = 0; i < trackPoints.size(); i++){
        //     // cout << trackPoints[i][0] <<" "<<trackPoints[i][1]<< endl;

        //     if(i  == 1 ){
        //         double px = trackPoints[i][0];
        //         double py = trackPoints[i][1];

        //         PointXYZ o;
        //         o.x = px;
        //         o.y = py;
        //         o.z = 0;
        //         targetPoints.push_back(o);

        //         vector<double> VandYaw;
        //         VandYaw.push_back(0);
        //         VandYaw.push_back(0);
        //         targetVandYaw.push_back(VandYaw);
        //         isStaticVec.push_back(false);
        //         isVisVec.push_back(false);

	    // 		VectorXd initMeas = VectorXd(2);
	    // 		initMeas << px, py;

        //         UKF ukf;
        //         ukf.Initialize(initMeas, timestamp);
        //         targets_.push_back(ukf);
        //         //initialize trackNumVec
        //         trackNumVec_.push_back(1);
    	// 	}
    	// }
        timestamp_ = timestamp;
        egoPreYaw_ = egoYaw_;
        trackManage = trackNumVec_;
        init_ = true;
        // cout << "target state: "<<endl<< targets_[0].x_merge_ << endl;
        return;
    }
    cout << targets_.size() << " " << trackNumVec_.size() <<endl;

    // assert (targets_.size() == trackNumVec_.size());

    // params initialization for ukf process
    vector<int> matchingVec(trackPoints.size()); // make 0 vector
    double dt = (timestamp - timestamp_);
    // cout<<"dt = "<<dt<<endl;
        // double dt = (timestamp - timestamp_)/1000000.0;

    timestamp_ = timestamp;

    // cout << "2222222"<<endl;
    // start UKF process


    for(int i = 0; i < targets_.size(); i++){
        //reset isVisBB_ to false
        targets_[i].isVisBB_ = false;
    // cout << "33333"<<endl;
        // making local2local vector
        double diffYaw = (egoYaw_ - egoPreYaw_);
    	//todo: modify here. This skips irregular measurement and nan
    	if(trackNumVec_[i] == 0) continue;
        // prevent ukf not to explode
        if(targets_[i].P_merge_.determinant() > 50 || targets_[i].P_merge_(4,4) > 1000){
            trackNumVec_[i] = 0;
            continue;
        }
        // cout << "target state start -----------------------------------"<<endl;
        // cout << "covariance"<<endl<<targets_[i].P_merge_<<endl;
        VectorXd maxDetZ;
        MatrixXd maxDetS;
    	vector<VectorXd> measVec;
        vector<VectorXd> bboxVec; //输入的聚类框
    	vector<double> lambdaVec;
    	// cout << "ProcessIMMUKF" << endl;
    	targets_[i].ProcessIMMUKF(dt);
    	// cout << "measurementValidation" << endl;
        // pre gating
        findMaxZandS(targets_[i], maxDetZ, maxDetS);
        double detS = maxDetS.determinant();

        if(isnan(detS) || detS > 500) {
            trackNumVec_[i] = 0;
            continue;
        }

        bool secondInit;
        if(trackNumVec_[i] == 1){
            secondInit = true;
        }
        else{
            secondInit = false;
        }

        // transform local to The target anchor TF
        // redundant calculation, modyfy here later
        vector<vector<double>> dcTrackPoints;
        dcTrackPoints = trackPoints;

        // measurement gating
        measurementValidation(dcTrackPoints, targets_[i], secondInit, maxDetZ, maxDetS,
         measVec, bboxVec, matchingVec);

        // bounding box association
        // input: track number, bbox measurements, &target 
        associateBB(trackNumVec_[i], bboxVec, targets_[i]);

        // bounding box validation
        updateBB(targets_[i]);

        // cout << "validated meas "<<measVec[0][0]<<" "<<measVec[0][1]<<endl; 

        // doing combined initialization
        if(secondInit){
            if(measVec.size() == 0){
                trackNumVec_[i] = 0;
                continue;
            }
            
            assert(measVec.size() == 1);
            // record init measurement for env classification
            targets_[i].initMeas_ << targets_[i].x_merge_(0), targets_[i].x_merge_(1);
            targets_[i].lastMeas_ << targets_[i].x_merge_(0), targets_[i].x_merge_(1);
            // abs update
            double targetX = measVec[0](0);
            double targetY = measVec[0](1);
            double targetDiffX = targetX - targets_[i].x_merge_(0);
            double targetDiffY = targetY - targets_[i].x_merge_(1);
            // cout << "target diff x and y "<< targetDiffX << " " << targetDiffY << endl;
            // cout << "target diff yaw "<<atan2(targetDiffY, targetDiffX)<<endl;
            double targetYaw = atan2(targetDiffY, targetDiffX);
            double dist      = sqrt(targetDiffX*targetDiffX + targetDiffY* targetDiffY);
            double targetV   = dist/dt;
            // double targetV   = 2;
            // cout << "second init "<<dist<<" "<<dt<<endl;
            
           
            while (targetYaw> M_PI) targetYaw -= 2.*M_PI;
            while (targetYaw<-M_PI) targetYaw += 2.*M_PI;
            // double targetV  = dist/dt;
            // targetV = 1.;
            targets_[i].x_merge_(0) = targets_[i].x_cv_(0) = targets_[i].x_ctrv_(0) = targets_[i].x_rm_(0) = targetX;
            targets_[i].x_merge_(1) = targets_[i].x_cv_(1) = targets_[i].x_ctrv_(1) = targets_[i].x_rm_(1) = targetY;
            targets_[i].x_merge_(2) = targets_[i].x_cv_(2) = targets_[i].x_ctrv_(2) = targets_[i].x_rm_(2) = targetV;
            targets_[i].x_merge_(3) = targets_[i].x_cv_(3) = targets_[i].x_ctrv_(3) = targets_[i].x_rm_(3) = targetYaw;
            targets_[i].x_ca_(0) = targetX; targets_[i].x_ca_(1) = targetY;
            targets_[i].x_ca_(2) = targetV; targets_[i].x_ca_(3) = targetYaw;
            targets_[i].x_ca_(5) = 0.0;
            targets_[i].yaw_history_ = targets_[i].x_merge_(3);
            
            // cout << "dx, dy " << dX << " " << dY<<endl; 
            // cout << "init valo: " << targetV << endl;
            // cout << "init yaw: "  << targetYaw << endl;
            trackNumVec_[i]++;
            continue;
        }

    	// update tracking number 
    	if(measVec.size() > 0) {
    		if(trackNumVec_[i] < 3){
    			trackNumVec_[i]++;
    		}
    		else if(trackNumVec_[i] == 3){
    			trackNumVec_[i] = 5;
    		}
    		else if(trackNumVec_[i] >= 5){
    			trackNumVec_[i] = 5;
    		}
    	}else{
            if(trackNumVec_[i] < 5){
                trackNumVec_[i]--;
                if(trackNumVec_[i] <= 0) trackNumVec_[i] = 0;
            }
    		else if(trackNumVec_[i] >= 5 && trackNumVec_[i] < 10){
    			trackNumVec_[i]++;
    		}
    		else if(trackNumVec_[i] == 10){
    			trackNumVec_[i] = 0;
    		}
    	}

        if(trackNumVec_[i] == 0) continue;


	    filterPDA(targets_[i], measVec, lambdaVec);
	    // cout << "PostIMMUKF" << endl;
	    targets_[i].PostProcessIMMUKF(lambdaVec);
        // TODO: might be wrong
        //将前两帧速度置零
        if(targets_[i].count_<=2)
        {
            targets_[i].count_++;
            targets_[i].x_merge_(2) = 0;
        }
        
        
        double yaw_ = targets_[i].x_merge_(3);
        double yaw_pre_ = targets_[i].yaw_history_;
        
        if((abs(yaw_+yaw_pre_)<M_PI) && abs(yaw_-yaw_pre_)>M_PI)
        {
            cout<<"发生2pi突变"<<endl;
            if(abs(yaw_)>=abs(yaw_pre_)) 
            {
                if(yaw_>=0)
                {
                    yaw_ -= 2*M_PI;
                    double targetYaw_ = filter(yaw_,yaw_pre_,alpha);
                    targets_[i].x_merge_(3) = targets_[i].yaw_history_ = targetYaw_;
                }else{
                    yaw_ += 2*M_PI;
                    double targetYaw_ = filter(yaw_,yaw_pre_,alpha);
                    targets_[i].x_merge_(3) = targets_[i].yaw_history_ = targetYaw_;
                }
            }else{
                if(yaw_pre_>=0)
                {
                    yaw_pre_ -= 2*M_PI;
                    double targetYaw_ = filter(yaw_,yaw_pre_,alpha);
                    targets_[i].x_merge_(3) = targets_[i].yaw_history_ = targetYaw_;
                }else{
                    yaw_pre_ += 2*M_PI;
                    double targetYaw_ = filter(yaw_,yaw_pre_,alpha);
                    targets_[i].x_merge_(3) = targets_[i].yaw_history_ = targetYaw_;
                }
            }
        }else{
                double targetYaw_ = filter(targets_[i].x_merge_(3),targets_[i].yaw_history_,alpha);
                targets_[i].x_merge_(3) = targets_[i].yaw_history_ = targetYaw_;
        }
        double targetVelo = targets_[i].x_merge_(2); 
        targets_[i].velo_history_.push_back(targetVelo);

        if(targets_[i].velo_history_.size()>1)
        {
            double velChange = abs(targets_[i].velo_history_[targets_[i].velo_history_.size()-1]-targets_[i].velo_history_[targets_[i].velo_history_.size()-2]);
            double velChangeThres = (targets_[i].modeProbCA_ > 0.3) ? 8.0 : 3.0;
            if(velChange > velChangeThres)
            {
                targets_[i].x_merge_(2) = targets_[i].velo_history_[targets_[i].velo_history_.size()-2];
            }
        }
        // if(targets_[i].velo_history_.size() == 5) {
        //     targets_[i].velo_history_.erase (targets_[i].velo_history_.begin());
        //     double var = variance(targets_[i].velo_history_);
        //     if(var<2.25)
        //     {
        //          targets_[i].x_merge_(2) = 0;
        //     }
           
        // }
        // targets_[i].x_cv_(0) = targets_[i].x_ctrv_(0) = targets_[i].x_rm_(0) = targets_[i].x_merge_(0) = measVec[0](0);
        // targets_[i].x_cv_(1) = targets_[i].x_ctrv_(1) = targets_[i].x_rm_(1) = targets_[i].x_merge_(1) = measVec[0](1);
    }
    // end UKF process
    

    // cout << trackPoints[0][0] << endl;
    // cout << targets_[0].x_merge_(0) << endl;

    // deling with over segmentation, update trackNumVec_ 处理过度分割问题，并更新trackNumVec_
    mergeOverSegmentation(targets_);


    // making new ukf target
    int addedNum = 0;
    int targetNum = targets_.size();
    for(int i = 0; i < matchingVec.size(); i ++){
        if(matchingVec[i] == 0){
            double px = trackPoints[i][0];
            double py = trackPoints[i][1];

            VectorXd initMeas = VectorXd(2);
            initMeas << px, py;

            UKF ukf;
            ukf.Initialize(initMeas, timestamp);
            ukf.track_id_ = nextTrackId_++;
            targets_.push_back(ukf);
            trackNumVec_.push_back(1);
            addedNum ++;
        }
    }

    // cout <<"11matchingVec.size() = "<<matchingVec.size()<<endl;
    // cout <<"11trackNumVec_.size() = "<<trackNumVec_.size()<<endl;
    // assert(targets_.size() == (addedNum + targetNum));
    
    // making poitns for visualization
    // cout << "making points for vis" <<endl;
    int targetNumCount = 0;
    for(int i = 0; i < targets_.size(); i++){
        double tx   = targets_[i].x_merge_(0);
        double ty   = targets_[i].x_merge_(1);
        double mx = targets_[i].lastMeas_(0);
        double my = targets_[i].lastMeas_(1);
        targets_[i].lastMeas_ << targets_[i].x_merge_(0), targets_[i].x_merge_(1);
        // double mx = targets_[i].initMeas_(0);
        // double my = targets_[i].initMeas_(1);
        // cout <<"fisrt meas "<<mx << " "<<my<<endl;
        // cout <<"estimate "<<tx << " "<<ty<<endl;
        // float distance = sqrt((tx-car_pos.x)*(tx-car_pos.x) + (ty-car_pos.y)*(ty-car_pos.y));
        targets_[i].distFromInit_ = sqrt((tx - mx)*(tx - mx) + (ty - my)*(ty - my));
        // cout <<"targets_[i].distFromInit_: "<<targets_[i].distFromInit_<<" "<<sqrt((tx - mx)*(tx - mx) + (ty - my)*(ty - my))<<" "<<dt<<endl;
        vector<double> cp;
        cp.push_back(tx);
        cp.push_back(ty);


        double tv = targets_[i].x_merge_(2);
        double tyaw = targets_[i].x_merge_(3);

        // tyaw += egoPoints_[0][2];
        // tyaw -= egoYaw_;
        while (tyaw> M_PI) tyaw -= 2.*M_PI;
        while (tyaw<-M_PI) tyaw += 2.*M_PI;
        // cout << "testing yaw off "<< tyaw << endl;

        PointXYZ o;
        o.x = cp[0];
        o.y = cp[1];
        o.z = 0;
        
        targetPoints.push_back(o);

        vector<double> VandYaw;
        // VandYaw fields:
        // [0] speed v, [1] yaw, [2] modeProbCA, [3] accel(from CA state),
        // [4] dominant mode index (CV/CTRV/RM/CA), [5] yaw_rate
        VandYaw.push_back(tv);
        VandYaw.push_back(tyaw);
        VandYaw.push_back(targets_[i].modeProbCA_);
        VandYaw.push_back(targets_[i].x_ca_(5));
        // Dominant mode: 0=CV, 1=CTRV, 2=RM, 3=CA
        double pmax = targets_[i].modeProbCV_;
        int modeIdx = 0;
        if (targets_[i].modeProbCTRV_ > pmax) { pmax = targets_[i].modeProbCTRV_; modeIdx = 1; }
        if (targets_[i].modeProbRM_   > pmax) { pmax = targets_[i].modeProbRM_;   modeIdx = 2; }
        if (targets_[i].modeProbCA_   > pmax) { pmax = targets_[i].modeProbCA_;   modeIdx = 3; }
        // 角速度辅助：|yaw_rate|超过阈值时视为旋转，强制CTRV
        double yawRate = targets_[i].x_merge_(4);  // 合并状态的角速度
        if (fabs(yawRate) > YAW_RATE_THRESH_ && tv > 0.3) {
            modeIdx = 1;  // 强制CTRV
        }
        VandYaw.push_back(static_cast<double>(modeIdx));
        VandYaw.push_back(yawRate);
        targetVandYaw.push_back(VandYaw);

        isStaticVec.push_back(false);
        // isVisVec.push_back(false);
        isVisVec.push_back(targets_[i].isVisBB_);
        if(targets_[i].isVisBB_){
            visBBs.push_back(targets_[i].BBox_);
            if(targets_[i].isStatic_&&targets_[i].staticCount>=5)
            {
                float S = getBboxArea(targets_[i].BBox_);
                double distance = getnearestdistance(targets_[i].BBox_,car_pos);
                if(distance<1.5&&S<0.5)
                {
                    // cout<<"distance = "<<i<<" "<<distance<<endl;
                    if(!change_flag)
                    {
                        if(targets_[i].isRecord)
                        {
                            for (size_t count = 0; count < his_nearvisBBs.size(); count++)
                            {
                                if(his_nearvisBBs[count][0].z == i)
                                {
                                    for(size_t j = 0; j < 4; j++)
                                    {
                                        his_nearvisBBs[count][j].x = targets_[i].BBox_.points[j].x;
                                        his_nearvisBBs[count][j].y = targets_[i].BBox_.points[j].y;
                                        his_nearvisBBs[count][j].intensity = 0;
                                    }
                        
                                    cout<<"代替之前障碍物！！！"<<i<<endl;
                                }
                            }

                        }else{
                            pcl::PointCloud<pcl::PointXYZI> temp;
                            for (size_t j = 0; j < 4; j++)
                            {
                                pcl::PointXYZI point;
                                point.x = targets_[i].BBox_.points[j].x;
                                point.y = targets_[i].BBox_.points[j].y;
                                point.z = i;
                                point.intensity = 0;
                                temp.points.push_back(point);
                            }
                            his_nearvisBBs.push_back(temp);
                            cout<<"较近的障碍物出现！！！"<<i<<endl;
                            targets_[i].isRecord = true;
                        }
                    }else{
                        if(targets_[i].isRecord)
                        {
                            for (size_t count = 0; count < his_nearvisBBs.size(); count++)
                            {
                                if(his_nearvisBBs[count][0].z == i+addNum)
                                {
                                    for(size_t j = 0; j < 4; j++)
                                    {
                                        his_nearvisBBs[count][j].x = targets_[i].BBox_.points[j].x;
                                        his_nearvisBBs[count][j].y = targets_[i].BBox_.points[j].y;
                                        his_nearvisBBs[count][j].intensity = 0;
                                    }
                        
                                    cout<<"代替之前障碍物！！！"<<i<<endl;
                                }
                            }

                        }else{
                            pcl::PointCloud<pcl::PointXYZI> temp;
                            for (size_t j = 0; j < 4; j++)
                            {
                                pcl::PointXYZI point;
                                point.x = targets_[i].BBox_.points[j].x;
                                point.y = targets_[i].BBox_.points[j].y;
                                point.z = i;
                                point.intensity = 0;
                                temp.points.push_back(point);
                            }
                            his_nearvisBBs.push_back(temp);
                            cout<<"较近的障碍物出现！！！"<<i<<endl;
                            targets_[i].isRecord = true;
                        }
                    }
                    
                }
            }
            
           
        }else{
            pcl::PointCloud<pcl::PointXYZ>  temp;
            temp.push_back(pcl::PointXYZ(0,0,0));
            visBBs.push_back(temp);
        }
        
        if(trackNumVec_[i] != 0){
            targetNumCount ++;
        }
    }

    // static dynamic classification
    // cout <<"classification"<<endl;
    for (int i = 0; i < trackNumVec_.size(); i++){
        // once target is static, it is dtatic until lost
        // if(targets_[i].isStatic_ ){
        //     isStaticVec[i] = true;
        //     continue;
        // } 
       
        if(trackNumVec_[i] == 5 && targets_[i].lifetime_ > 0 ){
            // assuming below 0.3 m/s for static onject
            double distThres = 0.015;
            // 当前频率是20Hz,dt=0.05s,distThres=0.015m

            // cout <<"distThres = "<<distThres<< endl;
            // cout << "print "<<targets_[i].x_merge_(0)<< " "<< targets_[i].x_merge_(1)<< " "<< targets_[i].distFromInit_ <<endl;
            // cout << "mode prob "<< " "<<targets_[i].modeProbCV_ << " "<<targets_[i].modeProbCTRV_ << " "<< targets_[i].modeProbRM_ << endl;
            if(((targets_[i].distFromInit_ < distThres)&&(targets_[i].modeProbRM_ > targets_[i].modeProbCV_ || 
                targets_[i].modeProbRM_ > targets_[i].modeProbCTRV_ )))
            {
                isStaticVec[i] = true;
                targets_[i].isStatic_ = true;
                targets_[i].staticCount++;
                // trackNumVec_[i] = -1;
            }
            else{
                isStaticVec[i] = false;
                targets_[i].isStatic_ = false;
                targets_[i].staticCount = 0;
            }
        }
    }

    trackManage = trackNumVec_;
    egoPreYaw_ = egoYaw_;

    if (trackIds) {
        trackIds->clear();
        for (size_t i = 0; i < targets_.size(); i++)
            trackIds->push_back(targets_[i].track_id_);
    }

    cout<<"addedNum = "<<addedNum<<endl;

    // Remove stale targets (trackNumVec_ == 0) to prevent unbounded growth
    if(targets_.size()>30)
    {
        change_flag = true;
        vector<UKF> kept_targets;
        vector<int> kept_trackNums;
        for(size_t i = 0; i < targets_.size(); i++){
            if(trackNumVec_[i] != 0){
                kept_targets.push_back(targets_[i]);
                kept_trackNums.push_back(trackNumVec_[i]);
            }
        }
        addNum = targets_.size() - kept_targets.size();
        targets_ = kept_targets;
        trackNumVec_ = kept_trackNums;
    }
}
