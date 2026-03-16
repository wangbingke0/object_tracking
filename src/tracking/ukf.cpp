//
// Created by kosuke on 12/23/17.
//

#include "Ukf.h"
// #include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using namespace Eigen;
//using Eigen::MatrixXd;
//using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_merge_ = MatrixXd(5, 1);

    // initial state vector
    x_cv_ = MatrixXd(5, 1);

    // initial state vector
    x_ctrv_ = MatrixXd(5, 1);

    // initial state vector
    x_rm_ = MatrixXd(5, 1);

    // CA state vector (6D)
    x_ca_ = MatrixXd(6, 1);

    // initial covariance matrix
    P_merge_ = MatrixXd(5, 5);

    // initial covariance matrix
    P_cv_ = MatrixXd(5, 5);

    // initial covariance matrix
    P_ctrv_ = MatrixXd(5, 5);

    // initial covariance matrix
    P_rm_ = MatrixXd(5, 5);

    // CA covariance matrix (6x6)
    P_ca_ = MatrixXd(6, 6);

    // Process noise standard deviation longitudinal acceleration in m/s^2

    // std_a_cv_   = 0.8;
    // std_a_ctrv_ = 0.8;
    // std_a_rm_   = 5;
    // std_ctrv_yawdd_ = 0.8;
    // std_cv_yawdd_   = 0.8;
    // std_rm_yawdd_ = 3;


    // std_a_cv_   = 2;
    // std_a_ctrv_ = 2;
    // std_a_rm_   = 5;
    // std_ctrv_yawdd_ = 2;
    // std_cv_yawdd_   = 2;
    // std_rm_yawdd_ = 3;

    std_a_cv_   = 2;
    std_a_ctrv_ = 2;
    std_a_rm_   = 3;
    std_ctrv_yawdd_ = 2;
    std_cv_yawdd_   = 2;
    std_rm_yawdd_ = 3;

    std_jerk_ca_  = 2;
    std_ca_yawdd_ = 2;

    // std_a_cv_   = 3;
    // std_a_ctrv_ = 3;
    // std_a_rm_   = 3;
    // std_ctrv_yawdd_ = 3;
    // std_cv_yawdd_   = 3;
    // std_rm_yawdd_ = 3;

    // ------------- not delete here
    // std_a_cv_   = 2;
    // std_a_ctrv_ = 2;
    // std_a_rm_   = 8;
    // std_ctrv_yawdd_ = 2;
    // std_cv_yawdd_   = 2;
    // std_rm_yawdd_ = 3;
    //------------------
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    // std_laspx_ = 0.3;
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    // std_laspy_ = 0.3;

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // time when the state is true, in us
    time_us_ = 0.0;

    // state dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // CA dimensions
    n_x_ca_   = 6;
    n_aug_ca_ = 8;   // 6 state + 2 noise (jerk, yaw_accel)
    lambda_aug_ca_ = 3 - n_aug_ca_;

    // Sigma point spreading parameter
    lambda_ = 3 - n_x_;

    // Augmented sigma point spreading parameter
    lambda_aug_ = 3 - n_aug_;

    // predicted sigma points matrix
//    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // predicted sigma points matrix
    Xsig_pred_cv_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // predicted sigma points matrix
    Xsig_pred_ctrv_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // predicted sigma points matrix
    Xsig_pred_rm_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // CA predicted sigma points matrix (6 x 17)
    Xsig_pred_ca_ = MatrixXd(n_x_ca_, 2 * n_aug_ca_ + 1);

    //create vector for weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    // CA weights
    weights_ca_ = VectorXd(2 * n_aug_ca_ + 1);
    double w0_ca = lambda_aug_ca_ / (lambda_aug_ca_ + n_aug_ca_);
    weights_ca_(0) = w0_ca;
    for (int i = 1; i < 2 * n_aug_ca_ + 1; i++)
        weights_ca_(i) = 0.5 / (n_aug_ca_ + lambda_aug_ca_);

    // the current NIS for radar
    NIS_radar_ = 0.0;

    // the current NIS for laser
    NIS_laser_ = 0.0;

    count_ = 0;
    count_empty_ = 0;

    ini_u_.push_back(0.25);
    ini_u_.push_back(0.25);
    ini_u_.push_back(0.25);
    ini_u_.push_back(0.25);

    // 4x4 mode transition probability matrix
    p1_.push_back(0.85);  // CV->CV
    p1_.push_back(0.05);  // CV->CTRV
    p1_.push_back(0.05);  // CV->RM
    p1_.push_back(0.05);  // CV->CA

    p2_.push_back(0.15);  // CTRV->CV
    p2_.push_back(0.45);  // CTRV->CTRV
    p2_.push_back(0.15);  // CTRV->RM
    p2_.push_back(0.15);  // CTRV->CA

    p3_.push_back(0.05);  // RM->CV
    p3_.push_back(0.05);  // RM->CTRV
    p3_.push_back(0.85);  // RM->RM
    p3_.push_back(0.05);  // RM->CA

    p4_.push_back(0.05);  // CA->CV
    p4_.push_back(0.05);  // CA->CTRV
    p4_.push_back(0.05);  // CA->RM
    p4_.push_back(0.85);  // CA->CA

    modeMatchProbCV2CV_ = 0;
    modeMatchProbCTRV2CV_ = 0;
    modeMatchProbRM2CV_ = 0;
    modeMatchProbCA2CV_ = 0;

    modeMatchProbCV2CTRV_ = 0;
    modeMatchProbCTRV2CTRV_ = 0;
    modeMatchProbRM2CTRV_ = 0;
    modeMatchProbCA2CTRV_ = 0;

    modeMatchProbCV2RM_ = 0;
    modeMatchProbCTRV2RM_ = 0;
    modeMatchProbRM2RM_ = 0;
    modeMatchProbCA2RM_ = 0;

    modeMatchProbCV2CA_ = 0;
    modeMatchProbCTRV2CA_ = 0;
    modeMatchProbRM2CA_ = 0;
    modeMatchProbCA2CA_ = 0;

    modeProbCV_   = 0.25;
    modeProbCTRV_ = 0.25;
    modeProbRM_   = 0.25;
    modeProbCA_   = 0.25;

    zPredCVl_ = VectorXd(2);
    zPredCTRVl_ =  VectorXd(2);
    zPredRMl_ =  VectorXd(2);
    zPredCAl_ = VectorXd(2);

    zPredCVr_ = VectorXd(3);
    zPredCTRVr_ =  VectorXd(3);
    zPredRMr_ =  VectorXd(3);

//    lS_ = MatrixXd(2,2);
//    rS_ = MatrixXd(3,3);
    lS_cv_   = MatrixXd(2,2);
    lS_ctrv_ = MatrixXd(2,2);
    lS_rm_   = MatrixXd(2,2);
    lS_ca_   = MatrixXd(2,2);

    rS_cv_   = MatrixXd(3,3);
    rS_ctrv_   = MatrixXd(3,3);
    rS_rm_   = MatrixXd(3,3);

    K_cv_   = MatrixXd(2,2);
    K_ctrv_ = MatrixXd(2,2);
    K_rm_   = MatrixXd(2,2);
    K_ca_   = MatrixXd(6,2);

//    NISvals_laser_cv_.open( "../NISvals_laser_cv.txt", ios::out );
//    NISvals_laser_ctrv_.open( "../NISvals_laser_ctrv.txt", ios::out );
//    NISvals_laser_rm_.open( "../NISvals_laser_rm.txt", ios::out );
//    // Check for errors opening the files
//    if( !NISvals_laser_cv_.is_open() )
//    {
//        cout << "Error opening NISvals_laser.txt" << endl;
//        exit(1);
//    }

    gammaG_ = 9.21;
    pD_     = 0.9;
    pG_     = 0.99;

    //track parameter
    lifetime_ = 0;
    velo_history_;
    isStatic_ = false;
    staticCount = 0;
    //bounding box params
    isVisBB_ = false;
    bestYaw_ = 0;
    bb_yaw_  = 0;
    bb_area_ = 0;

    //for env classification
    initMeas_ = VectorXd(2);
    lastMeas_ = VectorXd(2);
    distFromInit_ = 0;

    // local2local yaw (t-1 to t)
    // local2localYaw_ = 0;

    x_merge_yaw_ = 0;

    isRecord = false;

    change_BBs_num = 0;
    track_id_ = -1;

    // globalYaw_ = 0;
    // anchorTF_  = VectorXd(2);

    //    double gammaG = 4.61; // 90%
//    double gammaG = 5.99; // 95%
//    double gammaG = 7.38; // 97.5%
//    gammaG_ = 9.21; // 99% global variable
}

UKF::~UKF() {
//    NISvals_laser_cv_.close();
//    NISvals_laser_ctrv_.close();
//    NISvals_laser_rm_.close();
}

void UKF::Initialize(VectorXd z, double timestamp) {
    // first measurement
    // x_merge_ << 1, 1, 1, 1, 0.1;
    x_merge_ << 1, 1, 0, 0, 0.1;

    // init covariance matrix
    // P_merge_ <<  0.5,    0, 0, 0, 0,
    //         0,  0.5, 0, 0, 0,
    //         0,    0, 1, 0, 0,
    //         0,    0, 0, 1, 0,
    //         0,    0, 0, 0, 1;

    // P_merge_ <<   0.5,    0, 0, 0, 0,
    //                 0,  0.5, 0, 0, 0,
    //                 0,    0, 3, 0, 0,
    //                 0,    0, 0, 1, 0,
    //                 0,    0, 0, 0, 1;

    P_merge_ <<   0.5,    0, 0, 0, 0,
                    0,  0.5, 0, 0, 0,
                    0,    0, 3, 0, 0,
                    0,    0, 0,10, 0,
                    0,    0, 0, 0, 1;

    // set weights
    double weight_0 = lambda_aug_ / (lambda_aug_ + n_aug_);
    weights_(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_aug_);
        weights_(i) = weight;
    }

    // init timestamp
    time_us_ = timestamp;

    x_merge_(0) = z(0);
    x_merge_(1) = z(1);

    zPredCVl_(0) = z(0);
    zPredCVl_(1) = z(1);

    zPredCTRVl_(0) = z(0);
    zPredCTRVl_(1) = z(1);

    zPredRMl_(0) = z(0);
    zPredRMl_(1) = z(1);

    zPredCAl_(0) = z(0);
    zPredCAl_(1) = z(1);

    x_cv_ = x_ctrv_ = x_rm_ = x_merge_;
    P_cv_ = P_ctrv_ = P_rm_ = P_merge_;

    // Initialize CA state (6D)
    x_ca_ << x_merge_(0), x_merge_(1), x_merge_(2), x_merge_(3), x_merge_(4), 0.0;
    P_ca_ = MatrixXd::Zero(6, 6);
    P_ca_.topLeftCorner(5, 5) = P_merge_;
    P_ca_(5, 5) = 4.0;  // initial acceleration variance

    lS_cv_   <<  1, 0,
                 0, 1;
    lS_ctrv_ <<  1, 0,
                 0, 1;
    lS_rm_   <<  1, 0,
                 0, 1;
    lS_ca_   <<  1, 0,
                 0, 1;

    // anchorTF_ << 0, 0;
}

double UKF::CalculateGauss(VectorXd z, int sensorInd, int modelInd){
    if(sensorInd == 0){
        if      (modelInd == 0) {
            double  detS = fabs(lS_cv_.determinant());
            MatrixXd inS = lS_cv_.inverse();
//            cout << z << endl << zPredCVl_ << endl;
//            VectorXd s = (z-zPredCVl_).transpose();
//            double a = ((z-zPredCVl_).transpose()*inS*(z-zPredCVl_));
            return exp(-1*(((z-zPredCVl_).transpose()*inS*(z-zPredCVl_))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS));
        }
        else if (modelInd == 1) {
            double  detS = fabs(lS_ctrv_.determinant());
            MatrixXd inS = lS_ctrv_.inverse();
            return exp(-1*(((z-zPredCTRVl_).transpose()*inS*(z-zPredCTRVl_))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS));
        }
        else if (modelInd == 2) {
            double  detS = fabs(lS_rm_.determinant());
            MatrixXd inS = lS_rm_.inverse();
            return exp(-1*(((z-zPredRMl_).transpose()  *inS*(z-zPredRMl_))(0))/2)  /sqrt(((2*M_PI)*(2*M_PI)*detS));
        }
        else {
            double  detS = fabs(lS_ca_.determinant());
            MatrixXd inS = lS_ca_.inverse();
            return exp(-1*(((z-zPredCAl_).transpose()  *inS*(z-zPredCAl_))(0))/2)  /sqrt(((2*M_PI)*(2*M_PI)*detS));
        }
    }
    else if(sensorInd == 1){
        if (modelInd == 0){
            double  detS = fabs(rS_cv_.determinant());
            MatrixXd inS = rS_cv_.inverse();
            double cvProb = exp(-1*(((z-zPredCVr_).transpose()  *inS*(z-zPredCVr_))(0))/2)  /sqrt((2*M_PI)*(2*M_PI)*(2*M_PI)*detS);
            if(cvProb != 0) return cvProb;
            else {
                z[1] = -1 * z[1];
                cvProb = exp(-1*(((z-zPredCVr_).transpose()  *inS*(z-zPredCVr_))(0))/2)  /sqrt((2*M_PI)*(2*M_PI)*(2*M_PI)*detS);
                return cvProb;
            }
        }
        else if (modelInd == 1) {
            double detS = fabs(rS_ctrv_.determinant());
            MatrixXd inS = rS_ctrv_.inverse();
            double ctrvProb = exp(-1 * (((z - zPredCTRVr_).transpose() * inS * (z - zPredCTRVr_))(0)) / 2) /
                              sqrt((2 * M_PI) * (2 * M_PI) * (2 * M_PI) * detS);
            if (ctrvProb != 0) return ctrvProb;
            else {
                z[1] = -1 * z[1];
                ctrvProb = exp(-1 * (((z - zPredCTRVr_).transpose() * inS * (z - zPredCTRVr_))(0)) / 2) /
                           sqrt((2 * M_PI) * (2 * M_PI) * (2 * M_PI) * detS);
                return ctrvProb;
            }
        }
        else    {
            double  detS = fabs(rS_rm_.determinant());
            MatrixXd inS = rS_rm_.inverse();
            double rmProb = exp(-1*(((z-zPredRMr_).transpose()  *inS*(z-zPredRMr_))(0))/2)  /sqrt((2*M_PI)*(2*M_PI)*(2*M_PI)*detS);
            if(rmProb != 0)return rmProb;
            else{
                z[1] = -1 * z[1];
                rmProb = exp(-1*(((z-zPredRMr_).transpose()  *inS*(z-zPredRMr_))(0))/2)  /sqrt((2*M_PI)*(2*M_PI)*(2*M_PI)*detS);
                return rmProb;
            }
        }
    }
}

void UKF::UpdateModeProb(vector<double> lambdaVec){
    double cvGauss   = lambdaVec[0];
    double ctrvGauss = lambdaVec[1];
    double rmGauss   = lambdaVec[2];
    double caGauss   = lambdaVec[3];
    double sumGauss  = cvGauss*modeProbCV_ + ctrvGauss*modeProbCTRV_ + rmGauss*modeProbRM_ + caGauss*modeProbCA_;
    modeProbCV_   = (cvGauss  *modeProbCV_)  /sumGauss;
    modeProbCTRV_ = (ctrvGauss*modeProbCTRV_)/sumGauss;
    modeProbRM_   = (rmGauss  *modeProbRM_)  /sumGauss;
    modeProbCA_   = (caGauss  *modeProbCA_)  /sumGauss;
    if(fabs(modeProbCV_)   < 0.0001) modeProbCV_   = 0.0001;
    if(fabs(modeProbCTRV_) < 0.0001) modeProbCTRV_ = 0.0001;
    if(fabs(modeProbRM_)   < 0.0001) modeProbRM_   = 0.0001;
    if(fabs(modeProbCA_)   < 0.0001) modeProbCA_   = 0.0001;
}

void UKF::UpdateYawWithHighProb(){
    double maxProb = modeProbCV_;
    x_merge_yaw_ = x_cv_(3);

    if (modeProbCTRV_ > maxProb) { maxProb = modeProbCTRV_; x_merge_yaw_ = x_ctrv_(3); }
    if (modeProbRM_   > maxProb) { maxProb = modeProbRM_;   x_merge_yaw_ = x_rm_(3);   }
    if (modeProbCA_   > maxProb) { maxProb = modeProbCA_;   x_merge_yaw_ = x_ca_(3);   }

    x_merge_(3) = x_merge_yaw_;
}

void UKF::MergeEstimationAndCovariance(){
    // project CA 6D -> 5D for merge
    MatrixXd x_ca_5 = x_ca_.block(0, 0, 5, 1);

    x_merge_ = modeProbCV_*x_cv_ + modeProbCTRV_*x_ctrv_ + modeProbRM_*x_rm_ + modeProbCA_*x_ca_5;
    while (x_merge_(3)> M_PI) x_merge_(3) -= 2.*M_PI;
    while (x_merge_(3)<-M_PI) x_merge_(3) += 2.*M_PI;

    UpdateYawWithHighProb();

    MatrixXd P_ca_5 = P_ca_.topLeftCorner(5, 5);

    P_merge_ = modeProbCV_  *(P_cv_   +(x_cv_    - x_merge_)*(x_cv_    - x_merge_).transpose()) +
               modeProbCTRV_*(P_ctrv_ +(x_ctrv_  - x_merge_)*(x_ctrv_  - x_merge_).transpose())+
               modeProbRM_  *(P_rm_   +(x_rm_    - x_merge_)*(x_rm_    - x_merge_).transpose())+
               modeProbCA_  *(P_ca_5  +(x_ca_5   - x_merge_)*(x_ca_5   - x_merge_).transpose());

}

void UKF::MixingProbability() {
    // c_j = sum_i(pi_ij * mu_i) for each target model j
    double sumProb1 = modeProbCV_*p1_[0]+modeProbCTRV_*p2_[0]+modeProbRM_*p3_[0]+modeProbCA_*p4_[0];
    double sumProb2 = modeProbCV_*p1_[1]+modeProbCTRV_*p2_[1]+modeProbRM_*p3_[1]+modeProbCA_*p4_[1];
    double sumProb3 = modeProbCV_*p1_[2]+modeProbCTRV_*p2_[2]+modeProbRM_*p3_[2]+modeProbCA_*p4_[2];
    double sumProb4 = modeProbCV_*p1_[3]+modeProbCTRV_*p2_[3]+modeProbRM_*p3_[3]+modeProbCA_*p4_[3];

    modeMatchProbCV2CV_     = modeProbCV_  *p1_[0]/sumProb1;
    modeMatchProbCTRV2CV_   = modeProbCTRV_*p2_[0]/sumProb1;
    modeMatchProbRM2CV_     = modeProbRM_  *p3_[0]/sumProb1;
    modeMatchProbCA2CV_     = modeProbCA_  *p4_[0]/sumProb1;

    modeMatchProbCV2CTRV_   = modeProbCV_  *p1_[1]/sumProb2;
    modeMatchProbCTRV2CTRV_ = modeProbCTRV_*p2_[1]/sumProb2;
    modeMatchProbRM2CTRV_   = modeProbRM_  *p3_[1]/sumProb2;
    modeMatchProbCA2CTRV_   = modeProbCA_  *p4_[1]/sumProb2;

    modeMatchProbCV2RM_     = modeProbCV_  *p1_[2]/sumProb3;
    modeMatchProbCTRV2RM_   = modeProbCTRV_*p2_[2]/sumProb3;
    modeMatchProbRM2RM_     = modeProbRM_  *p3_[2]/sumProb3;
    modeMatchProbCA2RM_     = modeProbCA_  *p4_[2]/sumProb3;

    modeMatchProbCV2CA_     = modeProbCV_  *p1_[3]/sumProb4;
    modeMatchProbCTRV2CA_   = modeProbCTRV_*p2_[3]/sumProb4;
    modeMatchProbRM2CA_     = modeProbRM_  *p3_[3]/sumProb4;
    modeMatchProbCA2CA_     = modeProbCA_  *p4_[3]/sumProb4;
}


void UKF::Interaction() {

    MatrixXd x_pre_cv   = x_cv_;
    MatrixXd x_pre_ctrv = x_ctrv_;
    MatrixXd x_pre_rm   = x_rm_;
    MatrixXd x_pre_ca   = x_ca_;
    MatrixXd P_pre_cv   = P_cv_;
    MatrixXd P_pre_ctrv = P_ctrv_;
    MatrixXd P_pre_rm   = P_rm_;
    MatrixXd P_pre_ca   = P_ca_;

    // project CA 6D -> 5D for mixing into 5D models
    MatrixXd x_pre_ca5 = x_pre_ca.block(0, 0, 5, 1);

    // mix states for CV/CTRV/RM (5D)
    x_cv_   = modeMatchProbCV2CV_  *x_pre_cv   + modeMatchProbCTRV2CV_  *x_pre_ctrv +
              modeMatchProbRM2CV_  *x_pre_rm   + modeMatchProbCA2CV_    *x_pre_ca5;
    x_ctrv_ = modeMatchProbCV2CTRV_*x_pre_cv   + modeMatchProbCTRV2CTRV_*x_pre_ctrv +
              modeMatchProbRM2CTRV_*x_pre_rm   + modeMatchProbCA2CTRV_  *x_pre_ca5;
    x_rm_   = modeMatchProbCV2RM_  *x_pre_cv   + modeMatchProbCTRV2RM_  *x_pre_ctrv +
              modeMatchProbRM2RM_  *x_pre_rm   + modeMatchProbCA2RM_    *x_pre_ca5;

    // mix states for CA (6D): project 5D models -> 6D by appending a=0
    MatrixXd x_pre_cv6   = MatrixXd::Zero(6, 1);
    MatrixXd x_pre_ctrv6 = MatrixXd::Zero(6, 1);
    MatrixXd x_pre_rm6   = MatrixXd::Zero(6, 1);
    x_pre_cv6.block(0,0,5,1)   = x_pre_cv;
    x_pre_ctrv6.block(0,0,5,1) = x_pre_ctrv;
    x_pre_rm6.block(0,0,5,1)   = x_pre_rm;

    x_ca_ = modeMatchProbCV2CA_  *x_pre_cv6   + modeMatchProbCTRV2CA_*x_pre_ctrv6 +
            modeMatchProbRM2CA_  *x_pre_rm6   + modeMatchProbCA2CA_  *x_pre_ca;

    // preserve yaw per model (not interacting yaw)
    x_cv_(3)   = x_pre_cv(3);
    x_ctrv_(3) = x_pre_ctrv(3);
    x_rm_(3)   = x_pre_rm(3);
    x_ca_(3)   = x_pre_ca(3);

    // normalizing angle
    while (x_cv_(3)  > M_PI) x_cv_(3)   -= 2.*M_PI;
    while (x_cv_(3)  <-M_PI) x_cv_(3)   += 2.*M_PI;
    while (x_ctrv_(3)> M_PI) x_ctrv_(3) -= 2.*M_PI;
    while (x_ctrv_(3)<-M_PI) x_ctrv_(3) += 2.*M_PI;
    while (x_rm_(3)  > M_PI) x_rm_(3)   -= 2.*M_PI;
    while (x_rm_(3)  <-M_PI) x_rm_(3)   += 2.*M_PI;
    while (x_ca_(3)  > M_PI) x_ca_(3)   -= 2.*M_PI;
    while (x_ca_(3)  <-M_PI) x_ca_(3)   += 2.*M_PI;

    // covariance mixing for CV (5x5)
    MatrixXd P_pre_ca5 = P_pre_ca.topLeftCorner(5, 5);
    MatrixXd d_ca_cv   = x_pre_ca5 - x_cv_;
    P_cv_   = modeMatchProbCV2CV_    *(P_pre_cv  +(x_pre_cv -  x_cv_)*(x_pre_cv -  x_cv_).transpose()) +
              modeMatchProbCTRV2CV_  *(P_pre_ctrv+(x_pre_ctrv -x_cv_)*(x_pre_ctrv -x_cv_).transpose())+
              modeMatchProbRM2CV_    *(P_pre_rm + (x_pre_rm -  x_cv_)*(x_pre_rm -  x_cv_).transpose())+
              modeMatchProbCA2CV_    *(P_pre_ca5 + d_ca_cv * d_ca_cv.transpose());

    // covariance mixing for CTRV (5x5)
    MatrixXd d_ca_ctrv = x_pre_ca5 - x_ctrv_;
    P_ctrv_ = modeMatchProbCV2CTRV_  *(P_pre_cv+  (x_pre_cv -  x_ctrv_)* (x_pre_cv-   x_ctrv_).transpose()) +
              modeMatchProbCTRV2CTRV_*(P_pre_ctrv+(x_pre_ctrv -x_ctrv_)* (x_pre_ctrv -x_ctrv_).transpose())+
              modeMatchProbRM2CTRV_  *(P_pre_rm + (x_pre_rm -  x_ctrv_)* (x_pre_rm -  x_ctrv_).transpose())+
              modeMatchProbCA2CTRV_  *(P_pre_ca5 + d_ca_ctrv * d_ca_ctrv.transpose());

    // covariance mixing for RM (5x5)
    MatrixXd d_ca_rm = x_pre_ca5 - x_rm_;
    P_rm_   = modeMatchProbCV2RM_    *(P_pre_cv+  (x_pre_cv -  x_rm_)*(x_pre_cv -  x_rm_).transpose()) +
              modeMatchProbCTRV2RM_  *(P_pre_ctrv+(x_pre_ctrv -x_rm_)*(x_pre_ctrv -x_rm_).transpose())+
              modeMatchProbRM2RM_    *(P_pre_rm + (x_pre_rm -  x_rm_)*(x_pre_rm -  x_rm_).transpose())+
              modeMatchProbCA2RM_    *(P_pre_ca5 + d_ca_rm * d_ca_rm.transpose());

    // covariance mixing for CA (6x6): project 5D covariances -> 6D
    MatrixXd P_pre_cv6   = MatrixXd::Zero(6, 6);
    MatrixXd P_pre_ctrv6 = MatrixXd::Zero(6, 6);
    MatrixXd P_pre_rm6   = MatrixXd::Zero(6, 6);
    P_pre_cv6.topLeftCorner(5,5)   = P_pre_cv;   P_pre_cv6(5,5)   = 4.0;
    P_pre_ctrv6.topLeftCorner(5,5) = P_pre_ctrv; P_pre_ctrv6(5,5) = 4.0;
    P_pre_rm6.topLeftCorner(5,5)   = P_pre_rm;   P_pre_rm6(5,5)   = 4.0;

    MatrixXd d_cv6   = x_pre_cv6   - x_ca_;
    MatrixXd d_ctrv6 = x_pre_ctrv6 - x_ca_;
    MatrixXd d_rm6   = x_pre_rm6   - x_ca_;
    MatrixXd d_ca6   = x_pre_ca     - x_ca_;

    P_ca_ = modeMatchProbCV2CA_  *(P_pre_cv6   + d_cv6  *d_cv6.transpose()) +
            modeMatchProbCTRV2CA_*(P_pre_ctrv6 + d_ctrv6*d_ctrv6.transpose())+
            modeMatchProbRM2CA_  *(P_pre_rm6   + d_rm6  *d_rm6.transpose())+
            modeMatchProbCA2CA_  *(P_pre_ca    + d_ca6  *d_ca6.transpose());

}


/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessIMMUKF(double dt) {
    /*****************************************************************************
    *  IMM Mixing and Interaction
    ****************************************************************************/
    MixingProbability();
    Interaction();
    /*****************************************************************************
    *  Prediction
    ****************************************************************************/
    Prediction(dt, 0);
    Prediction(dt, 1);
    Prediction(dt, 2);
    PredictionCA(dt);

    /*****************************************************************************
    *  Update
    ****************************************************************************/
    UpdateLidar(0);
    UpdateLidar(1);
    UpdateLidar(2);
    UpdateLidarCA();

}

void UKF::PostProcessIMMUKF(vector<double> lambdaVec) {
    /*****************************************************************************
    *  IMM Merge Step
    ****************************************************************************/
    UpdateModeProb(lambdaVec);
    MergeEstimationAndCovariance();
}



void UKF::Ctrv(double p_x, double p_y, double v, double yaw, double yawd, double nu_a, double nu_yawdd,
               double delta_t, vector<double>&state) {
    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;


    state[0] = px_p;
    state[1] = py_p;
    state[2] = v_p;
    state[3] = yaw_p;
    state[4] = yawd_p;
}

void UKF::Cv(double p_x, double p_y, double v, double yaw, double yawd, double nu_a, double nu_yawdd,
             double delta_t, vector<double>&state) {
    //predicted state values
    double px_p = p_x + v*cos(yaw)*delta_t;
    double py_p = p_y + v*sin(yaw)*delta_t;

    double v_p = v;
    // not sure which one, works better in curve by using yaw
    double yaw_p = yaw;
//    double yaw_p = 0;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    state[0] = px_p;
    state[1] = py_p;
    state[2] = v_p;
    state[3] = yaw_p;
    state[4] = yawd_p;

}


void UKF::randomMotion(double p_x, double p_y, double v, double yaw, double yawd, double nu_a, double nu_yawdd,
                       double delta_t, vector<double>&state) {
    // double px_p   = p_x + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    // double py_p   = p_y + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    // double v_p    = v   + nu_a*delta_t;

    // double yaw_p  = yaw  + 0.5*nu_yawdd*delta_t*delta_t;
    // double yawd_p = yawd + nu_yawdd*delta_t;

    double px_p   = p_x;
    double py_p   = p_y;
    double v_p    = v;

    double yaw_p  = yaw;
    double yawd_p = yawd;

    state[0] = px_p;
    state[1] = py_p;
    state[2] = v_p;
    state[3] = yaw_p;
    state[4] = yawd_p;
}

void UKF::Ca(double p_x, double p_y, double v, double yaw, double yawd, double a,
             double nu_j, double nu_yawdd, double delta_t, vector<double>&state) {
    double px_p   = p_x + v*cos(yaw)*delta_t + 0.5*a*cos(yaw)*delta_t*delta_t;
    double py_p   = p_y + v*sin(yaw)*delta_t + 0.5*a*sin(yaw)*delta_t*delta_t;
    double v_p    = v + a*delta_t;
    double yaw_p  = yaw + yawd*delta_t + 0.5*nu_yawdd*delta_t*delta_t;
    double yawd_p = yawd + nu_yawdd*delta_t;
    double a_p    = a + nu_j*delta_t;

    state[0] = px_p;
    state[1] = py_p;
    state[2] = v_p;
    state[3] = yaw_p;
    state[4] = yawd_p;
    state[5] = a_p;
}

void UKF::PredictionCA(double delta_t) {
    VectorXd x_ = x_ca_.col(0);
    MatrixXd P_ = P_ca_;

    // augmented state (8D)
    VectorXd x_aug = VectorXd(n_aug_ca_);
    x_aug.head(n_x_ca_) = x_;
    x_aug(6) = 0;  // nu_j
    x_aug(7) = 0;  // nu_yawdd

    MatrixXd P_aug = MatrixXd::Zero(n_aug_ca_, n_aug_ca_);
    P_aug.topLeftCorner(n_x_ca_, n_x_ca_) = P_;
    P_aug(6, 6) = std_jerk_ca_ * std_jerk_ca_;
    P_aug(7, 7) = std_ca_yawdd_ * std_ca_yawdd_;

    MatrixXd L = P_aug.llt().matrixL();

    // generate sigma points
    MatrixXd Xsig_aug = MatrixXd(n_aug_ca_, 2 * n_aug_ca_ + 1);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_ca_; i++) {
        Xsig_aug.col(i + 1)              = x_aug + sqrt(lambda_aug_ca_ + n_aug_ca_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_ca_)  = x_aug - sqrt(lambda_aug_ca_ + n_aug_ca_) * L.col(i);
    }

    // predict sigma points through CA model
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++) {
        double p_x      = Xsig_aug(0, i);
        double p_y      = Xsig_aug(1, i);
        double v        = Xsig_aug(2, i);
        double yaw      = Xsig_aug(3, i);
        double yawd     = Xsig_aug(4, i);
        double a        = Xsig_aug(5, i);
        double nu_j     = Xsig_aug(6, i);
        double nu_yawdd = Xsig_aug(7, i);

        vector<double> state(6);
        Ca(p_x, p_y, v, yaw, yawd, a, nu_j, nu_yawdd, delta_t, state);

        for (int j = 0; j < n_x_ca_; j++)
            Xsig_pred_ca_(j, i) = state[j];
    }

    // predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++)
        x_ = x_ + weights_ca_(i) * Xsig_pred_ca_.col(i);

    while (x_(3) >  M_PI) x_(3) -= 2.*M_PI;
    while (x_(3) < -M_PI) x_(3) += 2.*M_PI;

    // predicted covariance
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++) {
        VectorXd x_diff = Xsig_pred_ca_.col(i) - x_;
        while (x_diff(3) >  M_PI) x_diff(3) -= 2.*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;
        P_ = P_ + weights_ca_(i) * x_diff * x_diff.transpose();
    }

    x_ca_.col(0) = x_;
    P_ca_        = P_;
}

void UKF::UpdateLidarCA() {
    VectorXd x = x_ca_.col(0);
    MatrixXd P = P_ca_;

    int n_z = 2;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ca_ + 1);

    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++) {
        Zsig(0, i) = Xsig_pred_ca_(0, i);
        Zsig(1, i) = Xsig_pred_ca_(1, i);
    }

    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++)
        z_pred += weights_ca_(i) * Zsig.col(i);

    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_ca_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
         0, std_laspy_*std_laspy_;
    S = S + R;

    MatrixXd Tc = MatrixXd::Zero(n_x_ca_, n_z);
    for (int i = 0; i < 2 * n_aug_ca_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_ca_.col(i) - x;
        Tc += weights_ca_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();

    x_ca_.col(0)  = x;
    P_ca_         = P;
    zPredCAl_     = z_pred;
    lS_ca_        = S;
    K_ca_         = K;
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t, int modelInd) {
    /*****************************************************************************
   *  Initialize model parameters
   ****************************************************************************/
    double std_yawdd, std_a;
    MatrixXd x_(x_cv_.rows(), 1);
    MatrixXd P_(P_cv_.rows(),P_cv_.cols());
    MatrixXd Xsig_pred_(Xsig_pred_cv_.rows(), Xsig_pred_cv_.cols());
    if(modelInd == 0){
        x_ = x_cv_.col(0);
        P_ = P_cv_;
        Xsig_pred_ = Xsig_pred_cv_;
        std_yawdd = std_cv_yawdd_;
        std_a     = std_a_cv_;
    }
    else if(modelInd == 1){
        x_ = x_ctrv_.col(0);
        P_ = P_ctrv_;
        Xsig_pred_ = Xsig_pred_ctrv_;
        std_yawdd = std_ctrv_yawdd_;
        std_a     = std_a_ctrv_;
    }
    else{
        x_ = x_rm_.col(0);
        P_ = P_rm_;
        Xsig_pred_ = Xsig_pred_rm_;
        std_yawdd = std_rm_yawdd_;
        std_a     = std_a_rm_;
    }

    /*****************************************************************************
    *  Generate Sigma Points
    ****************************************************************************/
    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();

    /*****************************************************************************
    *  Augment Sigma Points
    ****************************************************************************/
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a*std_a;
    P_aug(6, 6) = std_yawdd*std_yawdd;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
//    if(modelInd == 2) cout<< "x state: "<<endl<<x_rm_<<endl;
//    if(modelInd == 2) cout<< "aug x state: "<<endl<<x_aug<<endl;
    for (int i = 0; i< n_aug_; i++)
    {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_aug_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_aug_ + n_aug_) * L.col(i);
    }
//    if(modelInd == 2) cout<< "aug sigma x points: "<<endl<<Xsig_aug<<endl;

    /*****************************************************************************
    *  Predict Sigma Points
    ****************************************************************************/
    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        //extract values for better readability
        double p_x      = Xsig_aug(0, i);
        double p_y      = Xsig_aug(1, i);
        double v        = Xsig_aug(2, i);
        double yaw      = Xsig_aug(3, i);
        double yawd     = Xsig_aug(4, i);
        double nu_a     = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        vector<double> state(5);
        if(modelInd == 0)        Cv(p_x, p_y, v, yaw, yawd, nu_a, nu_yawdd, delta_t, state);
        else if(modelInd == 1) Ctrv(p_x, p_y, v, yaw, yawd, nu_a, nu_yawdd, delta_t, state);
        else           randomMotion(p_x, p_y, v, yaw, yawd, nu_a, nu_yawdd, delta_t, state);

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = state[0];
        Xsig_pred_(1, i) = state[1];
        Xsig_pred_(2, i) = state[2];
        Xsig_pred_(3, i) = state[3];
        Xsig_pred_(4, i) = state[4];
    }
//    if(modelInd == 2) cout<< "predicted sigma x points: "<<endl<<Xsig_pred_<<endl;

    /*****************************************************************************
    *  Convert Predicted Sigma Points to Mean/Covariance
    ****************************************************************************/
    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    while (x_(3)> M_PI) x_(3) -= 2.*M_PI;
    while (x_(3)<-M_PI) x_(3) += 2.*M_PI;
    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

    /*****************************************************************************
    *  Update model parameters
    ****************************************************************************/
    if(modelInd == 0){
        x_cv_.col(0) = x_;
        P_cv_ = P_;
        Xsig_pred_cv_ = Xsig_pred_;
    }
    else if(modelInd == 1){
        x_ctrv_.col(0) = x_;
        P_ctrv_ = P_;
        Xsig_pred_ctrv_ = Xsig_pred_;
    }
    else{
        x_rm_.col(0) = x_;
        P_rm_ = P_;
        Xsig_pred_rm_ = Xsig_pred_;
    }
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(int modelInd) {
    // TODO refactoring
    /*****************************************************************************
   *  Initialize model parameters
   ****************************************************************************/
    VectorXd x(x_cv_.rows());
    MatrixXd P(P_cv_.rows(),P_cv_.cols());
    MatrixXd Xsig_pred(Xsig_pred_cv_.rows(), Xsig_pred_cv_.cols());
    if(modelInd == 0){
        x = x_cv_.col(0);
        P = P_cv_;
        Xsig_pred = Xsig_pred_cv_;
    }
    else if(modelInd == 1){
        x = x_ctrv_.col(0);
        P = P_ctrv_;
        Xsig_pred = Xsig_pred_ctrv_;
    }
    else{
        x = x_rm_.col(0);
        P = P_rm_;
        Xsig_pred = Xsig_pred_rm_;
    }

//    count_++;
    //extract measurement as VectorXd
    // VectorXd z = meas_package.raw_measurements_;
    //set measurement dimension, lidar can measure p_x and p_y
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    S = S + R;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    /*****************************************************************************
    *  UKF Update for Lidar
    ****************************************************************************/
    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // //residual
    // VectorXd z_diff = z - z_pred;

    // //update state mean and covariance matrix
    // x = x + K * z_diff;
    // P = P - K*S*K.transpose();

    // while (x(3)> M_PI) x(3) -= 2.*M_PI;
    // while (x(3)<-M_PI) x(3) += 2.*M_PI;
    /*****************************************************************************
    *  Update model parameters
    ****************************************************************************/
    if(modelInd == 0){
        x_cv_.col(0)  = x;
        P_cv_         = P;
        Xsig_pred_cv_ = Xsig_pred;
        zPredCVl_     = z_pred;
        lS_cv_        = S;
        K_cv_         = K;
    }
    else if(modelInd == 1){
        x_ctrv_.col(0)  = x;
        P_ctrv_         = P;
        Xsig_pred_ctrv_ = Xsig_pred;
        zPredCTRVl_     = z_pred;
        lS_ctrv_        = S;
        K_ctrv_         = K;
    }
    else{
        x_rm_.col(0)    = x;
        P_rm_           = P;
        Xsig_pred_rm_   = Xsig_pred;
        zPredRMl_       = z_pred;
        lS_rm_          = S;
        K_rm_           = K;
    }
}

void UKF::PDAupdate(vector<VectorXd> z, int modelInd){
    VectorXd z_pred;
    MatrixXd S, x_, P_, K;
    if(modelInd == 0){
        z_pred = zPredCVl_;
        S      = lS_cv_;
        x_     = x_cv_;
        P_     = P_cv_;
        K      = K_cv_;
    }
    else if(modelInd == 1){
        z_pred = zPredCTRVl_;
        S      = lS_ctrv_;
        x_     = x_ctrv_;
        P_     = P_ctrv_;
        K      = K_ctrv_;
    }
    else if(modelInd == 2){
        z_pred = zPredRMl_;
        S      = lS_rm_;
        x_     = x_rm_;
        P_     = P_rm_;
        K      = K_rm_;
    }
    else{
        z_pred = zPredCAl_;
        S      = lS_ca_;
        x_     = x_ca_;
        P_     = P_ca_;
        K      = K_ca_;
    }

    int numMeas = z.size();
    double unitV = M_PI;
//    double valiV = unitV*sqrt(gammaG_*)
    double b = (2*M_PI*numMeas*(1-pD_*pG_))/(gammaG_*unitV*pD_);

    //residual
    VectorXd z_diff = z[0] - z_pred;

    //calculate NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    while (x_(3)> M_PI) x_(3) -= 2.*M_PI;
    while (x_(3)<-M_PI) x_(3) += 2.*M_PI;

    /*****************************************************************************
    *  Update model parameters
    ****************************************************************************/
    if(modelInd == 0){
        x_cv_.col(0)  = x_;
        P_cv_         = P_;
    }
    else if(modelInd == 1){
        x_ctrv_.col(0)  = x_;
        P_ctrv_         = P_;
    }
    else if(modelInd == 2){
        x_rm_.col(0)    = x_;
        P_rm_           = P_;
    }
    else{
        x_ca_.col(0)    = x_;
        P_ca_           = P_;
    }
}