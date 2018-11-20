/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>



cv::Mat patch1(PatchSize,PatchSize,CV_32F); //change8
cv::Mat patch2(PatchSize,PatchSize,CV_32F); 
cv::Mat patch3(PatchSize,PatchSize,CV_32F); 
int m_count=0;
int n_count=0;
 int n_writeout=0;




namespace dso
{
 std::vector<float> Costtowrite; 
  std::vector<float> Normtowrite; 
  std::vector<float> Costallwrite; 
 std::vector<std::string> So_prt; 
 std::vector<std::string> Sp_prt; 


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
//	f.open("cost.txt");
	instanceCounter++;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	int n16=patternPatchNum;
	color16_bool =rawPoint->color16_bool;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(color16, rawPoint->color16, sizeof(float)*n16);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0;


}


void PointHessian::makepatch(FrameHessian* frame,const Mat33f &K, const SE3 &hostToNew, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{//TODO 还可以生成加了a、b矫正的数据集。

//	Mat33f hostToFrame_KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
//	Vec3f hostToFrame_Kt = K * hostToNew.translation().cast<float>();
//	std::cout<<hostToFrame_KRKi<<std::endl<<hostToFrame_Kt<<std::endl;
	//加干扰
      float hitu0_save,hitv0_save;
	if(color16_bool) return;
	if(n_writeout>13000) return;
	Vec6 update_se3; //加入的干扰量
	for(int i=0;i<30;i++)
	{
	update_se3.setZero();	
// 	if(i!=0)
// 	{
	  for (int i_up=0;i_up<6;i_up++){//TODO 可改成离散的固定值
// 	    update_se3(i_up) =0; //-0.006~0.006
  	    update_se3(i_up) =rand()/(double)(RAND_MAX)*0.12-0.06; //-0.006~0.006
//   	    update_se3(i_up) =i*0.03/30; //-0.006~0.006
	    } 
 	 if(i==0){
 	   	update_se3.setZero();		   
 	 }
// 	}
	
	SE3 hostToNew_delta = SE3::exp(update_se3)*hostToNew;
	Mat33f hostToFrame_KRKi1 = K * hostToNew_delta.rotationMatrix().cast<float>() * K.inverse();
	Vec3f hostToFrame_Kt1 = K * hostToNew_delta.translation().cast<float>();
	
// 	std::cout<<update_se3<<std::endl;
  	Vec3f pr1 = hostToFrame_KRKi1 * Vec3f(u, v, 1);
	
//	Vec3f hitpoint = pr + hostToFrame_Kt*idepth;  //idepth_min有关尺度的量	
	Vec3f hitpoint1 = pr1 + hostToFrame_Kt1*idepth;  //idepth_min有关尺度的量
 
 	float hitu1 = hitpoint1[0] / hitpoint1[2];
	float hitv1 = hitpoint1[1] / hitpoint1[2];
	 if(i==0){
	   	hitu0_save=hitu1;
	   	hitv0_save=hitv1;	
	 }
	 
	if(!(hitu0_save > 3 && hitv0_save >3 && hitu0_save < wG[0]-3 && hitv0_save < hG[0]-3))
	  break; 
	 
	 float uv_distance=(hitu0_save-hitu1)*(hitu0_save-hitu1)+(hitv0_save-hitv1)*(hitv0_save-hitv1);
	 
	 
 
	if(!(hitu1 > 3 && hitv1 >3 && hitu1 < wG[0]-3 && hitv1 < hG[0]-3))//看是否映射到了图像上。
	  return; 

	
	
	Mat22f Rplane = hostToFrame_KRKi1.topLeftCorner<2,2>(); //TODO 核对hostToFrame_KRKi1矩阵
	Vec2f rotatetPattern[patternPatchNum];//这应该就是那个图像块,256个的间隔的,patten是指的采样模型
	
	for(int idx=0;idx<patternPatchNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternPatch[idx][0], patternPatch[idx][1]);//平面旋转，这个也跟着转了一下 就是那个点也跟着转了
	
      for(int idx=0;idx<patternPatchNum;idx++)
      {//uv 加上模型别超出
	//投射出来后也别超出。
	if(!(u+patternPatch[idx][0] > 3 && v+patternPatch[idx][1] >3 && u+patternPatch[idx][0] < wG[0]-3 && v+patternPatch[idx][1] < hG[0]-3))
	  return;
	
	if(!(hitu1+rotatetPattern[idx][0] > 3 && hitv1+rotatetPattern[idx][1] >3 && hitu1+rotatetPattern[idx][0] < wG[0]-3 && hitv1+rotatetPattern[idx][1] < hG[0]-3))
	  return;
	
	float hitColor1 = getInterpolatedElement31(frame->dI,
								(float)(hitu1+rotatetPattern[idx][0]),
								(float)(hitv1+rotatetPattern[idx][1]),
								wG[0]);//插值方法而已
	
// 	float ucolor=u+staticPattern64[idx][0];
// 	float vcolor=v+staticPattern64[idx][1];
// 	if(!(ucolor > 3 && vcolor >3 && ucolor < wG[0]-3 && vcolor < hG[0]-3))
// 	  return;

 // 	Vec3f pr = hostToFrame_KRKi * Vec3f(ucolor,vcolor, 1);
//   	Vec3f pr1 = hostToFrame_KRKi1 * Vec3f(ucolor,vcolor, 1);
// 	
// //	Vec3f hitpoint = pr + hostToFrame_Kt*idepth;  //idepth_min有关尺度的量	
// 	Vec3f hitpoint1 = pr1 + hostToFrame_Kt1*idepth;  //idepth_min有关尺度的量

//	float hitu = hitpoint[0] / hitpoint[2];
//	float hitv = hitpoint[1] / hitpoint[2];
	
// 	float hitu1 = hitpoint1[0] / hitpoint1[2];
// 	float hitv1 = hitpoint1[1] / hitpoint1[2];
	
//	if(!(hitu > 3 && hitv >3 && hitu < wG[0]-3 && hitv < hG[0]-3))//看是否映射到了图像上。
//	  return;      
// 	if(!(hitu1 > 3 && hitv1 >3 && hitu1 < wG[0]-3 && hitv1 < hG[0]-3))//看是否映射到了图像上。
// 	  return;     
	
	//等会自己生成pattern模型
//	float hitColor = getInterpolatedElement31(frame->dI,
//										(float)(hitu),
//										(float)(hitv),
//										wG[0]);
// 	float hitColor1 = getInterpolatedElement31(frame->dI,
// 										(float)( hitu1),
// 										(float)(hitv1),
// 										wG[0]);

	int m = idx%PatchSize;//change8
	int n=floor(idx/PatchSize);
//	patch1.at<float>(m+m_count*16,n+n_count*16)=hitColor;
//	patch3.at<float>(m+m_count*16,n+n_count*16)=hitColor1;
//	patch2.at<float>(m+m_count*16,n+n_count*16)=color16[idx];

	patch2.at<float>(m,n)=hitColor1;
	patch1.at<float>(m,n)=color16[idx];
	
//	float errorColor=hitColor-color16[idx];
      }
//能到这一步证明获得了图像块
//      f<<update_se3.norm()<<std::endl;
//      std::cout<<update_se3.norm()<<std::endl;
      
//       if(n_count%100==0)
//       {
//*********************输出所有的位姿***************************
//       for (int i_up=0;i_up<6;i_up++){
// 	    Costallwrite.push_back(update_se3(i_up) );
//       } 
//*********************输出所有的位姿***************************
      float upNorm=(float)update_se3.norm();
      Normtowrite.push_back(upNorm);
//  	 if(i!=0){
	      Costtowrite.push_back(uv_distance);
//  	 }      
//      m_count++;
//      if(m_count>99)//160*160 就是9
//      {
//	m_count=0;
//	n_count++;
//	if(n_count>99)
//	{
//	  n_count=0;
//	  char ph1="%06i.png"
	  char s1[255], s2[255],so_prt[255],sp_prt[255];
//	 std::string so_prt,sp_prt;//,s3[255]
	  sprintf(s1, "/home/lsp/PycharmProjects/metric_learn/%s/original_test/%d.jpg", filepath.c_str(), n_writeout); // 这里是绝对路径 test csh  original_test_0
	  sprintf(s2, "/home/lsp/PycharmProjects/metric_learn/%s/project_test/%d.jpg", filepath.c_str(), n_writeout); //csh project_test_0
// 	  sprintf(s1, "/home/lsp/PycharmProjects/metric_learn/data/TUMmono32_32/sequence_01/original_test/%d.jpg", n_writeout); // 这里是绝对路径 test csh  original_test_0
// 	  sprintf(s2, "/home/lsp/PycharmProjects/metric_learn/data/TUMmono32_32/sequence_01/project_test/%d.jpg",  n_writeout); //csh project_test_0
	  sprintf(so_prt, "original_test/%d.jpg", n_writeout);//project_test_0
	  sprintf(sp_prt, "project_test/%d.jpg", n_writeout);//project_test_0
	  So_prt.push_back(so_prt);
	  Sp_prt.push_back(sp_prt);
	  
	  
	  
//	  sprintf(s3, "3p%d.png", n_writeout);

	// 先改成单独存放的 再改好路径  
// 	  std::cout<<patch1<<std::endl;
//  	 if(i!=0){
	  cv::imwrite(s1,patch1);
          cv::imwrite(s2,patch2);
	  n_writeout++;
//  	 }  	  

//	  cv::imwrite(s3,patch3);
//       }
//       n_count++;
//	  cv::imshow("show",patch1);
//	}
      }
  }
   



//	std::cout<<"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"<<errorColor<<"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"<<std::endl;
  
  



void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}


void FrameHessian::setStateZero(const Vec10 &state_zero)
{//求仿射变换矩阵的参数
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;

// 李群里面的计算,加一个小扰动
	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}


void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{

	for(int i=0;i<pyrLevelsUsed;i++)//pyrLevelsUsed为常值６，为金字塔的层数
	{//先完成初始化
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];//分别代表像素值，ｄｘ，ｄｙ
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	dI = dIp[0];//最下边一层


	// make d0
	int w=wG[0];
	int h=hG[0];
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl];
		Eigen::Vector3f* dI_l = dIp[lvl];

		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];



			for(int y=0;y<hl;y++)//降采样
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy;

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;

	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;


	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

