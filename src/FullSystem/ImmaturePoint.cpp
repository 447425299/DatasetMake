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



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace dso
{
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{

	gradH.setZero();

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];
// 		if(!(u+dx >12 && v+dy >12 && u+dx < wG[0]-12 && v+dy < hG[0]-12))
// 		{
// 		   dx=0;
// 		   dy=0;
// 		}

        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]);

		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}


		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}
	for(int idx=0;idx<patternPatchNum;idx++)
	{
		int dx = patternPatch[idx][0];
		int dy = patternPatch[idx][1];
		int udx=u+dx;
		int udy=v+dy;//可以先给个错误的值，反正也用不上； 
		color16_bool =false;
		if(!(udx >2 && udy >2 && udx < wG[0]-1-1 && udy < hG[0]-1-1))//patternPatch 记得修改这里的条件 change8 尝试减去约束
		{
		   udx=u;
		   udy=v;
		  color16_bool =true;
		}
		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, udx, udy,wG[0]);



		color16[idx] = ptc[0]; //TODO 有无穷大 或者超出边界了，怎么办？　这里是不是会有坏值啊
		if(!std::isfinite(color16[idx])) {energyTH=NAN; return;}//change 尝试减去约束


//		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

//		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}
	
	energyTH = patternNum*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;
}

ImmaturePoint::~ImmaturePoint()
{
}



/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{	//NOTE 对这些点进行跟踪,返回值为对该点的状态.若跟踪成功则完成匹配.
  // 为甚OOB的时候就返回呢,mag掉的就只能继续mag掉?

	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;//上帧该点被mag,该帧就是相同的状态


	debugPrint = false;//rand()%100==0;
	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;//深度值搜索的最大值??

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, idepth_max,
				hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

//	const float stepsize = 1.0;				// stepsize for initial discrete search.
//	const int GNIterations = 3;				// max # GN iterations
//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
	// ============== project min and max. return if one of them is OOB ===================
	//最小和最大的深度,其中之一落在的图像外边则直接返回,改点定义为OOB并返回
	//用最小的深度值计算
	//相关计算
	Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
	Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;  //idepth_min有关尺度的量
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	if(!(uMin > patternPadding+1 && vMin >patternPadding+1 && uMin < wG[0]-patternPadding-1 && vMin < hG[0]-patternPadding-1))//看是否映射到了图像上。change8
// 	if(!(uMin > 2 && vMin >2 && uMin < wG[0]-2 && vMin < hG[0]-2))//看是否映射到了图像上。change8　尝试减去约束
	{//没有投影到图像上
		if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
				u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	//用最大的深度值计算
	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	if(std::isfinite(idepth_max))//深度最大值有限
	{
		ptpMax = pr + hostToFrame_Kt*idepth_max;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];


		if(!(uMax > patternPadding+1 && vMax > patternPadding+1 && uMax < wG[0]-patternPadding-1 && vMax < hG[0]-patternPadding-1))//change8
// 		if(!(uMax > 2 && vMax > 2 && uMax < wG[0]-2 && vMax < hG[0]-2))//change8 尝试减去约束
		{
			if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
			lastTraceUV = Vec2f(-1,-1);//给-1就是坏值,不用了
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}


		//到了这一步证明没有被定义为OOB,地图点投影到了该帧,未超出边界
		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		//和深度滤波类似,投影到新帧为一个线段,如果线段太短,就不用再滤波优化了,直接去个中间值就行,在优化没太大意义,设为IPS_SKIPPED
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
		dist = sqrtf(dist);
		if(dist < setting_trace_slackInterval)
		{
			if(debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;//把位置设置了
			lastTracePixelInterval=dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}
		assert(dist>0);
	}
	else//深度的最大值无限大
	{
		dist = maxPixSearch;//自己定了个深度最大值,再远的点不要了,要了也没用.

		// project to arbitrary depth to get direction.
		ptpMax = pr + hostToFrame_Kt*0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		float dx = uMax-uMin;
		float dy = vMax-vMin;
		float d = 1.0f / sqrtf(dx*dx+dy*dy);

		// set to [setting_maxPixSearch].
		uMax = uMin + dist*dx*d;
		vMax = vMin + dist*dy*d;

		// may still be out!
		//在这种情况下,由于深度最大值很大,线段不可能短,所以不会成为SKIP,单有个能超出图像外,那就不要了,给-1.设置IPS_OOB
		if(!(uMax > patternPadding+1 && vMax > patternPadding+1 && uMax < wG[0]-patternPadding-1 && vMax < hG[0]-patternPadding-1))//change8
// 		if(!(uMax > 2 && vMax > 2 && uMax < wG[0]-2 && vMax < hG[0]-2))//change8　尝试减去约束
		{
			if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist>0);
	}


	// set OOB if scale change too big.
	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))//出现深度为负,或尺度变化较大,那也不要了,给-1.设置IPS_OOB
	{
		if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}


	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin);//setting_trace_stepsize为初始离散搜索步长,事先给定的.之后再迭代改变
	float dy = setting_trace_stepsize*(vMax-vMin);

	float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));// gradH 2*2的信息矩阵??
	float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
	float errorInPixel = 0.2f + 0.2f * (a+b) / a;

	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))// 
	{//这个的意思是在优化也没有什么提升空间了,所以不优化了,取个中间值算球.
		if(debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
		lastTracePixelInterval=dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if(errorInPixel >10) errorInPixel=10;


	//其他特殊情况都排除了,剩下就是进行深度滤波计算了,根据投影找到最合适的深度
	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, uMin, vMin,
				idepth_max, uMax, vMax,
				errorInPixel
				);


	if(dist>maxPixSearch)//搜索空间太长就截掉一部分,太远的不要了
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;//搜索步骤数
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>(); //投影计算 崔书豪

	float randShift = uMin*1000-floorf(uMin*1000);// 
	float ptx = uMin-randShift*dx;
	float pty = vMin-randShift*dy;

	//rotatetPattern代表那8个像素的位置//要在这里有改动啊
	Vec2f rotatetPattern[MAX_RES_PER_POINT];//这应该就是那个图像块,8个的间隔的,patten是指的采样模型
	for(int idx=0;idx<patternNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);//平面旋转，这个也跟着转了一下 就是那8个点也跟着转了




	if(!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}



	float errors[100];
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) numSteps = 99;//errors就100维的,所以顶多99步,要不占不下了,报错

	for(int i=0;i<numSteps;i++)
	{//迭代的步数
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{//pattern的8个像素 遍历
			
		  	if(!(ptx+rotatetPattern[idx][0] > 1 && pty+rotatetPattern[idx][1] >1 && ptx+rotatetPattern[idx][0] < wG[0]-2 && pty+rotatetPattern[idx][1] < hG[0]-2))//看是否映射到了图像上。
			{
			  rotatetPattern[idx][1] =0;
			    rotatetPattern[idx][0]=0; 
			}
			  
			  float hitColor = getInterpolatedElement31(frame->dI,
										(float)(ptx+rotatetPattern[idx][0]),
										(float)(pty+rotatetPattern[idx][1]),
										wG[0]);//插值方法而已
// 			std::cout<<"hitColor"<< ptx+rotatetPattern[idx][0]<<"  "<<pty+rotatetPattern[idx][1]<<std::endl;

			if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);//将参考帧的像素值转换过来求差
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}//求出了误差和


		if(debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n",
					ptx, pty, 0.0f, energy);


		errors[i] = energy;//保存
		if(energy < bestEnergy)
		{
			bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
		}

		ptx+=dx;
		pty+=dy;
	}//迭代完毕,找到最佳的像素位置,这个过程其实就是匹配的过程, NOTE 完成初步匹配.

// 	      std::cout<<"hitColor"<<std::endl;
	// find best score outside a +-2px radius.
	//找到最佳2个像素之外的最佳值
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++)
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;//这个点的质量,也就是这个位置的重要性
	if(newQuality < quality || numSteps > 10) quality = newQuality;

	//现在深度值求出来了,接下来就是跟踪,采用高斯牛顿法优化
	// ============== do GN optimization ===================
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) bestEnergy = 1e5;//最大迭代步数,设为3
	int gnStepsGood=0, gnStepsBad=0;
	//开始迭代,一共三步(setting_trace_GNIterations)
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{//8个像素一个个来算
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
					(float)(bestU+rotatetPattern[idx][0]),
					(float)(bestV+rotatetPattern[idx][1]),wG[0]);

			if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}//插值出来的值无穷大,那就不好办了.给个最大值,作为惩罚
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float dResdDist = dx*hitColor[1] + dy*hitColor[2];// 这是什么意思 Res代表residual误差
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;//hessien矩阵
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack*=0.5;
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
			if(debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, stepBack,
						uBak, vBak, bestU, bestV);
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU;
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

			if(debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, step,
						uBak, vBak, bestU, bestV);
		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) break;
	}//迭代完毕,找到最佳的像素位置,这个过程其实就是匹配的过程, NOTE 采用牛顿迭代完成进一步匹配.


	// ============== detect energy-based outlier. ===================
//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))//最后的像素差还是很大
//			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
//		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
//			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if(debugPrint)
			printf("OUTLIER!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);//没找到对应的位置
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}


	// ============== set new interval ===================
	//确定深度的搜索范围,为接下来的继续滤波使用
	if(dx*dx>dy*dy)
	{
		idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
		idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
	}
	else
	{
		idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
		idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
	}
	if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


	if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval=2*errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}//看完搞定


float ImmaturePoint::getdPixdd(
		CalibHessian *  HCalib,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	float drescale, u=0, v=0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
			precalc->PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

	float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl();
	float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl();
	return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(this->u+patternP[idx][0], this->v+patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if(!std::isfinite((float)hitColor[0])) {return 1e10;}
		//if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
	}

	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
	}
	return energyLeft;
}




double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	// check OOB due to scale angle change.

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		float dxInterp = hitColor[1]*HCalib->fxl();
		float dyInterp = hitColor[2]*HCalib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth;
		bd += (hw*residual)*d_idepth;
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}



}
