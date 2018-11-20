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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>



#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;


int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{// 这个是什么意思?
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();//收到信号就唤醒线程
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}






void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}

	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}
		if(1==sscanf(arg,"filepath=%s",buf))    //csh
	{
		filepath  = buf;
		printf("loading filepath from %s!\n", filepath.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}



int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
  
    int j_pattern=0;
    int k_pattern=0;
    for(int i_pattren=0;i_pattren<4096;i_pattren++) //staticPattern64
    {
      staticPattern64[i_pattren][0]=j_pattern-32;
      staticPattern64[i_pattren][1]=k_pattern-32;
      if(i_pattren%64 == 63  )     {
	j_pattern = 0;
	k_pattern++;	
      }
      j_pattern++;
            
    }
    
      j_pattern=0;
    k_pattern=0;    
     for(int i_pattren=0;i_pattren<3136;i_pattren++) //staticPattern56
    {
      staticPattern56[i_pattren][0]=j_pattern-28;
      staticPattern56[i_pattren][1]=k_pattern-28;
      if(i_pattren%56 == 55  )     {
	j_pattern = 0;
	k_pattern++;	
      }
      j_pattern++;            
    }
    
      
    j_pattern=0;
    k_pattern=0;    
     for(int i_pattren=0;i_pattren<1024;i_pattren++) //staticPattern32
    {
      staticPattern32[i_pattren][0]=j_pattern-16;
      staticPattern32[i_pattren][1]=k_pattern-16;
      if(i_pattren%32 == 31  )     {
	j_pattern = 0;
	k_pattern++;	
      }
      j_pattern++;            
    }
    
	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);//接受输入的参数

	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);//  这个线程是不是负责判断何时暂停，键入随便一个键就暂停
// 	cv::Mat iamge = cv::imread("/home/lsp/PycharmProjects/metric_learn/project/1.png");
	
// 	std::cout<<iamge<<std::endl;
	
	ImageFolderReader* reader = new ImageFolderReader(source,calib, gammaCalib, vignette);//读入配置的参数
	reader->setGlobalCalibration();//将配置文件传给相应的变量
	//调试的时候实在不行就把值表示出来呗
//	Mat33f Ktest = Mat33f::Identity();
//	std::cout<<Ktest<<std::endl;

	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}




	int lstart=start;
	int lend = end;
	int linc = 1;
	if(reverse)//如果数据集倒着执行
	{
		printf("REVERSE!!!!");
		lstart=end-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
		lend = start;
		linc = -1;
	}



	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());//gama矫正，如果存在伽马文件的话
	fullSystem->linearizeOperation = (playbackSpeed==0);//最快地跑，还是按照时间戳的时间，playbackSpeed这个变量后面的ｆｌａｇ






//负责显示
    IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());



// 都是用的这个runThread　PangolinDSOViewer也是用的这个线程，为什么要共用一个如何管理的，在单线程的模式下又是怎么运行的，不是可以选的吗？不是同一个变量
    // to make MacOS happy: run this in dedicated Thread -- and use this one to run the GUI.
    std::thread runthread([&]() {//这个线程用来跟踪优化,主线程是IOWrap的线程
        std::vector<int> idsToPlay;//图片的ID
        std::vector<double> timesToPlayAt;//真实的时间，从零开始
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)//linc只有１或－１是为了实现让数据集倒着跑而设置的变量
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);//为了获取真实的时间
            }
        }


        std::vector<ImageAndExposure*> preloadedImages;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;
//自己为画时间图加的
      std::vector<float> vTimesTrackToPaint;    
      int NNN=(int)idsToPlay.size();
      vTimesTrackToPaint.resize(NNN);
      
      for(int ii=0;ii<(int)idsToPlay.size(); ii++)//少朋
//      for(int ii=0;ii<1; ii++)
      {
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img;
            if(preload)
                img = preloadedImages[ii];
            else
                img = reader->getImage(i);


//这部分是为了实时的跑程序．
            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));// 这两种计时方式有什么不一样的地方吗？

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }
            
            //界限来的工作8.3 把addactiveframe部分证明白再读其他的

//#ifdef COMPILEDWITHC11
	    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//#else		activeResiduals[k]->applyRes(true);

//        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
//#endif

            if(!skipFrame) fullSystem->addActiveFrame(img, i);

//#ifdef COMPILEDWITHC11
          std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//#else
//        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
//#endif
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	vTimesTrackToPaint[ii]=ttrack;

            delete img;
//如果失败或重置
            if(fullSystem->initFailed || setting_fullResetRequested)//按下了重置按钮
            {
                if(ii < 250 || setting_fullResetRequested)//哦 , 如果250帧过去了,还是没有初始化成功,那就算了吧 
                {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();//ｒｅｓｅｔ的时候把原来的指针删掉腾出空间
		    //重新初始化
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed==0);


                    fullSystem->outputWrapper = wraps;//接着按原本来的显示，只不过每个指针都ｒｅｓｅｔ了

                    setting_fullResetRequested=false;
                }
            }

            if(fullSystem->isLost)
            {//丢了就算了
                    printf("LOST!!\n");
                    break;
            }

        } //循环结束了
        fullSystem->blockUntilMappingIsFinished();//把mapping,先结束了
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

//这是我自己加上去的
//         fullSystem->printResult("result.txt");//没有其他再用了啊
//         fullSystem->printTimePerFrame("TimePerFrame.txt",vTimesTrackToPaint);
	char path_cost[255];
	 sprintf(path_cost, "/home/lsp/PycharmProjects/metric_learn/%s/So_Cost_test.txt", filepath.c_str());   //_0
        fullSystem->print_So_Cost(path_cost);//没有其他再用了啊 //pytorch-siamese/makepatch//change  //csh  //"/home/lsp/PycharmProjects/%s/So_Cost_test_032.txt", filepath.c_str()
//         fullSystem->print_Sp("/home/lsp/PycharmProjects/metric_learn/Sp.txt");//没有其他再用了啊

	

        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());//帧的数目
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));//时间戳经历的时间
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);//单线程用的时间
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"// 多线程和单线程怎么选择的用哪个方式计算?
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
  //      fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }
        
 //       exit(1);

    });

      std::cout<<"daozhelila"<<std::endl;
    if(viewer != 0)
        viewer->run();//这个才是主线程

    runthread.join();
      std::cout<<"daozhelila1"<<std::endl;

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}



	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("EXIT NOW!\n");
	return 0;
}
