// GrabCutMF.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "DenseCRF.h"
#include "GrabCutMF.h"
#include "CmGrabSal.h"

void RenameOneCutResult(CStr dir);

void GetForeVsBackRatio(CStr dir);

void checkDataSet(CStr wkDir);

int main(int argc, char* argv[]) {
    
    if(argc <= 1) {
        cout<<"arguments: {work-directory}"<<endl;
        return -1;
    }
    
	CStr rootDir = argv[1];
	//CStr wkDir = rootDir + "GrabCut/";
	CStr wkDir = rootDir + "ASD/";
	printf("WkDir = %s\n", _S(wkDir));
	
	GrabCutMF::Demo(wkDir, 6, 10, 2, 20, 33, 3, 41); // Best setting 0.958119
		
	//GrabCutMF::Demo(wkDir, 10, 0, 3, 15, 20, 0, 45); // 0.951792



	//checkDataSet(wkDir); 
	//GrabCutMF::runGrabCutOpenCV(wkDir);
	
	//RenameOneCutResult("D:/WkDir/OneCut/GrabCut/128bins/");
	//CmEvaluation::EvalueMask(wkDir + "Imgs/*.png", wkDir + "Sal/", ".png", "_OneCut.png");
	//GetForeVsBackRatio(wkDir);

	//getGrabSal2(imread("./Data/In1.png"), Rect(134, 137, 306, 159)); // For image 1
	//getGrabSal2(imread("./Data/In2.png"), Rect(220, 85, 206, 290)); // For image 2

	//const char* exts[] = {"GC", "GCMF", "CudaG4", "CudaH"};
	//CmEvaluation::EvalueMask(wkDir + "Imgs/*.png", wkDir + "Sal/", charPointers2StrVec(exts), wkDir + "Res.m");


	return 0;
}

void RenameOneCutResult(CStr dir)
{
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(dir + "*.bmp", namesNE);
	for (int i = 0; i < imgNum; i++)
		imwrite(dir + namesNE[i] + "_OneCut.png", imread(dir + namesNE[i] + ".bmp"));
}

void GetForeVsBackRatio(CStr dir)
{
	double sumV = 0;
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(dir + "Imgs/*.jpg", namesNE);
	for (int i = 0; i < imgNum; i++){
		Mat mask1u = CmFile::LoadMask(dir + "Imgs/" + namesNE[i] + ".png");
		blur(mask1u, mask1u, Size(3,3));
		Rect wkRect = CmCv::GetMaskRange(mask1u, 5, 200);
		mask1u = mask1u(wkRect);

		double v = sum(mask1u).val[0];
		sumV += v/(wkRect.width*wkRect.height*255 - sum(mask1u).val[0] + EPS);
	}
	printf("Average ratio = %g\n", sumV/imgNum);
}

void checkDataSet(CStr wkDir)
{
	CStr inDir = wkDir + "Imgs/", salDir = wkDir + "GrabCut/", outDir = wkDir + "Ranked2/";
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(inDir + "*.jpg", namesNE);
	CmFile::MkDir(outDir);
	vector<pair<double, string>> scoreName(imgNum);

#pragma omp parallel for
	for (int i = 0; i < imgNum; i++){
		Mat img = imread(inDir + namesNE[i] + ".jpg"); 
		Mat gt1u = CmFile::LoadMask(inDir + namesNE[i] + ".png"), bound1u, boundE1u;
		CV_Assert(img.size == gt1u.size && img.data != NULL);
		Canny(gt1u, bound1u, 30, 100);
		dilate(bound1u, bound1u, Mat());
		dilate(bound1u, boundE1u, Mat());
		img.setTo(Scalar(255, 255, 0), boundE1u);
		img.setTo(Scalar(0, 0, 255), bound1u);
		imwrite(inDir + namesNE[i] + "_Ilu.png", img);

		Mat res1u = CmFile::LoadMask(salDir + namesNE[i] + "_GCMF.png");
		scoreName[i] = make_pair(CmEvaluation::FMeasure(res1u, gt1u), namesNE[i]);
	}

	sort(scoreName.begin(), scoreName.end());

#pragma omp parallel for
	for (int i = 0; i < imgNum; i++){
		CStr nameNE = scoreName[i].second, outNameNE = format("%d_", i) + scoreName[i].second;

		Mat gt1u = imread(inDir + nameNE + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		blur(gt1u, gt1u, Size(3,3));
		Rect rect = CmCv::GetMaskRange(gt1u, 6, 128);
		Mat img = imread(inDir + nameNE + ".jpg");
		rectangle(img, rect, Scalar(0, 0, 255), 2);
		imwrite(outDir + outNameNE + ".jpg", img);
		CmFile::Copy(inDir + nameNE + "_Ilu.png", outDir + outNameNE + "_Ilu.png");
		CmFile::Copy(salDir + nameNE + "_GCMF.png", outDir + outNameNE + "_GCMF.png");
		CmFile::Copy(salDir + nameNE + "_GT.png", outDir + outNameNE + "_GT.png");
	}
}
