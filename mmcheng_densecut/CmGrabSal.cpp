#include "stdafx.h"
#include "CmGrabSal.h"



Mat getGrabSal(Mat img, Rect _rect)
{
	const int border = 30;
	CmGMM _gmmB(4), _gmmF(2);

	Point tlPnt = Point(max(_rect.x - border, 0), max(_rect.y - border, 0));
	Point brPnt = Point(min(_rect.x + _rect.width + border, img.cols), min(_rect.y + _rect.height + border, img.rows));
	Rect wkRect(tlPnt, brPnt);

	Mat img3f;
	img(wkRect).convertTo(img3f, CV_32F, 1.0/255);
	Rect rect(_rect.x - tlPnt.x, _rect.y - tlPnt.y, _rect.width, _rect.height);


	CmTimer tm("Timer");
	tm.Start();

	Mat _HistBinIdx1i, _HistBinClr3f, colorNums1i;
	int _binNum = CmColorQua::D_Quantize(img3f, _HistBinIdx1i, _HistBinClr3f, colorNums1i);
	Mat mask1u = Mat::zeros(img3f.size(), CV_8U);
	mask1u(rect) = 255;
	int h = img3f.rows, w = img3f.cols;
	Mat HistBinWf = Mat::zeros(1, _binNum, CV_32S);
	Mat HistBinWb = Mat::zeros(1, _binNum, CV_32S);
	int *bW = (int*)HistBinWb.data, *fW = (int*)HistBinWf.data;
	for (int r = 0; r < h; r++){
		const byte* m = mask1u.ptr<byte>(r);
		const int* idx = _HistBinIdx1i.ptr<int>(r);
		for (int c = 0; c < w; c++){
			if (m[c])
				fW[idx[c]]++;
			else
				bW[idx[c]]++;
		}
	}

	//CmShow::HistBins(_HistBinClr3f, HistBinWf, "_QTF", false);
	//CmShow::HistBins(_HistBinClr3f, HistBinWb, "_QTB", false);
	HistBinWb.convertTo(HistBinWb, CV_32F);
	HistBinWf.convertTo(HistBinWf, CV_32F);

	Mat compF1i, compB1i;
	_gmmB.BuildGMMs(_HistBinClr3f, compB1i, HistBinWb);
	_gmmB.RefineGMMs(_HistBinClr3f, compB1i, HistBinWb);

	_gmmF.BuildGMMs(_HistBinClr3f, compF1i, HistBinWf);
	_gmmF.RefineGMMs(_HistBinClr3f, compF1i, HistBinWf);

	Mat histBinSal1f(1, _binNum, CV_32F);
	float* binSal = (float*)histBinSal1f.data;
	Vec3f* binClr = (Vec3f*)_HistBinClr3f.data;
	for (int i = 0; i < _binNum; i++){
		float bP = _gmmB.P(binClr[i]), fP = _gmmF.P(binClr[i]);
		binSal[i] = fP / (bP*4.0f + fP + 1e-10f);
	}

	tm.StopAndReport();


	//CmShow::HistBins(_HistBinClr3f, histBinSal1f, "Histogram bin saliency", false);


	Mat sal1f = Mat::zeros(img.size(), CV_32F);
	Mat _sal1f = sal1f(wkRect);
	for (int r = 0; r < h; r++){
		const int* idx = _HistBinIdx1i.ptr<int>(r);
		float* salV = _sal1f.ptr<float>(r);
		for (int c = 0; c < w; c++)
			salV[c] = binSal[idx[c]];
	}


	imshow("Saliency", sal1f);
	Mat show3u;
	vecM chns;
	split(img, chns);
	sal1f.convertTo(chns[2], CV_8U, 255);
	merge(chns, show3u);

	rectangle(img, _rect, Scalar(0, 0, 255));
	rectangle(img, wkRect, Scalar(0, 255, 255));
	imshow("Image", img);
	imshow("Illustrate", show3u);
	waitKey(0);
	return Mat();
}



Mat getGrabSal2(Mat img, Rect _rect)
{
	const int border = 30;
	CmGMM _gmm(5);

	Point tlPnt = Point(max(_rect.x - border, 0), max(_rect.y - border, 0));
	Point brPnt = Point(min(_rect.x + _rect.width + border, img.cols), min(_rect.y + _rect.height + border, img.rows));
	Rect wkRect(tlPnt, brPnt);

	Mat img3f;
	img(wkRect).convertTo(img3f, CV_32F, 1.0/255);
	Rect rect(_rect.x - tlPnt.x, _rect.y - tlPnt.y, _rect.width, _rect.height);


	CmTimer tm("Timer");
	tm.Start();

	Mat _HistBinIdx1i, _HistBinClr3f, clrW1f, comp1i;
	int _binNum = CmColorQua::D_Quantize(img3f, _HistBinIdx1i, _HistBinClr3f, clrW1f);
	clrW1f.convertTo(clrW1f, CV_32F);
	_gmm.BuildGMMs(_HistBinClr3f, comp1i, clrW1f);
	_gmm.RefineGMMs(_HistBinClr3f, comp1i, clrW1f);


	Mat mask1u = Mat::zeros(img3f.size(), CV_8U);
	mask1u(rect) = 1;
	int h = img3f.rows, w = img3f.cols;
	Mat HistBinWf = Mat::zeros(1, _binNum, CV_32F);
	float *fW = (float*)HistBinWf.data;
	for (int r = 0; r < h; r++){
		const byte* m = mask1u.ptr<byte>(r);
		const int* idx = _HistBinIdx1i.ptr<int>(r);
		for (int c = 0; c < w; c++)
			fW[idx[c]] += m[c];
	}
	HistBinWf /= 4*(clrW1f - HistBinWf) + HistBinWf;

	//CmShow::HistBins(_HistBinClr3f, clrW1f, "Color Hist");
	CmShow::HistBins(_HistBinClr3f, HistBinWf, "Hist ratio");

	vector<Mat> pci1f;
	_gmm.GetProbs(_HistBinClr3f, pci1f);
	vecD salV1f(pci1f.size()), sum1f(pci1f.size());
	Mat histBinSal1f = Mat::zeros(1, _binNum, CV_32F);
	for (size_t i = 0; i < pci1f.size(); i++){
		float *clrW = (float*)pci1f[i].data; 
		for (int j = 0; j < _binNum; j++){
			salV1f[i] += fW[j] * clrW[j];
			sum1f[i] += clrW[j];
		}
		salV1f[i] /= sum1f[i];
		histBinSal1f += pci1f[i] * salV1f[i];
		
		//CmShow::HistBins(_HistBinClr3f, pci1f[i], format("Hist Sal %d:%g", i, salV1f[i]));
	}
	normalize(histBinSal1f, histBinSal1f, 0, 1, CV_MINMAX);
	//CmShow::HistBins(_HistBinClr3f, histBinSal1f, "Hist Sal");


	tm.StopAndReport();


	float* binSal = (float*)histBinSal1f.data;
	////CmShow::HistBins(_HistBinClr3f, histBinSal1f, "Histogram bin saliency", false);


	Mat sal1f = Mat::zeros(img.size(), CV_32F);
	Mat _sal1f = sal1f(wkRect);
	for (int r = 0; r < h; r++){
		const int* idx = _HistBinIdx1i.ptr<int>(r);
		float* salV = _sal1f.ptr<float>(r);
		for (int c = 0; c < w; c++)
			salV[c] = binSal[idx[c]];
	}


	imshow("Saliency", sal1f);
	//Mat show3u;
	//vecM chns;
	//split(img, chns);
	//sal1f.convertTo(chns[2], CV_8U, 255);
	//merge(chns, show3u);

	rectangle(img, _rect, Scalar(0, 0, 255));
	rectangle(img, wkRect, Scalar(0, 255, 255));
	imshow("Image", img);
	//imshow("Illustrate", show3u);
	waitKey(0);
	return Mat();
}