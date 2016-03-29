#include "stdafx.h"
#include "GrabCutMF.h"

// 3, 30, 20, 10
GrabCutMF::GrabCutMF(CMat &img3f, CMat &img3u, CStr &nameNE, float w1, float w2, float w3, float alpha, float beta, float gama, float mu)
	:_fGMM(5), _bGMM(5), _w(img3f.cols), _h(img3f.rows)
	,_crf(_w, _h, 2), _nameNE(nameNE)
{
	CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
	_imgBGR3f = img3f;
	_img3u = img3u;
	_trimap1i.create(_h, _w, CV_32S);
	_segVal1f.create(_h, _w, CV_32F);
	_unary2f.create(_h, _w, CV_32FC2);
	_res1u.create(_h, _w, CV_8U);

	if (w1 != 0)
		_crf.addPairwiseBilateral(alpha, alpha, beta, beta, beta, img3u.data, w1);
	if (w2 != 0)
		_crf.addPairwiseGaussian(gama, gama, w2);
	if (w3 != 0)
		_crf.addPairwiseColorGaussian(mu, mu, mu, img3u.data, w3);
}

// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper 
Mat GrabCutMF::initialize(const Rect &rect, CMat &bdMask, float backW, bool illustrate)
{
	_trimap1i = TrimapUnknown;
	_trimap1i.setTo(UserBack, bdMask);
	_segVal1f = Mat::ones(_img3u.size(), CV_32F);
	_segVal1f.setTo(0, bdMask);

	_bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, 1 - _segVal1f);
	_fGMM.BuildGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);

	//Mat histIdx1i, histClr3f, histNum1i;
	//int binNum = CmColorQua::D_Quantize(_imgBGR3f(rect), histIdx1i, histClr3f, histNum1i);
	//Vec3f *histClr = (Vec3f*)histClr3f.data;
	//int *histNum = (int*)histNum1i.data;
	//vector<CostfIdx> probIdx(binNum);
	//for (int i = 0; i < binNum; i++)
	//	probIdx[i] = make_pair(_bGMM.P(histClr[i]), i);
	//sort(probIdx.begin(), probIdx.end());
	//int maxNum = rect.width*rect.height*0.3333, crntNum = 0;
	//for (int i = 0; i < binNum; i++){
	//	int idx = probIdx[i].second;
	//	if (crntNum > maxNum)
	//		histNum[idx] = 0;
	//	crntNum += histNum[idx];
	//}
	//
	//histNum1i.convertTo(histNum1i, CV_32F);
	//_fGMM.BuildGMMs(histClr3f, _fGMMidx1i, histNum1i);
	

	_fGMM.RefineGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
	_bGMM.RefineGMMs(_imgBGR3f, _bGMMidx1i, 1 - _segVal1f);

	if (illustrate){
		//printf("Illustrate: %s\n", _nameNE.c_str());
		//_fGMM.iluProbsWN(_imgBGR3f, _nameNE + "_F");
		//_bGMM.iluProbsWN(_imgBGR3f, _nameNE + "_B");
		illuProb(_imgBGR3f, _bGMM, _fGMM, _nameNE);
		Mat weights = Mat::zeros(1, 10, CV_32F);
		Mat color = Mat::zeros(1, 10, CV_32FC3);
		for (int i = 0; i < 5; i++){
			weights.at<float>(i) = (float)_fGMM.getWeight(i);
			if (i < _fGMM.K())
				color.at<Vec3f>(i) = _fGMM.getMean(i);
			weights.at<float>(i + 5) = (float)_bGMM.getWeight(i);
			if (i < _bGMM.K())
				color.at<Vec3f>(i + 5) = _bGMM.getMean(i);
		}
		CmShow::HistBins(Mat(color), Mat(weights), _nameNE + "_FB.png");
	}
	return fitGMMs(illustrate ? _nameNE + format("_P%g.png", backW) : "", backW);
}

void GrabCutMF::illuProb(CMat sampleDf, CmGMM &bGMM, CmGMM &fGMM, CStr &nameNE)
{
	vector<Mat> pciF, pciB;
	Mat pI = Mat::zeros(sampleDf.size(), CV_32F);
	fGMM.GetProbsWN(sampleDf, pciF);
	bGMM.GetProbsWN(sampleDf, pciB);
	int fK = fGMM.K(), bK = bGMM.K();
	assert(fGMM.maxK() == bGMM.maxK());
	for (int i = 0; i < fK; i++)
		add(pciF[i], pI, pI);
	for (int i = 0; i < bK; i++)
		add(pciB[i], pI, pI);

	Mat tmpShow;
	for (int i = 0; i < fK; i++){
		divide(pciF[i], pI, pciF[i]);
		imwrite(nameNE + format("_F%d.png", i), pciF[i]*255);
	}
	for (int i = 0; i < bK; i++){
		divide(pciB[i], pI, pciB[i]);
		imwrite(nameNE + format("_B%d.png", i), pciB[i]*255);
	}

	for (int i = fK; i < fGMM.maxK(); i++)
		CmFile::WriteNullFile(nameNE + format("_F%d.nul", i));
	for (int i = bK; i < bGMM.maxK(); i++)
		CmFile::WriteNullFile(nameNE + format("_B%d.nul", i));

}


Mat GrabCutMF::fitGMMs(CStr &saveName, float backW)
{
#pragma omp parallel for
	for (int y = 0; y < _h; y++){
		float* segV = _segVal1f.ptr<float>(y);
		Vec2f* unryV = _unary2f.ptr<Vec2f>(y);
		int* triV = _trimap1i.ptr<int>(y); 
		Vec3f* img = _imgBGR3f.ptr<Vec3f>(y);
		for (int x = 0; x < _w; x++){
			float prb; // User Back
			switch (triV[x]){
			case UserBack: prb = 0; break;
			case UserFore: prb = 1.f; break;
			default: 
				float foreP = _fGMM.P(img[x]), backP = _bGMM.P(img[x]);
				prb = 0.8f * foreP/(foreP + backW*backP + 1e-8f);
			}
			segV[x] = prb;
			unryV[x] = Vec2f(prb, 1-prb);
		}
	}
	if (saveName.size())
		imwrite(saveName, _segVal1f*(255/0.8));
	return _segVal1f;
}

Mat GrabCutMF::refine(int iter)
{
	// Initialize _unary1f using GMM
	_crf.setUnaryEnergy(_unary2f.ptr<float>(0));
	float* prob = _crf.binarySeg(iter, 1.f);
	float* res = (float*)_segVal1f.data;
	const int N = _w * _h;
	for(int i=0; i<N; i++, prob+=2)
		res[i] = prob[1]/(prob[0]+prob[1]+1e-20f);

	return _segVal1f;
}

Mat GrabCutMF::showMedialResults(CStr& title)
{
	_show3u.create(_h, _w, CV_8UC3);
	_imgBGR3f.convertTo(_show3u, CV_8U, 255);

	for (int y = 0; y < _h; y++){
		const int* triVal = _trimap1i.ptr<int>(y);
		const float* segVal = _segVal1f.ptr<float>(y);
		Vec3b* triD = _show3u.ptr<Vec3b>(y);
		for (int x = 0; x < _w; x++, triD++) {
			switch (triVal[x]){
				case UserFore: (*triD)[2] = 255; break; // Red
				case UserBack: (*triD)[1] = 255; break; // Green
			}
			if (segVal[x] < 0.5){
				(*triD)[0] = 255;
				if (x-1 >= 0 && segVal[x-1] > 0.5 || x+1 < _w && segVal[x+1] > 0.5)
					(*triD) = Vec3b(0,0,255);
				if (y-1 >= 0 && _segVal1f.at<float>(y-1, x) > 0.5 || y+1 < _h && _segVal1f.at<float>(y+1, x) > 0.5)
					_show3u.at<Vec3b>(y,x) = Vec3b(0,0,255);				
			}
		}
	}
	CmShow::SaveShow(_show3u, title);
	return _show3u;
}

void convexHullOfMask(CMat &mask1u, PointSeti &hullPnts)
{
	const int H = mask1u.rows - 1, W = mask1u.cols - 1;
	PointSeti pntSet;
	pntSet.reserve(H*W);
	for (int r = 1; r < H; r++){
		const byte* m = mask1u.ptr<byte>(r);
		for (int c = 1; c < W; c++){
			if (m[c] < 200)
				continue;
			if (m[c-1] < 200 || m[c + 1] < 200 || mask1u.at<byte>(r-1, c) < 200 || mask1u.at<byte>(r+1, c) < 200)
				pntSet.push_back(Point(c, r));
		}
	}
	convexHull(pntSet, hullPnts);
}

void GrabCutMF::getGrabMask(CMat edge1u, Mat &grabMask)
{
	//imshow("Edge map", edge1u);
	//imshow("Grabmask", grabMask);
	//waitKey(1);

	queue<Point> selectedPnts;
	int _w = edge1u.cols, _h = edge1u.rows;
	for (int y = 1, maxY = _h - 1, stepSz = edge1u.step.p[0]; y < maxY; y++) {
		byte* m = grabMask.ptr<byte>(y);
		for (int x = 0, maxX = _w - 1; x < maxX; x++)
			if (m[x] == 255 && (m[x - 1] == 0 || m[x + 1] == 0 || m[x - stepSz] == 0 || m[x + stepSz] == 0))
				selectedPnts.push(Point(x, y));
	}

	// Flood fill
	while (!selectedPnts.empty()){
		Point crntPnt = selectedPnts.front();
		grabMask.at<byte>(crntPnt) = 255;
		selectedPnts.pop();
		for (int i = 0; i < 4; i++){
			Point nbrPnt = crntPnt + DIRECTION4[i];
			if (CHK_IND(nbrPnt) && grabMask.at<byte>(nbrPnt) == 0 && edge1u.at<byte>(nbrPnt) == 0)
				grabMask.at<byte>(nbrPnt) = 255, selectedPnts.push(nbrPnt);
		}
	}


	//imshow("Grabmask New", grabMask);
	//waitKey(0);
}


Mat GrabCutMF::getGrabMask(CMat &img3u, Rect rect)
{
	// Initialize flood fill
	queue<Point> selectedPnts;
	const int _h = img3u.rows, _w = img3u.cols, BW = 5;
	{// If not connected to image border, expand selection border unless stopped by edges
		Point rowT(rect.x, rect.y), rowB(rect.x, rect.y + rect.height - 1);
		Point colL(rect.x, rect.y), colR(rect.x + rect.width - 1, rect.y);
		if (rect.x >= BW) // Expand left edge
			for (int y = 0; y < rect.height; y++, colL.y++) selectedPnts.push(colL);
		else
			rect.x = BW;
		if (rect.y >= BW) // Expand top edge
			for (int x = 0; x < rect.width; x++, rowT.x++)	selectedPnts.push(rowT);
		else
			rect.y = BW;
		if (rect.x + rect.width + BW <= _w) // Expand right edge	
			for (int y = 0; y < rect.height; y++, colR.y++) selectedPnts.push(colR);
		else
			rect.width = _w - rect.x - BW;
		if (rect.y + rect.height + BW <= _h) // Expand bottom edge
			for (int x = 0; x < rect.width; x++, rowB.x++) selectedPnts.push(rowB);
		else
			rect.height = _h - rect.y - BW;
	}

	Mat mask1u(img3u.size(), CV_8U);
	memset(mask1u.data, 255, mask1u.step.p[0] * mask1u.rows);
	mask1u(rect) = Scalar(0);

	Mat edge1u;
	CmCv::CannySimpleRGB(img3u, edge1u, 120, 1200, 5);
	dilate(edge1u, edge1u, Mat(), Point(-1, -1), 3);
	//rectangle(edge1u, rect, Scalar(128));
	//imwrite(sameNameNE + "_Selection.png", edge1u);

	// Flood fill
	while (!selectedPnts.empty()){
		Point crntPnt = selectedPnts.front();
		mask1u.at<byte>(crntPnt) = 255;
		selectedPnts.pop();
		for (int i = 0; i < 4; i++){
			Point nbrPnt = crntPnt + DIRECTION4[i];
			if (CHK_IND(nbrPnt) && mask1u.at<byte>(nbrPnt) == 0 && edge1u.at<byte>(nbrPnt) == 0)
				mask1u.at<byte>(nbrPnt) = 255, selectedPnts.push(nbrPnt);
		}
	}
	cv::Mat temp(mask1u(Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)));
	CmCv::rubustifyBorderMask(temp);
	return mask1u;
}




void GrabCutMF::Demo(CStr &wkDir, float w1, float w2, float w3, float alpha, float beta, float gama, float mu)
{	
	CStr imgDir = wkDir + "Imgs/", salDir = wkDir + "Sal4N/", iluDir = wkDir + "Ilu4N/";
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(imgDir + "*.jpg", namesNE);
	CmFile::MkDir(salDir);
	CmFile::MkDir(iluDir);
	printf("w1 = %g, w2 = %g, w3 = %g, alpha = %g, beta = %g, gama = %g, mu = %g\n", w1, w2, w3, alpha, beta, gama, mu);

	// Number of labels
	//const int M = 2;
	CmTimer tm("Time"), tmIni("TimeIni"), tmRef("TimeRef");
	double maxWeight = 2; // 2: 0.958119, 1: 0.953818, 
	tm.Start();
#pragma omp parallel for
	for (int i = 0; i < imgNum; i++){
		printf("Processing %d/%d: %s%s.jpg%20s\r\n", i, imgNum, _S(imgDir), _S(namesNE[i]), "");
		CmFile::Copy(imgDir + namesNE[i] + ".jpg", salDir + namesNE[i] + ".jpg");
		//CmFile::Copy(imgDir + namesNE[i] + ".png", salDir + namesNE[i] + "_GT.png");
		Mat _imMat3u = imread(imgDir + namesNE[i] + ".jpg"), imMat3f, imMat3u, gt1u;
		Mat _gt1u = imread(imgDir + namesNE[i] + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		if(_gt1u.rows == 0 && _gt1u.cols == 0) {
            cout<<"Error: unable to open "<<(imgDir + namesNE[i] + ".png")<<endl;
            continue;
		}
		blur(_gt1u, _gt1u, Size(3,3));
		Mat _res1u = Mat::zeros(_imMat3u.size(), CV_8U);
		Rect wkRect = CmCv::GetMaskRange(_gt1u, 30, 200);
		_imMat3u(wkRect).copyTo(imMat3u);
		_gt1u(wkRect).copyTo(gt1u);
		imMat3u.convertTo(imMat3f, CV_32FC3, 1/255.0);
		Rect rect = CmCv::GetMaskRange(gt1u, 5, 128);


		Mat edge1u; // Use an edge map to expand the background mask in flat (no edge) region
		CmCv::CannySimpleRGB(imMat3u, edge1u, 120, 1200, 5);
		dilate(edge1u, edge1u, Mat(), Point(-1, -1), 3);
		Mat borderMask1u(imMat3u.size(), CV_8U), tmpMask;
		memset(borderMask1u.data, 255, borderMask1u.step.p[0] * borderMask1u.rows);
		borderMask1u(rect) = Scalar(0);
		getGrabMask(edge1u, borderMask1u);


		//* The Mean field based GrabCut
		//tmIni.Start();
		GrabCutMF cutMF(imMat3f, imMat3u, salDir + namesNE[i], w1, w2, w3, alpha, beta, gama, mu);
		//Mat borderMask1u = CmCv::getGrabMask(imMat3u, rect), tmpMask;
		imwrite(salDir + namesNE[i] + "_BM.png", borderMask1u);
		imwrite(salDir + namesNE[i] + ".jpg", imMat3u);
		//imwrite(salDir + namesNE[i] + "_GT.png", gt1u);
		cutMF.initialize(rect, borderMask1u, (float)maxWeight, true);
		//cutMF.setGrabReg(rect, CmCv::getGrabMask(imMat3u, rect));
		//tmIni.Stop();
		//tmRef.Start();
		cutMF.refine();
		//tmRef.Stop();
		
		Mat res1u = cutMF.drawResult(), invRes1u;

		res1u.copyTo(_res1u(wkRect));
		imwrite(salDir + namesNE[i] + "_GCMF1.png", _res1u);

		//if (sum(res1u).val[0] < EPS){
		//	printf("%s.jpg don't contains a salient object\n", _S(namesNE[i]));
		//	continue;
		//}

		dilate(res1u(rect), tmpMask, Mat(), Point(-1, -1), 10);	
		bitwise_not(tmpMask, borderMask1u(rect));
		getGrabMask(edge1u, borderMask1u);

		//blur(res1u, invRes1u, Size(3, 3));
		//
		//PointSeti hullPnts;
		//convexHullOfMask(invRes1u, hullPnts);
		//fillConvexPoly(invRes1u, hullPnts, 255);
		//bitwise_not(invRes1u, invRes1u);
		//bitwise_or(invRes1u, borderMask1u, borderMask1u);
		imwrite(salDir + namesNE[i] + "_MB2.png", borderMask1u);

		
		//double w =  maxWeight - (maxWeight-1)*sum(res1u).val[0]/(borderMask1u.rows*borderMask1u.cols*255 - sum(borderMask1u).val[0]);
		cutMF.initialize(rect, borderMask1u, 2);
		cutMF.refine();

		//printf("weight = %g\n", w);
		//imshow("Result", res1u);
		//imshow("Possible", borderMask1u);
		//imshow("Image", imMat3f);
		//waitKey(0);

		res1u = cutMF.drawResult();

		Rect rectRes = CmCv::GetMaskRange(res1u, 5, 128);
		if (rectRes.width * 1.1 < rect.width || rectRes.height * 1.1 < rect.height){ // Too short result
			printf("%s.jpg contains a small object\n", _S(namesNE[i]));
			memset(borderMask1u.data, 255, borderMask1u.step.p[0] * borderMask1u.rows);
			borderMask1u(rect) = Scalar(0);
			cutMF.initialize(rect, borderMask1u, 2);
			cutMF.refine();
			res1u = cutMF.drawResult();
			imwrite(salDir + namesNE[i] + "_MB2.png", borderMask1u);
			CmFile::Copy2Dir(salDir + namesNE[i] + "*.*", iluDir);
			imwrite(iluDir + namesNE[i] + "_GCMF.png", _res1u);
		}

		res1u.copyTo(_res1u(wkRect));
		imwrite(salDir + namesNE[i] + "_GCMF.png", _res1u);
		
	}
	tm.Stop();
	double avgTime = tm.TimeInSeconds()/imgNum;
	printf("Speed: %gs, %gfps\t\t\n", avgTime, 1/avgTime);
	//tmIni.Report();
	//tmRef.Report();
	//CmEvaluation::EvalueMask(imgDir + "*.png", salDir, ".png", "_GC.png");

	
	char* pDes[] = { "GCMF1", "GCMF"}; //, "CudaG4", "Onecut", "GC", "CudaH", 
	vecS des  = charPointers2StrVec (pDes);
	CStr rootDir = CmFile::GetFatherFolder(wkDir), dbName = CmFile::GetNameNE(wkDir.substr(0, wkDir.size() - 1));
	CmEvaluation::EvalueMask(imgDir + "*.png", salDir, des, wkDir.substr(0, wkDir.size() - 1) + "Res.m", 0.3, false, "", dbName);
}

Mat GrabCutMF::drawResult()
{
	compare(_segVal1f, 0.5, _res1u, CMP_GT); 
	return _res1u;
}

void GrabCutMF::runGrabCutOpenCV(CStr &wkDir)
{
	CStr imgDir = wkDir + "Imgs/", salDir = wkDir + "Sal/";
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(imgDir + "*.jpg", namesNE);
	CmFile::MkDir(salDir);

	// Number of labels
	CmTimer tm("Time");
	tm.Start();
	for (int i = 0; i < imgNum; i++){
		printf("Processing %d/%d: %s.jpg%20s\r\n", i, imgNum, _S(namesNE[i]), "");
		CmFile::Copy(imgDir + namesNE[i] + ".jpg", salDir + namesNE[i] + ".jpg");
		CmFile::Copy(imgDir + namesNE[i] + ".png", salDir + namesNE[i] + "_GT.png");
		Mat imMat3u = imread(imgDir + namesNE[i] + ".jpg");
		Mat gt1u = imread(imgDir + namesNE[i] + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		//imwrite(imgDir + namesNE[i] + ".bmp", gt1u);
		blur(gt1u, gt1u, Size(3,3));
		Rect wkRect = CmCv::GetMaskRange(gt1u, 1, 128);

		// Prepare data for OneCut
		//Mat rectImg = Mat::ones(gt1u.size(), CV_8U)*255;
		//rectImg(wkRect) = 0;
		//imwrite(salDir + namesNE[i] + ".bmp", imMat3u);
		//imwrite(salDir + namesNE[i] + "_t.bmp", rectImg);

		Mat res1u, bgModel, fgModel;
		grabCut(imMat3u, res1u, wkRect, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
		grabCut(imMat3u, res1u, wkRect, bgModel, fgModel, 5);
		compare(res1u, GC_PR_FGD, res1u, CMP_EQ);
		imwrite(salDir + namesNE[i] + "_GC.png", res1u);
	}
	tm.Stop();
	double avgTime = tm.TimeInSeconds()/imgNum;
	printf("Speed: %gs, %gfps\t\t\n", avgTime, 1/avgTime);

	CmEvaluation::EvalueMask(imgDir + "*.png", salDir, "GC", "");
}


//Mat GrabCutMF::setGrabReg(const Rect &rect, CMat &bordMask1u)
//{
//	CmGrabSal sGC(_imgBGR3f, bordMask1u, _nameNE);
//	sGC.HistgramGMMs();
//	_segVal1f = sGC.GetSaliencyCues();
//
//	_trimap1i = UserBack;
//	_trimap1i(rect) = TrimapUnknown;
//
//#pragma omp parallel for
//	for (int y = 0; y < _h; y++){
//		float* segV = _segVal1f.ptr<float>(y);
//		Vec2f* unryV = _unary2f.ptr<Vec2f>(y);
//		int* triV = _trimap1i.ptr<int>(y); 
//		Vec3f* img = _imgBGR3f.ptr<Vec3f>(y);
//		for (int x = 0; x < _w; x++){
//			float prb; // User Back
//			switch (triV[x]){
//			case UserBack: prb = 0; break;
//				//case UserFore: prb = 0.9f; break;
//			default: prb = 0.8f*segV[x];
//			}
//			unryV[x] = Vec2f(prb, 1-prb);
//			segV[x] = prb;
//		}
//	}
//	if (_nameNE.size())
//		imwrite(_nameNE + "_P.png", _segVal1f*255);
//
//	return _segVal1f;
//}
