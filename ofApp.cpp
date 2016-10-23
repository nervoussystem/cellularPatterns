#include "ofApp.h"
#include "constants.h"
#include "clipper.hpp"


#include<CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include<CGAL/Polygon_2.h>
#include<CGAL/create_straight_skeleton_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Voronoi_diagram_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2                   Point_2;
typedef CGAL::Polygon_2<K>           Polygon_2;
typedef CGAL::Straight_skeleton_2<K> Ss;

typedef CGAL::Delaunay_triangulation_2<K>                                    DT;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT>                 AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
typedef CGAL::Voronoi_diagram_2<DT, AT, AP>                                    VD;

typedef VD::Vertex_handle             Vertex_handle;

using namespace ClipperLib;
using namespace ofxCv;
using namespace cv;

float w = 750; //750
float h = 1050; //1000

float maxDensity(220);//200 90 //150 810
float minDensity(180);//18 //30  200

float maxDensity2(50);
float minDensity2(8.5);
float anisotrophyStr(0.6f);

float etchOffset = 2.85;
bool doSmooth = true;
float filletPercent = .4;
float sizeFallOffExp = 0.38;// 1.57;// .5;// .75;
float anisoLerpRamp = .5;
float rando = .5;

bool paused = true;
bool cleanEdge = false;
bool drawFill = true;
//for metal jewelry
float minThick = 2.0f;
float maxThick = 24;// 19.8;// minThick * 3.0f;
//for rubber 
//float minThick = 5.0f; //.05 inches rubber
//float maxThick = 9.9f;//minThick*2.0f; //.1 inches rubber
//for fabric
//float minThick = 6.0f;
//float maxThick = 10.0f;
float offsetPercent = 0.17f;

String imageName = "circle.png";
ofImage gradient;
//"circle25.4mm.png";
//"circle12.7mm.png";
//"circle40mm.png";

int binW, binH, binD, binWH;
vector< vector<int> > bins;
bool isOptimizing = false;
bool record = false;
bool hasMask = true; //if you are using an image to crop the pattern

vector<ofVec2f> patternPts;
vector<float> patternPtRads;
float drawOffsetX = 300;
vector<AnisoPoint2f> nearPts;
AnisoPoint2f(*getAnisoPoint)(const ofVec3f & pt);
vector<AnisoPoint2f(*)(const ofVec3f & pt) > anisoFunctions;
vector<string> functionNames;

int totalPts = 1100;
bool doEtchOffset = false;
float maxImgDist = 0;
Mat imgDist, imgMask;
Mat imgGradX, imgGradY;

int boundaryIndex = 0;
ofFbo fbo;
bool doAnim = true;

//--------------------------------------------------------------
void ofApp::setup(){
	ofSeedRandom(ofGetSystemTimeMicros());
	baseImage.load(imageName);
	setupImage();	
	
	//important thing
	//anisotropy function - give it a pt in space and it returns an anisotropic pt
	//getAnisoPtImg - uses image
	//getAnisoPtEdge - edge of the screen
	//getAnisoPtNoise
	//getAnisoPt - distance from a single Pt
	getAnisoPoint = &getAnisoPtNoise;// &getAnisoEdge;
	anisoFunctions.push_back(&getAnisoEdge);
	anisoFunctions.push_back(&getAnisoPt);
	anisoFunctions.push_back(&getAnisoPtSet);
	anisoFunctions.push_back(&getAnisoPtNoise);
	anisoFunctions.push_back(&getAnisoPointPts);
	anisoFunctions.push_back(&getAnisoPtImg);
	anisoFunctions.push_back(&getAnisoPtSin);
	anisoFunctions.push_back(&getAnisoPtBamboo);
	anisoFunctions.push_back(&getAnisoSide);
	functionNames.push_back("border");
	functionNames.push_back("pt");
	functionNames.push_back("ptSet");
	functionNames.push_back("noise");
	functionNames.push_back("secondStage");
	functionNames.push_back("img");
	functionNames.push_back("sin");
	functionNames.push_back("bamboo");
	functionNames.push_back("side");
	gradient.load("gradient3.jpg");
	//minDensity = minDensity2;
	//maxDensity = maxDensity2;
	setupGui();

	reset();

	//optThread.pts = pts;
	//optThread.w = w;
	//optThread.h = h;
	//optThread.minDensity = minDensity;
	//optThread.maxDensity = maxDensity;
	//optThread.initCcvt();
	//optThread.startThread(true, true);
	//optThread.waitForThread();
	//pts = optThread.pts;

	getDistances();
	dualContour();
	offsetCells();

	fbo.allocate(baseImage.getWidth(), baseImage.getHeight(),6408,4);
}


void ofApp::reset() {
	
	binW = floor(w / max(minDensity,maxDensity)) + 1;
	binH = floor(h / max(minDensity,maxDensity)) + 1;
	
	linesMesh.setMode(OF_PRIMITIVE_LINES);
	bins.resize(binW*binH);
	for (int i = 0; i < bins.size(); ++i)bins[i].clear();
	initPts();
	optThread.pts = pts;
	optThread.w = w;
	optThread.h = h;
	optThread.minDensity = minDensity;
	optThread.maxDensity = maxDensity;
	//optThread.startThread(true, true);
	//optThread.waitForThread();
	optThread.initCcvt();
	//optimize automatically
	for (int i = 0; i < 20;++i) optThread.ccvtStep();
	pts = optThread.pts;
	getDistances();
	dualContour();
	offsetCells();
}

void ofApp::setupImage() {
	
	w = baseImage.getWidth();
	h = baseImage.getHeight();
	ofSetWindowShape(w+300, h);

	Mat initImg(baseImage.getHeight(), baseImage.getWidth(), CV_8UC1);
	cvtColor(toCv(baseImage), initImg, COLOR_BGR2GRAY);
	threshold(initImg, 120);
	dilate(initImg, imgMask, getStructuringElement(MORPH_CROSS, Size(7, 7), cv::Point(3, 3)));
	//toCv(baseImage).copyTo(initImg);
	//imgGradX = Mat(baseImage.getHeight(), baseImage.getWidth(), CV_32FC1);
	//imgGradY = Mat(baseImage.getHeight(), baseImage.getWidth(), CV_32FC1);
	distanceTransform(initImg, imgDist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	double maxDist, minDist;
	minMaxLoc(imgDist,&minDist,&maxDist);
	maxImgDist = maxDist;
	normalize(imgDist, imgDist);
	Mat tempImg1 = imgDist;
	Mat tempImg2 = imgDist;
	Scharr(tempImg1, imgGradX, CV_32F, 1, 0);
	Scharr(tempImg2, imgGradY, CV_32F, 0, 1);
	toOf(imgDist, distImage);
}

void ofApp::setupGui() {
	gui = new ofxDatGui();
	gui->onButtonEvent(this, &ofApp::buttonEvent);


	ofxDatGuiSlider * slider = gui->addSlider("min density", 2, 600, minDensity);
	slider->bind(minDensity,2,600);
	slider = gui->addSlider("max density", 2, 600, maxDensity);
	slider->bind(maxDensity,2,600);
	slider = gui->addSlider("anisotropy", .5, 2, anisotrophyStr);
	slider->bind(anisotrophyStr,0.5,2);
	slider = gui->addSlider("min thickness", 0, 20, minThick);
	slider->bind(minThick,0,40);
	slider = gui->addSlider("max thickness", 0, 60, maxThick);
	slider->bind(maxThick,0,60);
	slider = gui->addSlider("offset", 0.1, .5, offsetPercent);
	slider->bind(offsetPercent);

	slider = gui->addSlider("anisotropy lerp ramp", 0,1, anisoLerpRamp);
	slider->bind(anisoLerpRamp, 0, 5);

	slider = gui->addSlider("size lerp ramp", 0, 1, sizeFallOffExp);
	slider->bind(sizeFallOffExp, 0, 5);


	slider = gui->addSlider("fillet percent", .25, .99);
	slider->bind(filletPercent, .25, .99);

	gui->addToggle("smoothing", doSmooth);
	gui->addToggle("cleanEdge", cleanEdge);
	ofxDatGuiDropdown * functionDd = gui->addDropdown("functions", functionNames);
	functionDd->onDropdownEvent(this, &ofApp::setFunction);
	gui->addButton("reset");
	gui->addButton("optimize");
	gui->addButton("setupStage2");
	gui->addButton("savePDF");
	gui->addButton("clear points");
	gui->addButton("randomize");
}

void ofApp::initPts() {
	pts.clear();
	int fail = 0;
	ofVec3f pt;
	int totalTries = 0;
	//density = ofLerp(maxDensity, minDensity, ofClamp(x/200.0,0,1));
	
	while(fail < 5000) {
		pt = ofVec3f(ofRandom(w), ofRandom(h));
		//check mask
		if (!hasMask || imgDist.at<float>(min((int)pt.y, imgDist.rows-1), min((int)pt.x,imgDist.cols-1)) > 0) {
			//density = ofLerp(maxDensity, minDensity, ofClamp(x/200.0,0,1));
			if (addPt(pt)) {
				fail = 0;
				cout << pts.size() << endl;
			}
		}
		if (pts.size() > 6) {
			if(doAnim) break;
		}
		fail++;
	}
}

void ofApp::progressAnimation() {
	int numPtsToAdd = ofLerp(1, 10, pts.size()*1.0 / totalPts);
	bool doSplit = false;
	float splitProb = .02;
	ofVec3f pt;
	if (pts.size() < totalPts) {
		int numPts = pts.size();
		for (int i = 0; i < numPts; ++i) {
			if (ofRandom(1.0) < splitProb) {
				doSplit = true;
				auto & pt = pts[i];
				pts.push_back(getAnisoPoint(ofVec3f(pt[0] + ofRandom(-1, 1), pt[1] + ofRandom(-1, 1))));
			}
		}
	}
	/*
	for (int j = 0; j < numPtsToAdd; ++j) {
		while (true) {
			pt = ofVec3f(ofRandom(w), ofRandom(h));
			//check mask
			if (!hasMask || imgDist.at<float>(min((int)pt.y, imgDist.rows - 1), min((int)pt.x, imgDist.cols - 1)) > 0) {
				//density = ofLerp(maxDensity, minDensity, ofClamp(x/200.0,0,1));
				if (addPt(pt)) {
					break;
				}
			}
		}
	}
	*/

	
	optThread.w = w;
	optThread.h = h;
	optThread.minDensity = minDensity;
	optThread.maxDensity = maxDensity;
	//optThread.startThread(true, true);
	//optThread.waitForThread();
	if (doSplit) {
		optThread.pts = pts;
		optThread.initCcvt();
		cout << pts.size() << endl;
	}

	optThread.ccvtStep();
	//pts = optThread.pts;

	Site<MyPoint>::Vector & sites = optThread.optimizer.sites();
	//Site<MyPoint>::Vector::const_iterator it = sites.begin();
	for (int i = 0; i < pts.size(); ++i) {
		ofVec3f cPt(pts[i][0], pts[i][1]);
		ofVec3f dPt(optThread.pts[i][0], optThread.pts[i][1]);
		pts[i] = getAnisoPoint((cPt*0.6 + dPt*0.4));
		sites[i].location = (MyPoint) pts[i];

	}

	getDistances();
	dualContour();
	offsetCells();

}

bool ofApp::addPt(ofVec3f & pt) {
	MyPoint aniPt = getAnisoPoint(pt);

	float binSize = max(maxDensity, minDensity);
	int sx = (int)(pt.x / binSize);
	int sy = (int)(pt.y / binSize);
	int sz = (int)(pt.z / binSize);
	int minX = max<int>(sx - 1, 0);
	int maxX = min<int>(sx + 1, binW - 1);
	int minY = max<int>(sy - 1, 0);
	int maxY = min<int>(sy + 1, binH - 1);
	int minZ = max<int>(sz - 1, 0);
	int maxZ = min<int>(sz + 1, binD - 1);
	//density = ofLerp(minDensity, maxDensity, ofClamp(1.0-abs(x)/133.0,0,1));

	for (int i = minX; i <= maxX; i++) {
		for (int j = minY; j <= maxY; j++) {
			vector<int> &bin = bins[j*binW + i];

			for (unsigned int index = 0; index < bin.size(); index++) {
				int ind = bin[index];
				MyPoint pt2 = pts[ind];
				float d = metric.distance_square(*aniPt.pt, *pt2.pt, *aniPt.jacobian);
				if (d < 1) return false;
				//d = metric.distance_square(pt2, aniPt);
				//if (d < 1) return false;
			}
		}
	}

	pts.push_back(aniPt);
	vector<int> &bin = bins[sy*binW + sx];
	bin.push_back(pts.size() - 1);
	return true;
}

//--------------------------------------------------------------
void ofApp::update() {
	if (!paused) {
		progressAnimation();
		if (isOptimizing) {
			if (!optThread.isThreadRunning()) {
				isOptimizing = false;
				pts = optThread.pts;
				getDistances();
				dualContour();
				offsetCells();
				cout << "done" << endl;
			}
			else if (ofGetFrameNum() % 20 == 0) {
				optThread.lock();
				pts = optThread.pts;
				optThread.unlock();
				getDistances();
				dualContour();
				offsetCells();
			}
		}
		
	}
}

void ofApp::setupStage2() {
	nearPts = pts;
	getAnisoPoint = &getAnisoPointPts;
	minDensity  = minDensity2;
	maxDensity  = maxDensity2;
	anisotrophyStr = .75;// 1.0f / 1.4f;
	
	for (auto & pt : pts) {
		ofVec3f v(pt[0], pt[1]);
		v.x += ofRandom(-1, 1);
		v.y += ofRandom(-1, 1);
		pt = getAnisoPoint(v);
	}

	optThread.setup(pts);
	optThread.initCcvt();

	if(!doAnim)reset();
}

//--------------------------------------------------------------
void ofApp::draw(){
	if(!paused && doAnim) fbo.begin();
	ofBackground(255);
	ofSetColor(255);

	std::ostringstream ss;
	ss << "voronoi_dir" << anisotrophyStr << "_cellSz_" << minDensity << "-" << maxDensity << "_" << "_thick_" << minThick << "-" << maxThick << "_" << ofGetTimestampString() << ".pdf";

	ofPushMatrix();
	if (!(!paused && doAnim)) ofTranslate(drawOffsetX, 0);
	if (record) ofBeginSaveScreenAsPDF(ss.str());

	/*
	const auto & entries = optThread.optimizer.entries();
	auto it = entries.begin();
	int ptNum = 0;
	for (; it != entries.end();++it) {
		int numPts = it->points.size();
		ofFill();
		float ci = ofNoise(ptNum*2.38912);
		ofColor c = gradient.getColor(ofRandom(gradient.getWidth()), ofRandom(gradient.getHeight()));
		ofSetColor(c);
		for (int i = 0; i < numPts; ++i) {
			ofDrawCircle(it->points[i][0], it->points[i][1], 1);
		}
		ptNum++;
	}
	*/
	
	
	
	//drawPtEllipses();
	//distImage.draw(0,0);
	//baseImage.draw(0,0);
	ofSetColor(255);
	//linesMesh.draw();
	//if (record) {
	/*
	for (int i = 0; i < linesMesh.getNumIndices();i+=2) {
		ofVec2f pt = (linesMesh.getVertex(linesMesh.getIndex(i))+ linesMesh.getVertex(linesMesh.getIndex(i + 1)))*.5;
		int cellIndex1 = distances[(w*int(pt.y) + int(pt.x)) * 3].index;
		int cellIndex2 = distances[(w*int(pt.y) + int(pt.x)) * 3 + 1].index;
		int numCell = 0;
		float weight = 0;
		if (cellIndex1 < pts.size()) {
			AnisoPoint2f cellPt = pts[cellIndex1];
			weight += 1 / sqrt(cellPt.jacobian->determinant());
			numCell++;
		}
		if (cellIndex2 < pts.size()) {
			AnisoPoint2f cellPt = pts[cellIndex2];
			weight += 1 / sqrt(cellPt.jacobian->determinant());
			numCell++;
		}
		//stroke
		//get 2 closest cell pts and use their area  (determinant of their jacobian of the size)
		weight *= .35 / numCell;

		//clamp in between min and max stroke weight
		weight = ofClamp(weight, minThick, maxThick);

		ofSetLineWidth(weight);
		ofDrawLine(linesMesh.getVertex(linesMesh.getIndex(i)), linesMesh.getVertex(linesMesh.getIndex(i+1)));		
	}*/

	
	int endIndex = cellOffsets.size();
	if (drawFill) {
		ofFill();
		endIndex = boundaryIndex;
		ofSetColor(0);
		for (int i = endIndex; i < cellOffsets.size(); ++i) {
			auto & cell = cellOffsets[i];
			if (cell.size() > 3) {
				ofBeginShape();
				for (auto & pt : cell) {
					ofVertex(pt);
				}
				ofEndShape(true);
			}
		}
		ofSetColor(255);
	}
	else {
		ofNoFill();
	}
	for (int i = 0; i < endIndex;++i) {
		auto & cell = cellOffsets[i];
		if (cell.size() > 3) {
			ofBeginShape();
			for (auto & pt : cell) {
				ofVertex(pt);
			}
			ofEndShape(true);
		}
	}
	
	if (!paused && doAnim) {
		fbo.end();
		ofImage im;
		fbo.readToPixels(im.getPixels());
		stringstream ss;
		ss << "frame_" << ofGetFrameNum() << ".png";
		im.save(ss.str());
	}

	ofFill();
	ofSetColor(0);
	for (auto & p : pts) {
		ofCircle(p[0], p[1], 2);
	}
	
	//draw voronax
	/*
	ofNoFill();
	ofSetColor(255, 0, 0);
	VD vd;
	for (int i = 0; i < pts.size(); ++i) {
		vd.insert(Point_2(pts[i][0],pts[i][1]));
	}
	for (auto it = vd.edges_begin(); it != vd.edges_end(); ++it) {
		if (it->has_target() && it->has_source()) {
			Point_2 p1 = it->source()->point();
			Point_2 p2 = it->target()->point();
			ofLine(p1.x(), p1.y(), p2.x(), p2.y());
		}
	}
	*/
	
	if (record) {
		record = false;
		ofEndSaveScreenAsPDF();
	}
	ofPopMatrix();
}

void ofApp::drawPtEllipses() {
	ofMatrix4x4 mat;
	mat.makeIdentityMatrix();
	ofSetColor(0);
	ofNoFill();
	for (auto & pt : pts) {
		ofPushMatrix();
		
		ofTranslate(pt[0],pt[1]);

		Matrix2f transform = pt.jacobian->inverse();
		
		mat(0, 0) = transform(0, 0)*0.5;
		mat(1, 0) = transform(0, 1)*0.5;
		mat(1, 1) = transform(1, 1)*0.5;
		mat(0, 1) = transform(1, 0)*0.5;
		
		ofMultMatrix(mat);
		
		ofDrawCircle(0, 0, 0, 1);
		ofPopMatrix();
	}
}

void ofApp::getDistance(int i) {
	Vector2f tempPt;
	AnisoPoint2f & pt = pts[i];
	vector<bool> visited(w*h, false);
	int xi = (int)pt[0];
	int yi = (int)pt[1];
	Matrix2f transform = pt.jacobian->inverse();
	Vector2f vec(0, 1);
	vec = transform*vec;
	Vector2f vec2(1, 0);
	vec2 = transform*vec2;
	vec(0) = max(abs(vec(0)), abs(vec2(0)));
	vec(1) = max(abs(vec(1)), abs(vec2(1)));
	list<int> indexStack;

	float distMult = 2.5;
	distMult = ofLerp(20, distMult, ofClamp(pts.size() / 300.0, 0, 1));
	int minX = max(0, (int)(pt[0] - vec(0) * distMult));
	int maxX = min((int)w - 1, (int)(pt[0] + vec(0) * distMult));
	int minY = max(0, (int)(pt[1] - vec(1) * distMult));
	int maxY = min((int)h - 1, (int)(pt[1] + vec(1) * distMult));

	indexStack.push_back(yi*w + xi);
	visited[yi*w + xi] = true;
	while (!indexStack.empty()) {
		int index = indexStack.front();
		indexStack.pop_front();

		int x = index%(int)w;
		int y = index / (int)w;
		int index3 = index * 3;
		tempPt[0] = x;
		tempPt[1] = y;
		float dist = metric.distance_square(*pt.pt, tempPt, *pt.jacobian);
		if (dist < distances[index3].dist) {
			distances[index3 + 2] = distances[index3 + 1];
			distances[index3 + 1] = distances[index3];
			distances[index3].index = i;
			distances[index3].dist = dist;
		}
		else if (dist < distances[index3 + 1].dist) {
			distances[index3 + 2] = distances[index3 + 1];
			distances[index3 + 1].index = i;
			distances[index3 + 1].dist = dist;
		}
		else if (dist < distances[index3 + 2].dist) {
			distances[index3 + 2].index = i;
			distances[index3 + 2].dist = dist;
		}
		if (imgMask.at<float>(y, x) >0) {
			if (x + 1 <= maxX) {
				index = y*w + x + 1;
				if (!visited[index]) {
					visited[index] = true;
					indexStack.push_back(index);
				}
			}
			if (x - 1 >= minX) {
				index = y*w + x - 1;
				if (!visited[index]) {
					visited[index] = true;
					indexStack.push_back(index);
				}
			}
			if (y + 1 <= maxY) {
				index = (y + 1)*w + x;
				if (!visited[index]) {
					visited[index] = true;
					indexStack.push_back(index);
				}
			}
			if (y - 1 >= minY) {
				index = (y - 1)*w + x;
				if (!visited[index]) {
					visited[index] = true;
					indexStack.push_back(index);
				}
			}
		}
	}
	for (int x = minX; x <= maxX; ++x) {
		for (int y = minY; y <= maxY; ++y) {
			unsigned int index = (w*y + x) * 3;
			tempPt[0] = x;
			tempPt[1] = y;
			float dist = metric.distance_square(*pt.pt, tempPt, *pt.jacobian);
			if (dist < distances[index].dist) {
				distances[index + 2] = distances[index + 1];
				distances[index + 1] = distances[index];
				distances[index].index = i;
				distances[index].dist = dist;
			}
			else if (dist < distances[index + 1].dist) {
				distances[index + 2] = distances[index + 1];
				distances[index + 1].index = i;
				distances[index + 1].dist = dist;
			}
			else if (dist < distances[index + 2].dist) {
				distances[index + 2].index = i;
				distances[index + 2].dist = dist;
			}



		}
	}
}

void ofApp::getDistances() {
	distances.resize(w*h * 3);
	for (int i = 0; i < distances.size(); ++i) distances[i] = IndexDist(pts.size(), 9e9);
	if (hasMask) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (imgDist.at<float>(y, x) == 0 || !cleanEdge) {
					
					float weirdEdgeMultiplier = 10; //the smaller it is it more frilly the edge is
					distances[(w*y + x) * 3] = IndexDist(pts.size(),imgDist.at<float>(y, x)*maxImgDist/(maxDensity+minDensity)*weirdEdgeMultiplier);// IndexDist(pts.size(), 0);
				}
			}
		}
	}
	Vector2f tempPt;
	for (int i = 0; i < pts.size();++i) {
		getDistance(i);
	}
	if (hasMask) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (imgDist.at<float>(y, x) == 0) {
					if(distances[(w*y + x) * 3 + 1].index != pts.size())
						distances[(w*y + x) * 3].dist = distances[(w*y + x) * 3+1 ].dist*.95;
				}
			}
		}
	}
}

ofVec2f getVoronoiIntersection(ofVec2f p1, ofVec2f p2, float side1A, float side1B, float side2A, float side2B) {
	//float eLen = p1.distance(p2);
	float x = (side1A - side2A) / (side1A - side1B + side2B - side2A);
	if ((side1A - side1B + side2B - side2A) == 0) {
		cout << "WGFTESIDF " << side1A << " " << side1B << " " << side2A << endl;
		x = .5;
	}
	x = ofClamp(x, 0, 1);
	return ofVec2f(p1.x + (p2.x - p1.x)*x, p1.y + (p2.y - p1.y)*x);
}

void ofApp::dualContour() {
	linesMesh.clear();
	vector<ofVec2f> edgePts(w*h * 2);
	vector<int> ptIndices(w*h,-1);
	for (int y = 0; y < h - 1; ++y) {
		int wy = w*y;
		for (int x = 0; x < w - 1; ++x) {
			IndexDist p1 = distances[(wy + x)*3];
			IndexDist p2 = distances[(wy + x + 1)*3];
			IndexDist p3 = distances[(wy + x + w)*3];
			if (p1.index != p2.index) {
				edgePts[(wy + x) * 2] = ofVec3f(x+0.5,y);
			}
			if (p1.index != p3.index) {
				edgePts[(wy + x) * 2+1] = ofVec3f(x,y+0.5);
			}
		}
	}
	//NOT PROPERLY DOING BOTTOM AND RIGHT EDGE
	vector<vector<int> > neighbors;
	vector<set<int> > ptCells;
	vector<list<pair<int, int> > > cellEdges(pts.size()+1);
	for (int y = 0; y < h - 2; ++y) {
		int wy = w*y;
		for (int x = 0; x < w - 2; ++x) {
			IndexDist p1 = distances[(wy + x) * 3];
			IndexDist p2 = distances[(wy + x + 1) * 3];
			IndexDist p3 = distances[(wy + x + w) * 3];
			IndexDist p4 = distances[(wy + x + w+1) * 3];
			IndexDist p1b = distances[(wy + x) * 3+1];
			IndexDist p2b = distances[(wy + x + 1) * 3+1];
			IndexDist p3b = distances[(wy + x + w) * 3+1];
			IndexDist p4b = distances[(wy + x + w + 1) * 3+1];
			int numInts = 0;
			ofVec3f center;
			bool conLeft = false;
			bool conUp = false;
			if (p1.index != p2.index) {
				//center += edgePts[(wy + x) * 2];
				center += getVoronoiIntersection(ofVec2f(x, y), ofVec2f(x + 1, y), p1.dist, p2b.dist, p1b.dist, p2.dist);
				numInts++;
				conUp = true;
			}
			if (p1.index != p3.index) {
				//center += edgePts[(wy + x) * 2+1];
				center += getVoronoiIntersection(ofVec2f(x, y), ofVec2f(x, y+1), p1.dist, p3b.dist, p1b.dist, p3.dist);
				numInts++;
				conLeft = true;
			}
			if (p2.index != p4.index) {
				//center += edgePts[(wy + x+1) * 2+1];
				center += getVoronoiIntersection(ofVec2f(x+1, y), ofVec2f(x+1, y + 1), p2.dist, p4b.dist, p2b.dist, p4.dist);
				numInts++;
			}
			if (p3.index != p4.index) {
				//center += edgePts[(wy + x+w) * 2];
				center += getVoronoiIntersection(ofVec2f(x, y+1), ofVec2f(x+1, y + 1), p3.dist, p4b.dist, p3b.dist, p4.dist);
				numInts++;
			}
			if (numInts > 0) {
				center /= numInts;
				linesMesh.addVertex(center);
				neighbors.push_back(vector<int>());
				int currIndex = linesMesh.getNumVertices() - 1;
				set<int> ptCell;
				ptCell.emplace(p1.index);
				ptCell.emplace(p2.index);
				ptCell.emplace(p3.index);
				ptCell.emplace(p4.index);
				ptCells.push_back(ptCell);
				ptIndices[wy + x] = currIndex;
				if (x > 0 && conLeft) {
					int nIndex = ptIndices[wy + x - 1];
					cellEdges[p1.index].push_back(make_pair(nIndex, currIndex));
					cellEdges[p3.index].push_back(make_pair(currIndex, nIndex));
					linesMesh.addIndex(currIndex);
					linesMesh.addIndex(nIndex);

					neighbors[currIndex].push_back(nIndex);
					neighbors[nIndex].push_back(currIndex);
				}
				if (y > 0 && conUp) {
					int nIndex = ptIndices[wy + x - w];
					cellEdges[p1.index].push_back(make_pair(currIndex, nIndex));
					cellEdges[p2.index].push_back(make_pair(nIndex, currIndex));
					linesMesh.addIndex(currIndex);
					linesMesh.addIndex(nIndex);
					neighbors[currIndex].push_back(nIndex);
					neighbors[nIndex].push_back(currIndex);

				}
			}
		}
	}

	cellLines.clear();
	for (auto & edges : cellEdges) {
		vector<list<int> > cells;
		list<int> cell;
		while (edges.size() > 2) {
			auto e = edges.back();
			edges.pop_back();
			cell.clear();
			cell.push_back(e.first);
			while (edges.size() > 2) {
				bool nextFound = false;
				for (auto it = edges.begin(); it != edges.end(); ++it) {
					if (it->first == e.second) {
						e = *it;
						cell.push_back(e.first);
						edges.erase(it);
						nextFound = true;
						break;
					}
				}
				if (!nextFound) break;
			}
			cells.push_back(cell);
		}
		cellLines.push_back(cells);
	}
	/*
	int numPts = linesMesh.getNumVertices();
	vector<bool> processed(numPts, false);
	//get boundaries
	polylines.clear();
	for (int i = 0; i < numPts; ++i) {
		if (!processed[i]) {
			vector<int> neighs = neighbors[i];
			//normal scenario
			if (neighs.size() == 2) {
				list<int> pline;
				pline.push_back(i);
				int next = neighs[0];
				//do one side
				while (true) {
					vector<int> neighs2 = neighbors[next];
					int curr = next;
					if (neighs2.size() == 2 && !processed[curr]) {
						if (neighs2[0] == pline.back()) {
							next = neighs2[1];
						}
						else {
							next = neighs2[0];
						}
						pline.push_back(curr);
						processed[curr] = true;
					}
					else {
						pline.push_back(curr);
						break;
					}
				}
				next = neighs[1];
				while (true) {
					vector<int> neighs2 = neighbors[next];
					int curr = next;
					if (neighs2.size() == 2 && !processed[curr]) {
						if (neighs2[0] == pline.front()) {
							next = neighs2[1];
						}
						else {
							next = neighs2[0];
						}
						pline.push_front(curr);
						processed[curr] = true;
					}
					else {
						pline.push_front(next);
						break;
					}
				}

				polylines.push_back(pline);
			}
			else if (neighs.size() > 2) {
				for (int j = 0; j < neighs.size(); ++j) {
					int next = neighs[j];
					if (next > i && neighbors[next].size() != 2) {
						list<int> pline;
						pline.push_back(i);
						pline.push_back(next);
						polylines.push_back(pline);
					}
				}
			}
		}
	}
	vector<list<list<int> > > cellPlines(pts.size()+1);
	for (auto & pline : polylines) {
		set<int> cells1 = ptCells[pline.front()];
		set<int> cells2 = ptCells[pline.back()];
		vector<int> lineCells;
		set_intersection(cells1.begin(), cells1.end(), cells2.begin(), cells2.end(), back_inserter(lineCells));
		for (auto & cell : lineCells) {
			cellPlines[cell].push_back(pline);
		}
	}

	cellLines.clear();
	for (auto & lines : cellPlines) {
		list<int> cell;
		if (lines.size() < 2) {
			cout << "something is wrong with this cell" << endl;
			cellLines.push_back(cell);
			continue;
		}
		list<int> line = lines.back();
		lines.pop_back();
		cell.insert(cell.end(), line.begin(), line.end());
		while (lines.size() > 0) {
			bool found = false;
			for (auto it = lines.begin(); it != lines.end(); ++it) {
				if (it->front() == cell.back()) {
					found = true;
					cell.insert(cell.end(), ++it->begin(), it->end());
					lines.erase(it);
					break;
				}
				else if(it->back() == cell.back()) {
					found = true;
					cell.insert(cell.end(), ++it->rbegin(), it->rend());
					lines.erase(it);
					break;
				}
				else if (it->front() == cell.front()) {
					found = true;
					for (auto it2 = ++it->begin(); it2 != it->end();++it2) {
						cell.push_front(*it2);
					}
					lines.erase(it);
					break;
				}
				else if (it->back() == cell.front()) {
					found = true;
					cell.insert(cell.begin(), it->begin(), --it->end());
					lines.erase(it);
					break;
				}
			}
			if (!found) {
				cout << "incomplete cell" << endl;
				break;
			}
		}
		cellLines.push_back(cell);
	}
	*/

}

void ofApp::savePDF() {
	record = true;

}
void ofApp::offsetCells() {
	cellOffsets.clear();
	for (int i = 0; i < pts.size(); ++i) {
		auto & cells = cellLines[i];
		for (auto & line : cells) {
			//if (line.size() > 3)	cellOffsets.push_back(offsetCell(line, pts[i]));
			if (line.size() > 3) {
				auto lines = offsetCell(line, pts[i]);
				cellOffsets.insert(cellOffsets.end(), lines.begin(), lines.end());
			}
		}
	}
	boundaryIndex = cellOffsets.size();
	auto & cells = cellLines.back();
	for (auto & line : cells) {
		cellOffsets.push_back(offsetCell(line, -(minThick*0.75 + maxThick*0.25)*0.5));
	}
}

vector<ofPoly> ofApp::offsetCell(list<int> & crv, AnisoPoint2f & pt) {
	float scaling = 3000;
	ClipperOffset co;
	// co.ArcTolerance = 1;
	Path P;
	Paths offsetP;
	float offset = ofClamp(offsetPercent / sqrt(pt.jacobian->determinant()), minThick*0.5, maxThick*0.5);

	if (doSmooth) {
		
		for (auto index : crv) {
			ofVec3f v = linesMesh.getVertex(index);
			IntPoint iPt(v.x * scaling, v.y * scaling);
			P.push_back(iPt);
		}
		co.AddPath(P, jtRound, etClosedPolygon);
		co.Execute(offsetP, -offset*scaling);
		if(offsetP.size() == 0) return vector<ofPoly>();
		P.clear();
		ofVec2f center;
		CleanPolygons(offsetP);
		for (auto & v : offsetP[0]) {
			//ofVec3f v = linesMesh.getVertex(index);
			Vector2f p(v.X, v.Y);
			p = (*pt.jacobian)*p;
			IntPoint iPt(p.coeff(0), p.coeff(1));
			P.push_back(iPt);
			center += ofVec2f(iPt.X, iPt.Y);
		}
		center /= crv.size();
		CleanPolygon(P);
		
		float radius = 9e20;

		//get exact radius from straight skeleton
		Polygon_2 poly;
		for (auto & pt : P) {
			poly.push_back(Point_2(pt.X, pt.Y));
		}
		boost::shared_ptr<Ss> iss = CGAL::create_interior_straight_skeleton_2(poly.vertices_begin(), poly.vertices_end());
		radius = 0;
		for (Ss::Vertex_handle vh = iss->vertices_begin(); vh != iss->vertices_end(); vh++) {
			if (!vh->has_infinite_time()) {
				radius = max(radius, (float)vh->time());
			}
		}

		//estimate radius
		//for (auto & iPt : P) {
		//	radius = min(radius, (iPt.X - center.x)*(iPt.X - center.x) + (iPt.Y - center.y)*(iPt.Y - center.y));
		//}
		//radius = sqrt(radius);
		//co.Execute(offsetP, -radius);
		//int tries = 0;
		//while (offsetP.size() == 0 && tries < 50) {
		//	radius *= .95;
		//	co.Execute(offsetP, -radius);
		//	tries++;
		//}
		//cout << tries << endl;
		if (radius < 200) {
			cout << "small " << radius << endl;
			float scaleFactor = 200 / radius;
			P.clear();
			center.set(0, 0);
			for (auto & v : offsetP[0]) {
				//ofVec3f v = linesMesh.getVertex(index);
				Vector2f p(v.X, v.Y);
				p = scaleFactor*(*pt.jacobian)*p;
				IntPoint iPt(p.coeff(0), p.coeff(1));
				P.push_back(iPt);
				center += ofVec2f(iPt.X, iPt.Y);
			}
			scaling *= scaleFactor;
			center /= crv.size();
			radius = 200;
		}
		CleanPolygon(P);
		co.Clear();
		co.AddPath(P, jtRound, etClosedPolygon);

		radius *= filletPercent;
		co.Execute(offsetP, -radius);
		//radius = min(radius,(radius - offset*scaling)*filletPercent + offset*scaling);
		
		//co.Execute(offsetP, -offset*scaling);
		vector<ofPoly> offsets;
		vector<ofVec3f> offsetPts;
		if (offsetP.size() > 0) {
			//visual offset for etching
			co.Clear();
			CleanPolygons(offsetP);
			co.AddPaths(offsetP, jtRound, etClosedPolygon);
			co.Execute(offsetP, radius);
			CleanPolygons(offsetP);
			Path longestP;
			int pLen = 0;
			Matrix2f inverse = pt.jacobian->inverse();
			
			for (auto & oP : offsetP) {
				if (oP.size() > pLen) {
					pLen = oP.size();
					longestP = oP;
				}
				for (int i = 0; i < oP.size(); i++) {
					//ofVec3f pt3D(oP[i].X / scaling, oP[i].Y/ scaling);
					Vector2f anisoPt(oP[i].X / scaling, oP[i].Y / scaling);
					if (!doEtchOffset)anisoPt = inverse*anisoPt;
					offsetPts.push_back(ofVec3f(anisoPt.coeff(0), anisoPt.coeff(1)));
				}
				offsets.push_back(offsetPts);
			}
			/*
			if (doEtchOffset) {
				co.Clear();
				for (auto & lPt : longestP) {
					Vector2f anisoPt(lPt.X, lPt.Y);
					anisoPt = inverse*anisoPt;
					lPt.X = anisoPt[0];
					lPt.Y = anisoPt[1];
				}
				co.AddPath(longestP, jtRound, etClosedPolygon);
				co.Execute(offsetP, etchOffset*scaling);
				longestP = offsetP[0];
			}*/
			
		}
		return offsets;
	}
	else {
		for (auto index : crv) {
			ofVec3f v = linesMesh.getVertex(index);
			IntPoint iPt(v.x * scaling, v.y * scaling);
			P.push_back(iPt);
		}
		CleanPolygon(P);
		//Paths simplerP;
		//SimplifyPolygon(P, simplerP);
		//CleanPolygons(simplerP);
		//P = simplerP[0];
		//CleanPolygon(P);
		//Polygon_2 poly;
		//for (auto & pt : P) {
		//	poly.push_back(Point_2(pt.X, pt.Y));
		//}
		//boost::shared_ptr<Ss> iss = CGAL::create_interior_straight_skeleton_2(poly.vertices_begin(), poly.vertices_end());
		//float radius = 0;
		//for (Ss::Vertex_handle vh = iss->vertices_begin(); vh != iss->vertices_end(); vh++) {
		//	if (!vh->has_infinite_time()) {
		//		radius = max(radius, (float)vh->time());
		//	}
		//}
		
		//Paths simplerP;
		//SimplifyPolygon(P, simplerP);
		//co.AddPaths(simplerP, jtRound, etClosedPolygon);
		//radius = ofClamp(radius*offsetPercent, minThick*0.5*scaling, maxThick*0.5*scaling);
		co.AddPath(P, jtRound, etClosedPolygon);
		co.Execute(offsetP, -offset*scaling);
		vector<ofVec3f> offsetPts;
		vector<ofPoly> offsets;
		if (offsetP.size() > 0) {
			CleanPolygons(offsetP);
			if (doEtchOffset) {
				co.Clear();
				co.AddPaths(offsetP, jtRound, etClosedPolygon);
				co.Execute(offsetP, etchOffset*scaling);
			}
			else {
				co.Clear();
				co.AddPaths(offsetP, jtRound, etClosedPolygon);
				co.Execute(offsetP, 1);
			}
			Path longestP;
			int pLen = 0;
			for (auto & oP : offsetP) {
				offsetPts.clear();
				if (oP.size() > pLen) {
					pLen = oP.size();
					longestP = oP;
				}

				for (int i = 0; i < oP.size(); i++) {
					ofVec3f pt3D(oP[i].X / scaling, oP[i].Y / scaling);
					offsetPts.push_back(pt3D);
				}
				offsets.push_back(offsetPts);
			}
			//Path & oP = longestP;
			
		}
		return offsets;
	}

}

vector<ofVec3f> ofApp::offsetCell(list<int> & crv, float amt) {
	float scaling = 10;
	ClipperOffset co;
	Path P;
	Paths offsetP;
	float offset = amt;

	for (auto index : crv) {
		ofVec3f v = linesMesh.getVertex(index);
		P.push_back(IntPoint(v.x*scaling, v.y*scaling));
	}
	co.AddPath(P, jtRound, etClosedPolygon);
	co.Execute(offsetP, -offset*scaling);

	vector<ofVec3f> offsetPts;
	if (offsetP.size() > 0) {
		//visual offset for etching
		CleanPolygons(offsetP);

		if (doEtchOffset) {
			co.Clear();
			co.AddPaths(offsetP, jtRound, etClosedPolygon);
			co.Execute(offsetP, -etchOffset*scaling);
		}

		Path & oP = offsetP[0];
		for (int i = 0; i < oP.size(); i++) {
			ofVec3f pt3D(oP[i].X / scaling, oP[i].Y / scaling);
			offsetPts.push_back(pt3D);
		}
	}
	return offsetPts;
}

void ofApp::optimize() {
	if (!isOptimizing) {
		optThread.setup(pts);
		optThread.w = w;
		optThread.h = h;
		optThread.minDensity = minDensity;
		optThread.maxDensity = maxDensity;
		isOptimizing = true;
		optThread.startThread(true, true);
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	ofTexture tempTex;
	switch (key) {
	case OF_KEY_RIGHT:
		progressAnimation();
		break;
	case 'o':
		cout << "optimize" << endl;
		optimize();
		break;
	case 'r':
		record = true;
		break;
	case 'p':
		paused = !paused;
		break;
	case 'a':
		anisotrophyStr = 1.0 / anisotrophyStr;
		break;
	case 'e':
		doEtchOffset = !doEtchOffset;
		offsetCells();
		break;
	case 's':
		cleanEdge = true;
		tempTex.allocate(baseImage.getPixels());
		tempTex.loadScreenData(drawOffsetX, 0, baseImage.getWidth(), baseImage.getHeight());
		tempTex.readToPixels(baseImage.getPixels());
		baseImage.mirror(true, false);
		baseImage.save("twesdf.png");
		setupImage();
		break;
	case 'f':
		drawFill = !drawFill;
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	if (x > drawOffsetX) {
		if (patternPts.size() > patternPtRads.size()) {
			//need to use this point to set radius
			ofVec2f myPt = ofVec2f(x - drawOffsetX, y);
			float myNewRad = myPt.distance(patternPts.back());
			patternPtRads.push_back(myNewRad);
			cout << myNewRad;
		}
		else {
			patternPts.push_back(ofVec2f(x - drawOffsetX, y));
		}
		
		
	}

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 
	baseImage.load(dragInfo.files[0]);
	setupImage();
}

void ofApp::setFunction(ofxDatGuiDropdownEvent e) {
	getAnisoPoint = anisoFunctions[e.child];
}

void ofApp::buttonEvent(ofxDatGuiButtonEvent e) {
	if (e.target->is("reset")) {
		reset();
	} else if (e.target->is("optimize")) {
		optimize();
	}
	else if (e.target->is("savePDF")) {
		savePDF();
	}
	else if (e.target->is("setupStage2")) {
		setupStage2();
	}
	else if (e.target->is("smoothing")) {
		doSmooth = e.target->getEnabled();
		offsetCells();
	}
	else if (e.target->is("cleanEdge")) {
		cleanEdge = e.target->getEnabled();
		getDistances();
		dualContour();
		offsetCells();
		
	}
	else if (e.target->is("clear points")) {
		patternPts.clear();
		patternPtRads.clear();
		reset();
	}
	else if (e.target->is("randomize")) {
		rando = ofRandom(20);
		reset();
	}
	
}