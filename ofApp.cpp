#include "ofApp.h"
#include "constants.h"
#include "clipper.hpp"
using namespace ClipperLib;
using namespace ofxCv;
using namespace cv;

float w = 750; //750
float h = 1050; //1000

float maxDensity(50);//200 90 //150 810
float minDensity(14);//18 //30  200

float maxDensity2(80);
float minDensity2(12);
float anisotrophyStr(1.4f);

float etchOffset = 2.85;
float sizeFallOffExp = .75;

float minThick = 9.9f; //.05 inches rubber
float maxThick = minThick*2.0f; //.1 inches rubber
String imageName = "circle.png";

float auxeticity = .5;
int binW, binH, binD, binWH;
vector< vector<int> > bins;
bool isOptimizing = false;
bool record = false;
bool hasMask = true; //if you are using an image to crop the pattern

vector<AnisoPoint2f> nearPts;
AnisoPoint2f(*getAnisoPoint)(const ofVec3f & pt);
vector<AnisoPoint2f(*)(const ofVec3f & pt) > anisoFunctions;
vector<string> functionNames;

bool doEtchOffset = false;
Mat imgDist;
Mat imgGradX, imgGradY;

//--------------------------------------------------------------
void ofApp::setup(){
	
	setupImage(imageName);	
	
	//important thing
	//anisotropy function - give it a pt in space and it returns an anisotropic pt
	//getAnisoPtImg - uses image
	//getAnisoPtEdge - edge of the screen
	//getAnisoPtNoise
	//getAnisoPt - distance from a single Pt
	getAnisoPoint = &getAnisoPtUniform;// &getAnisoEdge;
	anisoFunctions.push_back(&getAnisoPt);
	anisoFunctions.push_back(&getAnisoPtSet);
	anisoFunctions.push_back(&getAnisoPtNoise);
	anisoFunctions.push_back(&getAnisoPointPts);
	anisoFunctions.push_back(&getAnisoPtImg);
	anisoFunctions.push_back(&getAnisoPtSin);
	anisoFunctions.push_back(&getAnisoPtBamboo);
	anisoFunctions.push_back(&getAnisoPtUniform);
	functionNames.push_back("pt");
	functionNames.push_back("ptSet");
	functionNames.push_back("noise");
	functionNames.push_back("secondStage");
	functionNames.push_back("img");
	functionNames.push_back("sin");
	functionNames.push_back("bamboo");
	functionNames.push_back("uniform");
	//minDensity = minDensity2;
	//maxDensity = maxDensity2;
	setupGui();

	reset();
}

void ofApp::reset() {
	binW = floor(w / max(minDensity,maxDensity)) + 1;
	binH = floor(h / max(minDensity,maxDensity)) + 1;

	linesMesh.setMode(OF_PRIMITIVE_LINES);
	bins.resize(binW*binH);
	for (int i = 0; i < bins.size(); ++i)bins.clear();
	initPts();
	getDistances();
	dualContour();
	simplifyCells();
	//offsetCells();
}

void ofApp::setupImage(string filename) {
	baseImage.load(filename);
	w = baseImage.getWidth();
	h = baseImage.getHeight();
	ofSetWindowShape(w+300, h);

	Mat initImg(baseImage.getHeight(), baseImage.getWidth(), CV_8UC1);
	cvtColor(toCv(baseImage), initImg, COLOR_BGR2GRAY);
	//toCv(baseImage).copyTo(initImg);
	//imgGradX = Mat(baseImage.getHeight(), baseImage.getWidth(), CV_32FC1);
	//imgGradY = Mat(baseImage.getHeight(), baseImage.getWidth(), CV_32FC1);
	distanceTransform(initImg, imgDist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
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

	ofxDatGuiSlider * slider = gui->addSlider("min density", 2, 200, minDensity);
	slider->bind(minDensity);
	slider = gui->addSlider("max density", 2, 300, maxDensity);
	slider->bind(maxDensity);
	slider = gui->addSlider("anisotropy", .5, 2, anisotrophyStr);
	slider->bind(anisotrophyStr);
	slider = gui->addSlider("min thickness", 2, 20, minThick);
	slider->bind(minThick);
	slider = gui->addSlider("max thickness", 2, 40, maxThick);
	slider->bind(maxThick);
	slider = gui->addSlider("auxeticity", -2, 2, auxeticity);
	slider->bind(auxeticity);

	ofxDatGuiDropdown * functionDd = gui->addDropdown("functions", functionNames);
	functionDd->onDropdownEvent(this, &ofApp::setFunction);
	gui->addButton("reset");
	gui->addButton("optimize");
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
		if (!hasMask || imgDist.at<float>((int)pt.y, (int)pt.x) > 0) {
			//density = ofLerp(maxDensity, minDensity, ofClamp(x/200.0,0,1));
			if (addPt(pt)) {
				fail = 0;
				cout << pts.size() << endl;
			}
			else {
				fail++;
			}
		}
	}
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
void ofApp::update(){
	if (isOptimizing) {
		if (!optThread.isThreadRunning()) {
			isOptimizing = false;
			pts = optThread.pts;
			getDistances();
			dualContour();
			simplifyCells();
			//offsetCells();
			cout << "done" << endl;
		}
		else if (ofGetFrameNum() % 20 == 0) {
			optThread.lock();
			pts = optThread.pts;
			optThread.unlock();
			getDistances();
			dualContour();
			simplifyCells();
			//offsetCells();
		}
	}
}

void ofApp::resetAnisotropy() {
	for (int i = 0; i < pts.size(); ++i) {
		AnisoPoint2f & pt = pts[i];
		AnisoPoint2f newPt = getAnisoPoint(ofVec3f(pt[0], pt[1]));
		pt = newPt;
	}
}

void ofApp::setupStage2() {
	nearPts = pts;
	getAnisoPoint = &getAnisoPointPts;
	minDensity  = minDensity2;
	maxDensity  = maxDensity2;
	anisotrophyStr = 1.4;// 1.0f / 1.4f;
	reset();
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	ofBackground(255);


	std::ostringstream ss;
	ss << "voronoi_dir" << anisotrophyStr << "_cellSz_" << minDensity << "-" << maxDensity << "_" << ofGetTimestampString() << ".pdf";

	ofPushMatrix();
	ofTranslate(300, 0);
	if (record) ofBeginSaveScreenAsPDF(ss.str());
	//drawPtEllipses();
	//distImage.draw(0,0);
	ofSetColor(0);
	ofNoFill();
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

	drawBuckled();
	/*
	for (auto & lines : cellLines) {
		ofBeginShape();
		for (auto & i : lines) {
			ofVertex(linesMesh.getVertex(i));
		}

		ofEndShape();
	}
	*/
	/*
	for (auto & cell : cellOffsets) {
		ofBeginShape();
		for (auto & pt : cell) {
			ofVertex(pt);
		}
		ofEndShape(true);
	}
	*/
	if (record) {
		record = false;
		ofEndSaveScreenAsPDF();
	}

	ofPopMatrix();

}

void ofApp::drawBuckled() {
	//assign direction
	vector<int> nodeDir(linesMesh.getNumVertices(), 0);
	for (int i = 0; i < linesMesh.getNumVertices(); ++i) {
		nodeDir[i] = ofNoise(linesMesh.getVertex(i))*2;
	}
	for (auto & lines : cellLines) {
		ofBeginShape();
		int prevIndex = lines.back();
		for (auto it = lines.begin(); it != lines.end(); ++it) {
			int index = *it;

			ofVec3f pt1 = linesMesh.getVertex(prevIndex);
			ofVec3f pt2 = linesMesh.getVertex(index);
			ofVec3f dir = pt2 - pt1;
			float len = dir.length();

			dir /= len;
			ofVec3f perp = dir.getRotated(90, ofVec3f(0, 0, 1));
			float buckles = 2.0;
			float flip = 0.5;
			if (nodeAssignment[index] != nodeAssignment[prevIndex]) { buckles = 1.0; flip = 1.0; }
			if (nodeAssignment[prevIndex] == 0) flip *= -1;
			for (int i = 0; i < 12; ++i) {
				ofVertex(pt1 + dir*i*len / 12 + auxeticity*flip*len*0.25*sin(i / 12.0 * PI*buckles)*perp);
			}

			prevIndex = index;
		}
		ofEndShape(true);

	}
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

void ofApp::getDistances() {
	distances.resize(w*h * 3);
	for (int i = 0; i < distances.size(); ++i) distances[i] = IndexDist(pts.size(), 9e9);
	if (hasMask) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (imgDist.at<float>(y, x) == 0) {
					distances[(w*y + x) * 3] = IndexDist(pts.size(), 0);
				}
			}
		}
	}
	Vector2f tempPt;
	for (int i = 0; i < pts.size();++i) {
		AnisoPoint2f & pt = pts[i];
		ofPushMatrix();

		Matrix2f transform = pt.jacobian->inverse();
		Vector2f vec(0, 1);
		vec = transform*vec;
		Vector2f vec2(1, 0);
		vec2 = transform*vec2;
		vec(0) = max(abs(vec(0)), abs(vec2(0)));
		vec(1) = max(abs(vec(1)), abs(vec2(1)));
		int minX = max(0, (int) (pt[0] - vec(0)*2));
		int maxX = min((int) w-1, (int)(pt[0] + vec(0)*2));
		int minY = max(0, (int)(pt[1] - vec(1)*2));
		int maxY = min((int)h - 1, (int)(pt[1] + vec(1)*2));
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
	if (hasMask) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (imgDist.at<float>(y, x) == 0) {
					distances[(w*y + x) * 3].dist = distances[(w*y + x) * 3+1 ].dist*.95;
				}
			}
		}
	}
}

ofVec2f getVoronoiIntersection(ofVec2f p1, ofVec2f p2, float side1A, float side1B, float side2A, float side2B) {
	//float eLen = p1.distance(p2);

	float x = (side1A - side2A) / (side1A - side1B + side2B - side2A);
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
					linesMesh.addIndex(currIndex);
					linesMesh.addIndex(nIndex);

					neighbors[currIndex].push_back(nIndex);
					neighbors[nIndex].push_back(currIndex);
				}
				if (y > 0 && conUp) {
					int nIndex = ptIndices[wy + x - w];
					linesMesh.addIndex(currIndex);
					linesMesh.addIndex(nIndex);
					neighbors[currIndex].push_back(nIndex);
					neighbors[nIndex].push_back(currIndex);

				}
			}
		}
	}

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
				}
				else if (it->back() == cell.front()) {
					found = true;
					cell.insert(cell.begin(), it->begin(), --it->end());
					lines.erase(it);
				}
			}
			if (!found) {
				cout << "incomplete cell" << endl;
				break;
			}
		}
		cellLines.push_back(cell);
	}

	//auxetic deform cellLines
	/*
	vector<ofVec3f> newPos(linesMesh.getNumVertices());
	vector<int> inf(linesMesh.getNumVertices(), 0);
	for (int i = 0; i < pts.size(); ++i) {
		list<int> & line = cellLines[i];
		AnisoPoint2f &center = pts[i];
		ofVec2f centerPt(center[0], center[1]);
		for (int index : line) {
			ofVec2f pt = linesMesh.getVertex(index);
			metric.distance_square(*center.pt, Vector2f(pt.x,pt.y), *center.jacobian);
			pt -= centerPt;
			float angle = atan2(pt.y, pt.x);
			float r = auxeticity*(pow((cos(angle*2)+1)*0.5, 1.0)*.5);
			pt *= r;
			newPos[index] += pt;
			inf[index]++;
		}
	}
	for (int i = 0; i < linesMesh.getNumVertices(); ++i) {
		linesMesh.getVertices()[i] += newPos[i] / inf[i];
	}
	*/

}

void ofApp::simplifyCells() {
	vector<int> numCells(linesMesh.getNumVertices(), 0);
	for (auto & lines : cellLines) {
		for (auto i : lines) {
			numCells[i]++;
		}
	}
	for (auto & lines : cellLines) {
		for (auto it = lines.begin(); it != lines.end();) {
			if (numCells[*it] > 2) {
				it++;
			}
			else {
				it = lines.erase(it);
			}
		}
	}
	//merge close
	for (auto & lines : cellLines) {
		int prevIndex = lines.back();
		auto prevIt = --lines.end();
		for (auto it = lines.begin(); it != lines.end();) {
			int index = *it;
			ofVec2f v1 = linesMesh.getVertex(prevIndex);
			ofVec2f v2 = linesMesh.getVertex(index);
			if (v1.distanceSquared(v2) < 64) {
				it = lines.erase(it);
				v1 = (v1 + v2)*0.5;
				linesMesh.setVertex(index, v1);
				linesMesh.setVertex(prevIndex, v1);
				if (index > prevIndex) {
					*prevIt = index;
					prevIndex = index;
				}
			} else {
				prevIndex = index;
				prevIt = it;
				it++;
			}
			
		}
	}

	nodeAssignment.resize(linesMesh.getNumVertices());
	assignNodesGreedy();
}

void ofApp::assignNodesGreedy() {
	vector<set<int> > neighbors(linesMesh.getNumVertices());
	list<int> processStack;

	for (auto & lines : cellLines) {
		int prevIndex = lines.back();
		for (auto index : lines) {
			neighbors[index].emplace(prevIndex);
			neighbors[prevIndex].emplace(index);
			prevIndex = index;
		}
	}
	for (auto & a : nodeAssignment) a = -1;
	for (int i = 0; i < neighbors.size(); ++i) {
		if (neighbors[i].size() > 0) {
			processStack.push_back(i);
			nodeAssignment[i] = 0;
			break;
		}
	}
	while (processStack.size() > 0) {
		int currIndex = processStack.front();
		processStack.pop_front();
		set<int> & neighs = neighbors[currIndex];
		int a = (nodeAssignment[currIndex] + 1) % 2;
		for (auto n : neighs) {
			if (nodeAssignment[n] == -1) {
				nodeAssignment[n] = a;
				processStack.push_back(n);
			}
		}
	}
}

void ofApp::offsetCells() {
	cellOffsets.clear();
	for (int i = 0; i < pts.size(); ++i) {
		auto & line = cellLines[i];
		cellOffsets.push_back(offsetCell(line, pts[i]));
	}
	cellOffsets.push_back(offsetCell(cellLines.back(), -(minThick + maxThick)*0.25));
}

vector<ofVec3f> ofApp::offsetCell(list<int> & crv, AnisoPoint2f & pt) {
	float scaling = 10;
	ClipperOffset co;
	Path P;
	Paths offsetP;
	float offset = ofClamp(.2 / sqrt(pt.jacobian->determinant()), minThick*0.5, maxThick*0.5);

	for (auto index : crv) {
		ofVec3f v = linesMesh.getVertex(index);
		P.push_back(IntPoint(v.x*scaling, v.y*scaling));
	}
	co.AddPath(P, jtRound, etClosedPolygon);
	co.Execute(offsetP, -offset*scaling);
	vector<ofVec3f> offsetPts;
	if (offsetP.size() > 0) {
		//visual offset for etching
		if (doEtchOffset) {
			co.Clear();
			co.AddPaths(offsetP, jtRound, etClosedPolygon);
			co.Execute(offsetP, etchOffset*scaling);
		}
		Path & oP = offsetP[0];
		for (int i = 0; i < oP.size(); i++) {
			ofVec3f pt3D(oP[i].X / scaling, oP[i].Y/ scaling);
			offsetPts.push_back(pt3D);
		}
	}
	return offsetPts;
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
	switch (key) {
	case 'o':
		cout << "optimize" << endl;
		optimize();
		break;
	case 's':
		setupStage2();
		break;
	case 'r':
		record = true;
		break;
	case 'a':
		//anisotrophyStr = 1.0 / anisotrophyStr;
		resetAnisotropy();
		getDistances();
		dualContour();
		//offsetCells();
		break;
	case 'e':
		doEtchOffset = !doEtchOffset;
		//offsetCells();
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
	setupImage(dragInfo.files[0]);
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
}