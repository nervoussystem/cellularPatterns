#pragma once

#include "ofMain.h"
#include "ccvt_metric.h"
#include "OptimizerThread.h"
#include "aniso.h"
#include "ofxCv.h"
#include "ofxDatGui.h"

typedef vector<ofVec3f> ofPoly;

struct IndexDist {
	unsigned int index;
	float dist;
	IndexDist() {
	}
	IndexDist(unsigned int _index, float _dist) {
		index = _index;
		dist = _dist;
	}
};

class ofApp : public ofBaseApp{

	public:
		void setup();
		void initPts();
		bool addPt(ofVec3f & pt);
		void update();
		void draw();
		void drawPtEllipses();
		void optimize();

		ofxDatGui * gui;
		void setupGui();

		void setupImage();
		void reset();

		vector<AnisoPoint2f> pts;

		vector<IndexDist> distances;
		OptimizerThread optThread;
		ccvt::MetricAniso2d metric;
		ofVboMesh linesMesh;
		vector<list<int> > polylines;
		vector<vector<list<int> > > cellLines;
		vector<ofPoly > cellOffsets;

		vector<ofPoly> offsetCell(list<int> & crv, AnisoPoint2f & center);
		vector<ofVec3f> offsetCell(list<int> & crv, float amt);

		ofImage baseImage;
		ofImage distImage;

		void getDistances();
		void getDistance(int i);
		void dualContour();
		void progressAnimation();

		void offsetCells();
		void setupStage2();

		void buttonEvent(ofxDatGuiButtonEvent e);
		void setFunction(ofxDatGuiDropdownEvent e);
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		void savePDF();
};
