#ifndef CCVT_OPTIMIZER 
#define CCVT_OPTIMIZER 

#include "aniso.h"
#include "ccvt_metric.h"
#include "ccvt_optimizer.h"
#include "ccvt_point.h"
#include "ccvt_site.h"
#include "ofMain.h"

using namespace ccvt;

typedef Optimizer<Site<MyPoint>, MyPoint, MetricAniso2d > Ccvt;
extern bool hasMask;

class OptimizerThread : public ofThread {
public:
	Ccvt optimizer;
	vector<MyPoint> pts;
	bool isUniform;
	float maxDensity, minDensity;
	float w, h;
	list<MyPoint> fieldPts;
	MetricAniso2d metric;

	float stability;
	int stage;
	bool OptimizerThread::anisotropicPt(MyPoint & retPt) {
		
		//get random pt
		ofVec3f initPt(ofRandom(w), ofRandom(h), 0);
		//mask 
		if (hasMask && imgDist.at<float>(min((int)initPt.y, imgDist.rows-1), min((int)initPt.x,imgDist.cols-1)) <= 0) return false;
		MyPoint aniInitPt = getAnisoPoint(initPt);
		Matrix2f inverse = (*aniInitPt.jacobian).inverse();
		Vector2f spherePt(ofRandom(-1, 1), ofRandom(-1, 1));
		while (spherePt.squaredNorm() > 1) {
			spherePt = Vector2f(ofRandom(-1, 1), ofRandom(-1, 1));
		}
		spherePt *= 1.0 / min(maxDensity,minDensity);
		if (metric.distance_square(Vector2f::Zero(), spherePt, inverse) < 1) {
			Vector2f finalPt = inverse*spherePt;
			*retPt.pt = finalPt;
			*retPt.pt += Vector2f(initPt.x, initPt.y);
			*retPt.jacobian = *aniInitPt.jacobian;
			return true;
		}
		return false;
	}

	void OptimizerThread::initCcvt() {
		stage = 0;			
		int numFieldPts = (pts.size())*1024;
		fieldPts.clear();
		MyPoint newPt;
		
		while(fieldPts.size() < numFieldPts) {
			if(anisotropicPt(newPt)) {
				fieldPts.push_back(newPt);
			}			
		}
		unsigned int overallCapacity = static_cast<int>(fieldPts.size());
		Site<MyPoint>::List sites;
		for (int i = 0; i < pts.size(); ++i) {
			int capacity = overallCapacity / (pts.size() - i);
			overallCapacity -= capacity;
			sites.push_back(Site<MyPoint>(i, capacity, pts[i]));
		}

		stage = 1;
		optimizer.initialize(sites, fieldPts, metric);
		for (int i = 0; i < 6; ++i) optimizer.optimize(false);
	}

	void OptimizerThread::ccvtStep() {
	  stage = 2;
	  bool stable = false;
	  stability = 0;
	  stability = optimizer.optimize(true);

	  while(stability < 1) {
		  const Site<MyPoint>::Vector & sites = optimizer.sites();
		  Site<MyPoint>::Vector::const_iterator it = sites.begin();
		  lock();
		  for (int i = 0; i<pts.size(); ++i) {
			  pts[i] = it->location;
			  it++;
		  }
		  unlock();
		  for (int i = 0; i < 5; ++i) {
			  stability = optimizer.optimize(true);
		  }
	  }
	  const Site<MyPoint>::Vector & sites = optimizer.sites();
	  Site<MyPoint>::Vector::const_iterator it = sites.begin();
	  lock();
	  for (int i = 0; i<pts.size(); ++i) {
		  pts[i] = it->location;
		  it++;
	  }
	  unlock();
	}

	void OptimizerThread::threadedFunction() {
		initCcvt();
		ccvtStep();
	}

	void setup(vector<MyPoint> & _pts) {
		pts = _pts;
	}
};

#endif