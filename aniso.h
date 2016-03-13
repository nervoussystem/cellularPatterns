#ifndef ANISO_H
#define ANISO_H

#include "ofMain.h"
#include "matrix.h"
#include "constants.h"
#include "ofxCv.h"

typedef AnisoPoint2f MyPoint;

extern cv::Mat imgDist;
extern cv::Mat imgGradX, imgGradY;
extern vector<AnisoPoint2f> nearPts;
extern float anisotrophyStr;

extern AnisoPoint2f(*getAnisoPoint)(const ofVec3f & pt);

static float noiseScale = .0005;// .001;
static float noiseScaleDir = .001;// .002;

inline AnisoPoint2f getAnisoPt(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	ofVec2f centerPt(375, 525);
	ofVec2f dir = pt - centerPt;
	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float size = ofLerp(minDensity, maxDensity, ofClamp(pt.distance(centerPt)/500,0,1));
	
	
	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
	//jac << size*1.5*dir.y, size*.75*dir.x, -size*1.5*dir.x, size*0.75*dir.y;
	jac << size*anisotropy*dir.y, size/ anisotropy*dir.x, -size*anisotropy*dir.x, size/anisotropy*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos,jac);
}

inline AnisoPoint2f getAnisoPtSet(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	//ofVec2f centerPt(375, 525);

	//jessica do this part
	ofVec2f pts[3];
	pts[0] = ofVec2f(665,687);
	pts[1] = ofVec2f(306,518);
	pts[2] = ofVec2f(885,374);

	ofVec2f grad;
	float myLen = 10000000000000;
	for (auto & a_pt : pts) {
		ofVec3f myGrad = pt - a_pt;
		float lenSq = myGrad.lengthSquared();
		myGrad /= lenSq;
		//blend effects of the points
		grad += myGrad;

		float tempLen = sqrt(lenSq);
		myLen = min(tempLen, myLen);

	}

	//grad.normalize();
	
	ofVec2f dir = grad ;
	dir.normalize();

	float sLerp = (myLen - 7) /300;
	float size = ofLerp(minDensity, maxDensity, ofClamp(sLerp, 0, 1));
	

	//float size = ofLerp(minDensity, maxDensity, ofClamp(pt.distance(centerPt) / 500, 0, 1));
	//end do this stuff

	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
	jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoPtNoise(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	//ofVec2f dir = pt-ofVec2f(34.0,40.0);
	ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	//float size = ofLerp(minDensity, maxDensity, ofClamp(pt.distance(ofVec3f(375, 525))/500,0,1));
	float size = ofLerp(minDensity, maxDensity, ofNoise(pt.x*noiseScale, pt.y*noiseScale));

	Matrix2f jac;
	jac << size*anisotrophyStr*dir.y, size/ anisotrophyStr*dir.x, -size*anisotrophyStr*dir.x, size/anisotrophyStr*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoPointPts(const ofVec3f &pt) {
	//get nearest pt
	float minD = 9e9, minD2 = 9e9;
	AnisoPoint2f closest, closest2;
	Vector2f pos;
	pos << pt.x, pt.y;
	for (int i = 0; i < nearPts.size(); ++i) {
		AnisoPoint2f & nPt = nearPts[i];
		float d = nPt.distance_squared(pos);
		if (d < minD) {
			closest2 = closest;
			minD2 = minD;
			minD = d;
			closest = nPt;
		}
		else if (d < minD2) {
			minD2 = d;
			closest2 = nPt;
		}
	}
	minD = sqrt(minD);
	minD2 = sqrt(minD2);
	float totalD = minD + minD2;
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float maxD = sqrt(1.0/closest.jacobian->determinant());
	maxD *= .2;
	float size = ofLerp(maxD, minDensity, ofClamp(minD/totalD*2,0,1));
	float anisoness = ofLerp(1, anisotrophyStr,ofClamp(minD / totalD * 2, 0, 1));
	Matrix2f jac;
	Vector2f dir = (*closest.jacobian).transpose()*(*closest.jacobian)*(pos - *closest.pt);
	dir.normalize();
	jac << size*dir[1]* anisoness, size*dir[0]/ anisoness, -size*dir[0]* anisoness, size*dir[1]/ anisoness;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}


inline AnisoPoint2f getAnisoPtImg(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	//ofVec2f dir = pt-ofVec2f(34.0,40.0);
	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	//dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	//float size = ofLerp(minDensity, maxDensity, ofClamp(pt.distance(ofVec3f(375, 525))/500,0,1));
	float dist = imgDist.at<float>((int)pt.y, (int)pt.x);
	float dx = imgGradX.at<float>((int)pt.y, (int)pt.x);
	float dy = imgGradY.at<float>((int)pt.y, (int)pt.x);
	ofVec2f dir(dx, dy);
	dir.normalize();
	dist = pow(dist, 1);//falloff
	float size = ofLerp(minDensity, maxDensity, dist);

	Matrix2f jac;
	jac << size*anisotrophyStr*dir.y, size/anisotrophyStr*dir.x, -size*anisotrophyStr*dir.x, size/anisotrophyStr*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoPtSin(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	ofVec2f dir(1, sin(pt.x*.003));
	dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float size = (minDensity + maxDensity)*0.5;

	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
									  //jac << size*1.5*dir.y, size*.75*dir.x, -size*1.5*dir.x, size*0.75*dir.y;
	jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}


inline AnisoPoint2f getAnisoEdge(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	float w = 750;
	float h = 1050;
	ofVec2f centerPt(w*0.5, h*0.5);
	ofVec2f dir(1,0);
	float d = pt.x; 
	if (pt.y < d) {
		dir.set(0, 1);
		d = pt.y;
	}
	if (h-pt.y < d) {
		dir.set(0, -1);
		d = h-pt.y;
	}
	if (w - pt.x < d) {
		dir.set(-1, 0);
		d = w - pt.x;
	}
	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float size = ofLerp(minDensity, maxDensity, ofClamp(d / 525, 0, 1));


	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
									  //jac << size*1.5*dir.y, size*.75*dir.x, -size*1.5*dir.x, size*0.75*dir.y;
	jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}
#endif