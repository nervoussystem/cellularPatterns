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
extern float sizeFallOffExp;
extern float w;
extern float h;
extern AnisoPoint2f(*getAnisoPoint)(const ofVec3f & pt);
extern float anisoLerpRamp;
static float noiseScale = .002;// .001;
extern float rando;// .001;
static float noiseScaleDir = .002;// .002;

extern vector<ofVec2f> patternPts;
extern vector<float> patternPtRads;

inline AnisoPoint2f getAnisoPt(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	ofVec2f centerPt(375, 525);
	ofVec2f dir = ofVec2f(0, 1);// pt - centerPt;
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


	ofVec2f grad;
	int closest = 0;
	float myLen = 10000000000000;
	float size = minDensity;
	float anisotropy = 1;
	ofVec2f dir(0, 1);
	if (patternPts.size() > 0) {
		for (int i = 0; i < patternPts.size(); i++) {
			ofVec2f & a_pt = patternPts[i];
			ofVec3f myGrad = pt - a_pt;
			float lenSq = myGrad.lengthSquared();
			myGrad /= lenSq;
			//blend effects of the points
			grad += myGrad;

			float tempLen = sqrt(lenSq) / patternPtRads[i];
			if (tempLen < myLen) {
				closest = i;
				myLen = tempLen;
			}


		}

		 dir = grad;
		dir.normalize();

		float sLerp = (myLen);
		float localMax = min(maxDensity, patternPtRads[closest] / 3.0f);
		sLerp = ofClamp(sLerp, 0, 1);
		size = ofLerp(localMax, minDensity, pow(ofClamp(sLerp, 0, 1), sizeFallOffExp));
		 anisotropy = ofLerp(1, anisotrophyStr, pow(sLerp, anisoLerpRamp));
	}
	Matrix2f jac;
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
//stage 2
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
	//float borderD = minD+imgDist.at<float>((int)pt.y, (int)pt.x);
	//if (minD2 > borderD) minD2 = borderD;
	float totalD = minD + minD2;
	float t = ofClamp(minD / totalD * 2, 0, 1);
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float maxD = sqrt(1.0/closest.jacobian->determinant());
	maxD *= .3;
	minD = min(maxD, minDensity);
	maxD = min(maxD, maxDensity);
	float size = ofLerp(maxD, minD, pow(t, sizeFallOffExp));
	float anisoness = ofLerp(1, anisotrophyStr,pow(t, anisoLerpRamp));
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
float size = ofLerp(minDensity, maxDensity, pow(dist, sizeFallOffExp));

Matrix2f jac;
float anisotropy = ofLerp(1, anisotrophyStr, pow(1 - dist, anisoLerpRamp));
jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;

//jac << size*anisotrophyStr*dir.y, size / anisotrophyStr*dir.x, -size*anisotrophyStr*dir.x, size / anisotrophyStr*dir.y;
//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
//jac << size, 0.0, 0.0, size;
jac = jac.inverse().eval();
return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoPtSin(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;
	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	ofVec2f dir(1,0);
	dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float t = pow((sin((pt.x+ofNoise(pt.x*noiseScale,pt.y*noiseScale, rando)*151)*0.048) + 1)*0.5, sizeFallOffExp);
	float size = ofLerp(maxDensity, minDensity, t);

	Matrix2f jac;
	float anisotropy = ofLerp(1,anisotrophyStr,t);// sqrt(3);
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
	
	ofVec2f dir(1, 0);

	//get the distance from the edge, store that in d
	float d = pt.x;
	if (pt.y < d) {
		dir.set(0, 1);
		d = pt.y;
	}
	if (h - pt.y < d) {
		dir.set(0, -1);
		d = h - pt.y;
	}
	if (w - pt.x < d) {
		dir.set(-1, 0);
		d = w - pt.x;
	}


	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	//dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float lerpDistance = min(w, h) / 2.0f;
	float size = ofLerp(minDensity, maxDensity, ofClamp(d / lerpDistance, 0, 1));


	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
									  //jac << size*1.5*dir.y, size*.75*dir.x, -size*1.5*dir.x, size*0.75*dir.y;
	jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoSide(const ofVec3f &pt) {
	Vector2f pos;
	pos << pt.x, pt.y;

	ofVec2f dir(1, 0);

	//get the distance from the edge, store that in d
	float d = pt.x;
	

	//ofVec2f dir(ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 10), ofNoise(pt.x*noiseScaleDir, pt.y*noiseScaleDir, 20.123));
	//dir.normalize();
	//float size = ofLerp(5,9,ofClamp((pt.y-30)/300,0,1));
	float lerpDistance = w;
	float size = ofLerp(minDensity, maxDensity, ofClamp(d / lerpDistance, 0, 1));


	Matrix2f jac;
	float anisotropy = anisotrophyStr;// sqrt(3);
									  //jac << size*1.5*dir.y, size*.75*dir.x, -size*1.5*dir.x, size*0.75*dir.y;
	jac << size*anisotropy*dir.y, size / anisotropy*dir.x, -size*anisotropy*dir.x, size / anisotropy*dir.y;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}

inline AnisoPoint2f getAnisoPtBamboo(const ofVec3f &pt) {
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
	float maxD = sqrt(1.0 / closest.jacobian->determinant());
	maxD *= .2;
	float t = pow(ofClamp(minD / 0.4, 0, 1),1.5);
	float size = ofLerp(maxD, minDensity, t);
	float anisoness = ofLerp(1, anisotrophyStr, t);
	Matrix2f jac;
	Vector2f dir = (*closest.jacobian).transpose()*(*closest.jacobian)*(pos - *closest.pt);
	dir.normalize();

	if (dir.dot(Vector2f(0, 1)) > 0) {
		dir[0] = ofLerp(dir[0], 0, t);
		dir[1] = ofLerp(dir[1], 1, t);
	}
	else {
		dir[0] = ofLerp(dir[0], 0, t);
		dir[1] = ofLerp(dir[1], -1, t);
	}
	dir.normalize();

	jac << size*dir[1] * anisoness, size*dir[0] / anisoness, -size*dir[0] * anisoness, size*dir[1] / anisoness;
	//jac << 10*dir.y, 5*dir.x, -10*dir.x, 5*dir.y;
	//jac << size, 0.0, 0.0, size;
	jac = jac.inverse().eval();
	return AnisoPoint2f(pos, jac);
}
#endif