#ifndef CCVT_MATRIX_H
#define CCVT_MATRIX_H

#include <Eigen/Dense>
using namespace Eigen;

struct AnisoPoint2f {
	Vector2f * pt;
	Matrix2f * jacobian;

	AnisoPoint2f() {
		pt = new Vector2f(Vector2f::Zero());
		jacobian = new Matrix2f(Matrix2f::Identity());
	}
	
	AnisoPoint2f(const AnisoPoint2f & p) {
		pt = new Vector2f(*(p.pt));
		jacobian = new Matrix2f(*(p.jacobian));
	}

	AnisoPoint2f(const Vector2f & p, const Matrix2f & j) {
		pt = new Vector2f(p);
		jacobian = new Matrix2f(j);
	}

	~AnisoPoint2f() {
		delete pt;
		delete jacobian;
	}

	float & operator[](size_t i) {
		return (*pt)[i];
	}

	const float & operator[](size_t i) const {
		return (*pt)[i];
	}

	inline void operator=(const AnisoPoint2f & p) {
		*pt = *p.pt;
		*jacobian = *p.jacobian;
	}

	float distance_squared(const AnisoPoint2f & p) {
		Vector2f v = *pt - *p.pt;
		return ((*jacobian)*v).transpose()*((*jacobian)*v);
	}

	float distance_squared(const Vector2f & p) {
		Vector2f v = *pt - p;
		return ((*jacobian)*v).transpose()*((*jacobian)*v);
	}

	static const int D = 2;
};
#endif