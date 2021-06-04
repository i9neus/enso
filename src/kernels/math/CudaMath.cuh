#pragma once

#include "vec/CudaIVec2.cuh"
#include "vec/CudaIVec3.cuh"
#include "vec/CudaIVec4.cuh"
#include "vec/CudaVec2.cuh"
#include "vec/CudaVec3.cuh"
#include "vec/CudaVec4.cuh"
#include "mat/CudaMat3.cuh"
#include "mat/CudaMat4.cuh"

namespace Cuda
{    	
	__host__ void VerifyTypeSizes();
	
	__host__ __device__ inline float cubrt(float a)								{ return copysignf(1.0f, a) * powf(fabs(a), 1.0f / 3.0f); }
	__host__ __device__ inline float toRad(float deg)								{ return kTwoPi * deg / 360; }
	__host__ __device__ inline float toDeg(float rad)								{ return 360 * rad / kTwoPi; }
	template<typename T> __host__ __device__ inline T sqr(const T& a)           	{ return a * a; }
	__host__ __device__ inline int mod2(int a, int b)								{ return ((a % b) + b) % b; }
	__host__ __device__ inline float mod2(float a, float b)							{ return fmodf(fmodf(a, b) + b, b); }
	__host__ __device__ inline vec3 mod2(vec3 a, vec3 b)							{ return fmod(fmod(a, b) + b, b); }	
	__host__ __device__ inline int sum(ivec2 a)										{ return a.x + a.y; }
	__host__ __device__ inline float luminance(vec3 v)								{ return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	__host__ __device__ inline float mean(vec3 v)									{ return v.x / 3 + v.y / 3 + v.z / 3; }
	__host__ __device__ inline float sin01(float a)								{ return 0.5f * sin(a) + 0.5f; }
	__host__ __device__ inline float cos01(float a)								{ return 0.5f * cos(a) + 0.5f; }
	__host__ __device__ inline float saturate(float a)								{ return clamp(a, 0.0, 1.0); }
	__host__ __device__ inline float saw01(float a)								{ return fabs(fract(a) * 2 - 1); }
	__host__ __device__ inline void sort(float& a, float& b)						{ if(a > b) { float s = a; a = b; b = s; } }
	__host__ __device__ inline void swap(float& a, float& b)						{ float s = a; a = b; b = s; }
	__host__ __device__ inline float max3(const float& a, const float& b, const float& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	__host__ __device__ inline float min3(const float& a, const float& b, const float& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
	
	template<typename T>
	__host__ __device__ inline T mix(const T& a, const T& b, const float& v) { return T(float(a) * (1 - v) + float(b) * v); }

	// Multi-purpose class describing a position and a direction vector
	struct PosDir
	{
	public:		
		vec3 o;
		union { vec3 d, n; };

		__device__ PosDir() = default;
		__device__ PosDir(const vec3& o_, const vec3& d_) : o(o_), d(d_) {}
	};
	
	class BidirectionalTransform
	{
	public:
		vec3 trans;
		mat3 fwd;
		mat3 inv;
		mat3 nInv;
		vec3 scale;

		__host__ __device__ BidirectionalTransform() 
		{
			MakeIdentity();
		}
			

		__host__ __device__ inline BidirectionalTransform(const vec3& t, const mat3& f) : trans(t), fwd(f)
		{ 
			inv = inverse(fwd);
			nInv = transpose(fwd);
			scale = vec3(1 / length(fwd[0]), 1 / length(fwd[1]), 1 / length(fwd[2]));
		}

		__host__ __device__ inline void MakeIdentity()
		{
			trans = 0.0f;
			fwd = inv = nInv = mat3::Indentity();
			scale = 1.0f;
		}

		__device__ inline PosDir RayToObjectSpace(const PosDir& world) const
		{
			PosDir object;
			object.o = world.o - trans;
			object.d = world.d + object.o;
			object.o = fwd * object.o;
			object.d = (fwd * object.d) - object.o;
			return object;
		}

		__device__ inline vec3 PointToObjectSpace(const vec3& world) const
		{
			return fwd * (world - trans);
		}

		__device__ inline PosDir HitToWorldSpace(const PosDir& object) const
		{
			PosDir world;
			world.o = inv * object.o;
			world.n = nInv * object.n;
			return world;
		}

		__device__ inline vec3 PointToWorldSpace(const vec3& object) const
		{
			return (inv * object) + trans;
		}
	};

    #define kZero vec3(0.0f)
	#define kZero4f vec4(0.0f)
	#define kZero3f vec3(0.0f)
	#define kZero2f vec2(0.0f)
	#define kZero4i ivec4(0)
	#define kZero3i ivec3(0)
	#define kZero2i ivec2(0)
	#define kZero4u uvec4(0u)
	#define kZero3u uvec3(0u)
	#define kZero2u uvec2(0u)

	#define kBlack vec3(0.0f)
	#define kWhite vec3(1.0f)

	__host__ __device__ inline mat3 ScaleMat3(const vec3 scale)
	{
		const vec3 invScale = vec3(1.0f) / scale;
		return mat3(vec3(invScale.x, 0.0, 0.0),
					vec3(0.0, invScale.y, 0.0),
					vec3(0.0, 0.0, invScale.z));
	}

	__host__ __device__ inline mat3 RotXMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(1.0, 0.0, 0.0),
					vec3(0.0, cosTheta, -sinTheta),
					vec3(0.0, sinTheta, cosTheta));
	}

	__host__ __device__ inline mat3 RotYMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, 0.0, sinTheta),
					vec3(0.0, 1.0, 0.0),
					vec3(-sinTheta, 0.0, cosTheta));
	}

	__host__ __device__ inline mat3 RotZMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, -sinTheta, 0.0),
					vec3(sinTheta, cosTheta, 0.0),
					vec3(0.0, 0.0, 1.0));
	}

	// Builds a composite matrix from three Euler angles, scale and translation vectors
	__host__ __device__ inline BidirectionalTransform CreateCompoundTransform(const vec3& theta, const vec3& translate = vec3(0.0f), const vec3& scale = vec3(1.0f))
	{
		mat3 mat = mat3::Indentity();

		if (scale != vec3(1.0)) { mat *= ScaleMat3(scale); }

		if (theta.x != 0.0) { mat *= RotXMat3(theta.x); }
		if (theta.y != 0.0) { mat *= RotYMat3(theta.y); }
		if (theta.z != 0.0) { mat *= RotZMat3(theta.z); }

		return BidirectionalTransform(translate, mat);
	}

	// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
	// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
	// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
	__host__ __device__ inline mat3 createBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat3(vec3(1 + s * n.x * n.x * a, s * b, -s * n.x),
							  vec3(b, s + n.y * n.y * a, -n.y),
							  n));
	}
	
	/*__host__ __device__ inline mat4 createBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1.0 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat4(vec4(1.0f + s * n.x * n.x * a, s * b, -s * n.x, 0.0),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0),
			vec4(n, 0.0),
			vec4(0.0, 0.0, 0.0, 1.0)));

		vec3 tangent = normalize(cross(n, (abs(dot(n, vec3(1.0, 0.0, 0.0))) < 0.5) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
		vec3 cotangent = cross(tangent, n);

		return transpose(mat4(vec4(tangent, 0.0), vec4(cotangent, 0.0), vec4(n, 0.0), vec4(kZero, 1.0)));
	}*/

	__host__ __device__ inline mat3 createBasis(vec3 n, vec3 up)
	{
		/*float s = sign(n.z);
		float a = -1 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat3(vec4(1 + s * n.x * n.x * a, s * b, -s * n.x, 0.0f),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0f),
			vec4(n, 0.0f),
			vec4(0.0f, 0.0f, 0.0f, 1.0f)));*/

		const vec3 tangent = normalize(cross(n, up));
		const vec3 cotangent = cross(tangent, n);

		return transpose(mat3(tangent, cotangent, n));
	}

	// Finds the roots of a quadratic equation of the form a.x^2 + b.x + c = 0
	__device__ inline bool quadraticSolve(float a, float b, float c, float& t0, float& t1)
	{
		float b2ac4 = b * b - 4 * a * c;
		if (b2ac4 < 0) { return false; }

		float sqrtb2ac4 = sqrt(b2ac4);
		t0 = (-b + sqrtb2ac4) / (2 * a);
		t1 = (-b - sqrtb2ac4) / (2 * a);
		return true;
	}
}
