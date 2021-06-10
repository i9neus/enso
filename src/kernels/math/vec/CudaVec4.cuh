#pragma once

#include "CudaVecBase.cuh"
#include "CudaVec3.cuh"

namespace Cuda
{		
	struct __align__(16) vec4 : public VecBase<4>
	{
		enum _attrs : size_t { kDims = 4 };
		using kType = float;

		union
		{
			struct { float x, y, z, w; };
			struct { float i0, i1, i2, i3; };
			float data[4];

            __vec_swizzle<float, 4, 4, 0, 0, 0, 0> xxxx; __vec_swizzle<float, 4, 4, 0, 0, 0, 1> xxxy; __vec_swizzle<float, 4, 4, 0, 0, 0, 2> xxxz; __vec_swizzle<float, 4, 4, 0, 0, 0, 3> xxxw;
            __vec_swizzle<float, 4, 4, 0, 0, 1, 0> xxyx; __vec_swizzle<float, 4, 4, 0, 0, 1, 1> xxyy; __vec_swizzle<float, 4, 4, 0, 0, 1, 2> xxyz; __vec_swizzle<float, 4, 4, 0, 0, 1, 3> xxyw;
            __vec_swizzle<float, 4, 4, 0, 0, 2, 0> xxzx; __vec_swizzle<float, 4, 4, 0, 0, 2, 1> xxzy; __vec_swizzle<float, 4, 4, 0, 0, 2, 2> xxzz; __vec_swizzle<float, 4, 4, 0, 0, 2, 3> xxzw;
            __vec_swizzle<float, 4, 4, 0, 0, 3, 0> xxwx; __vec_swizzle<float, 4, 4, 0, 0, 3, 1> xxwy; __vec_swizzle<float, 4, 4, 0, 0, 3, 2> xxwz; __vec_swizzle<float, 4, 4, 0, 0, 3, 3> xxww;
            __vec_swizzle<float, 4, 4, 0, 1, 0, 0> xyxx; __vec_swizzle<float, 4, 4, 0, 1, 0, 1> xyxy; __vec_swizzle<float, 4, 4, 0, 1, 0, 2> xyxz; __vec_swizzle<float, 4, 4, 0, 1, 0, 3> xyxw;
            __vec_swizzle<float, 4, 4, 0, 1, 1, 0> xyyx; __vec_swizzle<float, 4, 4, 0, 1, 1, 1> xyyy; __vec_swizzle<float, 4, 4, 0, 1, 1, 2> xyyz; __vec_swizzle<float, 4, 4, 0, 1, 1, 3> xyyw;
            __vec_swizzle<float, 4, 4, 0, 1, 2, 0> xyzx; __vec_swizzle<float, 4, 4, 0, 1, 2, 1> xyzy; __vec_swizzle<float, 4, 4, 0, 1, 2, 2> xyzz; __vec_swizzle<float, 4, 4, 0, 1, 2, 3> xyzw;
            __vec_swizzle<float, 4, 4, 0, 1, 3, 0> xywx; __vec_swizzle<float, 4, 4, 0, 1, 3, 1> xywy; __vec_swizzle<float, 4, 4, 0, 1, 3, 2> xywz; __vec_swizzle<float, 4, 4, 0, 1, 3, 3> xyww;
            __vec_swizzle<float, 4, 4, 0, 2, 0, 0> xzxx; __vec_swizzle<float, 4, 4, 0, 2, 0, 1> xzxy; __vec_swizzle<float, 4, 4, 0, 2, 0, 2> xzxz; __vec_swizzle<float, 4, 4, 0, 2, 0, 3> xzxw;
            __vec_swizzle<float, 4, 4, 0, 2, 1, 0> xzyx; __vec_swizzle<float, 4, 4, 0, 2, 1, 1> xzyy; __vec_swizzle<float, 4, 4, 0, 2, 1, 2> xzyz; __vec_swizzle<float, 4, 4, 0, 2, 1, 3> xzyw;
            __vec_swizzle<float, 4, 4, 0, 2, 2, 0> xzzx; __vec_swizzle<float, 4, 4, 0, 2, 2, 1> xzzy; __vec_swizzle<float, 4, 4, 0, 2, 2, 2> xzzz; __vec_swizzle<float, 4, 4, 0, 2, 2, 3> xzzw;
            __vec_swizzle<float, 4, 4, 0, 2, 3, 0> xzwx; __vec_swizzle<float, 4, 4, 0, 2, 3, 1> xzwy; __vec_swizzle<float, 4, 4, 0, 2, 3, 2> xzwz; __vec_swizzle<float, 4, 4, 0, 2, 3, 3> xzww;
            __vec_swizzle<float, 4, 4, 0, 3, 0, 0> xwxx; __vec_swizzle<float, 4, 4, 0, 3, 0, 1> xwxy; __vec_swizzle<float, 4, 4, 0, 3, 0, 2> xwxz; __vec_swizzle<float, 4, 4, 0, 3, 0, 3> xwxw;
            __vec_swizzle<float, 4, 4, 0, 3, 1, 0> xwyx; __vec_swizzle<float, 4, 4, 0, 3, 1, 1> xwyy; __vec_swizzle<float, 4, 4, 0, 3, 1, 2> xwyz; __vec_swizzle<float, 4, 4, 0, 3, 1, 3> xwyw;
            __vec_swizzle<float, 4, 4, 0, 3, 2, 0> xwzx; __vec_swizzle<float, 4, 4, 0, 3, 2, 1> xwzy; __vec_swizzle<float, 4, 4, 0, 3, 2, 2> xwzz; __vec_swizzle<float, 4, 4, 0, 3, 2, 3> xwzw;
            __vec_swizzle<float, 4, 4, 0, 3, 3, 0> xwwx; __vec_swizzle<float, 4, 4, 0, 3, 3, 1> xwwy; __vec_swizzle<float, 4, 4, 0, 3, 3, 2> xwwz; __vec_swizzle<float, 4, 4, 0, 3, 3, 3> xwww;
            __vec_swizzle<float, 4, 4, 1, 0, 0, 0> yxxx; __vec_swizzle<float, 4, 4, 1, 0, 0, 1> yxxy; __vec_swizzle<float, 4, 4, 1, 0, 0, 2> yxxz; __vec_swizzle<float, 4, 4, 1, 0, 0, 3> yxxw;
            __vec_swizzle<float, 4, 4, 1, 0, 1, 0> yxyx; __vec_swizzle<float, 4, 4, 1, 0, 1, 1> yxyy; __vec_swizzle<float, 4, 4, 1, 0, 1, 2> yxyz; __vec_swizzle<float, 4, 4, 1, 0, 1, 3> yxyw;
            __vec_swizzle<float, 4, 4, 1, 0, 2, 0> yxzx; __vec_swizzle<float, 4, 4, 1, 0, 2, 1> yxzy; __vec_swizzle<float, 4, 4, 1, 0, 2, 2> yxzz; __vec_swizzle<float, 4, 4, 1, 0, 2, 3> yxzw;
            __vec_swizzle<float, 4, 4, 1, 0, 3, 0> yxwx; __vec_swizzle<float, 4, 4, 1, 0, 3, 1> yxwy; __vec_swizzle<float, 4, 4, 1, 0, 3, 2> yxwz; __vec_swizzle<float, 4, 4, 1, 0, 3, 3> yxww;
            __vec_swizzle<float, 4, 4, 1, 1, 0, 0> yyxx; __vec_swizzle<float, 4, 4, 1, 1, 0, 1> yyxy; __vec_swizzle<float, 4, 4, 1, 1, 0, 2> yyxz; __vec_swizzle<float, 4, 4, 1, 1, 0, 3> yyxw;
            __vec_swizzle<float, 4, 4, 1, 1, 1, 0> yyyx; __vec_swizzle<float, 4, 4, 1, 1, 1, 1> yyyy; __vec_swizzle<float, 4, 4, 1, 1, 1, 2> yyyz; __vec_swizzle<float, 4, 4, 1, 1, 1, 3> yyyw;
            __vec_swizzle<float, 4, 4, 1, 1, 2, 0> yyzx; __vec_swizzle<float, 4, 4, 1, 1, 2, 1> yyzy; __vec_swizzle<float, 4, 4, 1, 1, 2, 2> yyzz; __vec_swizzle<float, 4, 4, 1, 1, 2, 3> yyzw;
            __vec_swizzle<float, 4, 4, 1, 1, 3, 0> yywx; __vec_swizzle<float, 4, 4, 1, 1, 3, 1> yywy; __vec_swizzle<float, 4, 4, 1, 1, 3, 2> yywz; __vec_swizzle<float, 4, 4, 1, 1, 3, 3> yyww;
            __vec_swizzle<float, 4, 4, 1, 2, 0, 0> yzxx; __vec_swizzle<float, 4, 4, 1, 2, 0, 1> yzxy; __vec_swizzle<float, 4, 4, 1, 2, 0, 2> yzxz; __vec_swizzle<float, 4, 4, 1, 2, 0, 3> yzxw;
            __vec_swizzle<float, 4, 4, 1, 2, 1, 0> yzyx; __vec_swizzle<float, 4, 4, 1, 2, 1, 1> yzyy; __vec_swizzle<float, 4, 4, 1, 2, 1, 2> yzyz; __vec_swizzle<float, 4, 4, 1, 2, 1, 3> yzyw;
            __vec_swizzle<float, 4, 4, 1, 2, 2, 0> yzzx; __vec_swizzle<float, 4, 4, 1, 2, 2, 1> yzzy; __vec_swizzle<float, 4, 4, 1, 2, 2, 2> yzzz; __vec_swizzle<float, 4, 4, 1, 2, 2, 3> yzzw;
            __vec_swizzle<float, 4, 4, 1, 2, 3, 0> yzwx; __vec_swizzle<float, 4, 4, 1, 2, 3, 1> yzwy; __vec_swizzle<float, 4, 4, 1, 2, 3, 2> yzwz; __vec_swizzle<float, 4, 4, 1, 2, 3, 3> yzww;
            __vec_swizzle<float, 4, 4, 1, 3, 0, 0> ywxx; __vec_swizzle<float, 4, 4, 1, 3, 0, 1> ywxy; __vec_swizzle<float, 4, 4, 1, 3, 0, 2> ywxz; __vec_swizzle<float, 4, 4, 1, 3, 0, 3> ywxw;
            __vec_swizzle<float, 4, 4, 1, 3, 1, 0> ywyx; __vec_swizzle<float, 4, 4, 1, 3, 1, 1> ywyy; __vec_swizzle<float, 4, 4, 1, 3, 1, 2> ywyz; __vec_swizzle<float, 4, 4, 1, 3, 1, 3> ywyw;
            __vec_swizzle<float, 4, 4, 1, 3, 2, 0> ywzx; __vec_swizzle<float, 4, 4, 1, 3, 2, 1> ywzy; __vec_swizzle<float, 4, 4, 1, 3, 2, 2> ywzz; __vec_swizzle<float, 4, 4, 1, 3, 2, 3> ywzw;
            __vec_swizzle<float, 4, 4, 1, 3, 3, 0> ywwx; __vec_swizzle<float, 4, 4, 1, 3, 3, 1> ywwy; __vec_swizzle<float, 4, 4, 1, 3, 3, 2> ywwz; __vec_swizzle<float, 4, 4, 1, 3, 3, 3> ywww;
            __vec_swizzle<float, 4, 4, 2, 0, 0, 0> zxxx; __vec_swizzle<float, 4, 4, 2, 0, 0, 1> zxxy; __vec_swizzle<float, 4, 4, 2, 0, 0, 2> zxxz; __vec_swizzle<float, 4, 4, 2, 0, 0, 3> zxxw;
            __vec_swizzle<float, 4, 4, 2, 0, 1, 0> zxyx; __vec_swizzle<float, 4, 4, 2, 0, 1, 1> zxyy; __vec_swizzle<float, 4, 4, 2, 0, 1, 2> zxyz; __vec_swizzle<float, 4, 4, 2, 0, 1, 3> zxyw;
            __vec_swizzle<float, 4, 4, 2, 0, 2, 0> zxzx; __vec_swizzle<float, 4, 4, 2, 0, 2, 1> zxzy; __vec_swizzle<float, 4, 4, 2, 0, 2, 2> zxzz; __vec_swizzle<float, 4, 4, 2, 0, 2, 3> zxzw;
            __vec_swizzle<float, 4, 4, 2, 0, 3, 0> zxwx; __vec_swizzle<float, 4, 4, 2, 0, 3, 1> zxwy; __vec_swizzle<float, 4, 4, 2, 0, 3, 2> zxwz; __vec_swizzle<float, 4, 4, 2, 0, 3, 3> zxww;
            __vec_swizzle<float, 4, 4, 2, 1, 0, 0> zyxx; __vec_swizzle<float, 4, 4, 2, 1, 0, 1> zyxy; __vec_swizzle<float, 4, 4, 2, 1, 0, 2> zyxz; __vec_swizzle<float, 4, 4, 2, 1, 0, 3> zyxw;
            __vec_swizzle<float, 4, 4, 2, 1, 1, 0> zyyx; __vec_swizzle<float, 4, 4, 2, 1, 1, 1> zyyy; __vec_swizzle<float, 4, 4, 2, 1, 1, 2> zyyz; __vec_swizzle<float, 4, 4, 2, 1, 1, 3> zyyw;
            __vec_swizzle<float, 4, 4, 2, 1, 2, 0> zyzx; __vec_swizzle<float, 4, 4, 2, 1, 2, 1> zyzy; __vec_swizzle<float, 4, 4, 2, 1, 2, 2> zyzz; __vec_swizzle<float, 4, 4, 2, 1, 2, 3> zyzw;
            __vec_swizzle<float, 4, 4, 2, 1, 3, 0> zywx; __vec_swizzle<float, 4, 4, 2, 1, 3, 1> zywy; __vec_swizzle<float, 4, 4, 2, 1, 3, 2> zywz; __vec_swizzle<float, 4, 4, 2, 1, 3, 3> zyww;
            __vec_swizzle<float, 4, 4, 2, 2, 0, 0> zzxx; __vec_swizzle<float, 4, 4, 2, 2, 0, 1> zzxy; __vec_swizzle<float, 4, 4, 2, 2, 0, 2> zzxz; __vec_swizzle<float, 4, 4, 2, 2, 0, 3> zzxw;
            __vec_swizzle<float, 4, 4, 2, 2, 1, 0> zzyx; __vec_swizzle<float, 4, 4, 2, 2, 1, 1> zzyy; __vec_swizzle<float, 4, 4, 2, 2, 1, 2> zzyz; __vec_swizzle<float, 4, 4, 2, 2, 1, 3> zzyw;
            __vec_swizzle<float, 4, 4, 2, 2, 2, 0> zzzx; __vec_swizzle<float, 4, 4, 2, 2, 2, 1> zzzy; __vec_swizzle<float, 4, 4, 2, 2, 2, 2> zzzz; __vec_swizzle<float, 4, 4, 2, 2, 2, 3> zzzw;
            __vec_swizzle<float, 4, 4, 2, 2, 3, 0> zzwx; __vec_swizzle<float, 4, 4, 2, 2, 3, 1> zzwy; __vec_swizzle<float, 4, 4, 2, 2, 3, 2> zzwz; __vec_swizzle<float, 4, 4, 2, 2, 3, 3> zzww;
            __vec_swizzle<float, 4, 4, 2, 3, 0, 0> zwxx; __vec_swizzle<float, 4, 4, 2, 3, 0, 1> zwxy; __vec_swizzle<float, 4, 4, 2, 3, 0, 2> zwxz; __vec_swizzle<float, 4, 4, 2, 3, 0, 3> zwxw;
            __vec_swizzle<float, 4, 4, 2, 3, 1, 0> zwyx; __vec_swizzle<float, 4, 4, 2, 3, 1, 1> zwyy; __vec_swizzle<float, 4, 4, 2, 3, 1, 2> zwyz; __vec_swizzle<float, 4, 4, 2, 3, 1, 3> zwyw;
            __vec_swizzle<float, 4, 4, 2, 3, 2, 0> zwzx; __vec_swizzle<float, 4, 4, 2, 3, 2, 1> zwzy; __vec_swizzle<float, 4, 4, 2, 3, 2, 2> zwzz; __vec_swizzle<float, 4, 4, 2, 3, 2, 3> zwzw;
            __vec_swizzle<float, 4, 4, 2, 3, 3, 0> zwwx; __vec_swizzle<float, 4, 4, 2, 3, 3, 1> zwwy; __vec_swizzle<float, 4, 4, 2, 3, 3, 2> zwwz; __vec_swizzle<float, 4, 4, 2, 3, 3, 3> zwww;
            __vec_swizzle<float, 4, 4, 3, 0, 0, 0> wxxx; __vec_swizzle<float, 4, 4, 3, 0, 0, 1> wxxy; __vec_swizzle<float, 4, 4, 3, 0, 0, 2> wxxz; __vec_swizzle<float, 4, 4, 3, 0, 0, 3> wxxw;
            __vec_swizzle<float, 4, 4, 3, 0, 1, 0> wxyx; __vec_swizzle<float, 4, 4, 3, 0, 1, 1> wxyy; __vec_swizzle<float, 4, 4, 3, 0, 1, 2> wxyz; __vec_swizzle<float, 4, 4, 3, 0, 1, 3> wxyw;
            __vec_swizzle<float, 4, 4, 3, 0, 2, 0> wxzx; __vec_swizzle<float, 4, 4, 3, 0, 2, 1> wxzy; __vec_swizzle<float, 4, 4, 3, 0, 2, 2> wxzz; __vec_swizzle<float, 4, 4, 3, 0, 2, 3> wxzw;
            __vec_swizzle<float, 4, 4, 3, 0, 3, 0> wxwx; __vec_swizzle<float, 4, 4, 3, 0, 3, 1> wxwy; __vec_swizzle<float, 4, 4, 3, 0, 3, 2> wxwz; __vec_swizzle<float, 4, 4, 3, 0, 3, 3> wxww;
            __vec_swizzle<float, 4, 4, 3, 1, 0, 0> wyxx; __vec_swizzle<float, 4, 4, 3, 1, 0, 1> wyxy; __vec_swizzle<float, 4, 4, 3, 1, 0, 2> wyxz; __vec_swizzle<float, 4, 4, 3, 1, 0, 3> wyxw;
            __vec_swizzle<float, 4, 4, 3, 1, 1, 0> wyyx; __vec_swizzle<float, 4, 4, 3, 1, 1, 1> wyyy; __vec_swizzle<float, 4, 4, 3, 1, 1, 2> wyyz; __vec_swizzle<float, 4, 4, 3, 1, 1, 3> wyyw;
            __vec_swizzle<float, 4, 4, 3, 1, 2, 0> wyzx; __vec_swizzle<float, 4, 4, 3, 1, 2, 1> wyzy; __vec_swizzle<float, 4, 4, 3, 1, 2, 2> wyzz; __vec_swizzle<float, 4, 4, 3, 1, 2, 3> wyzw;
            __vec_swizzle<float, 4, 4, 3, 1, 3, 0> wywx; __vec_swizzle<float, 4, 4, 3, 1, 3, 1> wywy; __vec_swizzle<float, 4, 4, 3, 1, 3, 2> wywz; __vec_swizzle<float, 4, 4, 3, 1, 3, 3> wyww;
            __vec_swizzle<float, 4, 4, 3, 2, 0, 0> wzxx; __vec_swizzle<float, 4, 4, 3, 2, 0, 1> wzxy; __vec_swizzle<float, 4, 4, 3, 2, 0, 2> wzxz; __vec_swizzle<float, 4, 4, 3, 2, 0, 3> wzxw;
            __vec_swizzle<float, 4, 4, 3, 2, 1, 0> wzyx; __vec_swizzle<float, 4, 4, 3, 2, 1, 1> wzyy; __vec_swizzle<float, 4, 4, 3, 2, 1, 2> wzyz; __vec_swizzle<float, 4, 4, 3, 2, 1, 3> wzyw;
            __vec_swizzle<float, 4, 4, 3, 2, 2, 0> wzzx; __vec_swizzle<float, 4, 4, 3, 2, 2, 1> wzzy; __vec_swizzle<float, 4, 4, 3, 2, 2, 2> wzzz; __vec_swizzle<float, 4, 4, 3, 2, 2, 3> wzzw;
            __vec_swizzle<float, 4, 4, 3, 2, 3, 0> wzwx; __vec_swizzle<float, 4, 4, 3, 2, 3, 1> wzwy; __vec_swizzle<float, 4, 4, 3, 2, 3, 2> wzwz; __vec_swizzle<float, 4, 4, 3, 2, 3, 3> wzww;
            __vec_swizzle<float, 4, 4, 3, 3, 0, 0> wwxx; __vec_swizzle<float, 4, 4, 3, 3, 0, 1> wwxy; __vec_swizzle<float, 4, 4, 3, 3, 0, 2> wwxz; __vec_swizzle<float, 4, 4, 3, 3, 0, 3> wwxw;
            __vec_swizzle<float, 4, 4, 3, 3, 1, 0> wwyx; __vec_swizzle<float, 4, 4, 3, 3, 1, 1> wwyy; __vec_swizzle<float, 4, 4, 3, 3, 1, 2> wwyz; __vec_swizzle<float, 4, 4, 3, 3, 1, 3> wwyw;
            __vec_swizzle<float, 4, 4, 3, 3, 2, 0> wwzx; __vec_swizzle<float, 4, 4, 3, 3, 2, 1> wwzy; __vec_swizzle<float, 4, 4, 3, 3, 2, 2> wwzz; __vec_swizzle<float, 4, 4, 3, 3, 2, 3> wwzw;
            __vec_swizzle<float, 4, 4, 3, 3, 3, 0> wwwx; __vec_swizzle<float, 4, 4, 3, 3, 3, 1> wwwy; __vec_swizzle<float, 4, 4, 3, 3, 3, 2> wwwz; __vec_swizzle<float, 4, 4, 3, 3, 3, 3> wwww;

            __vec_swizzle<float, 4, 3, 0, 0, 0> xxx; __vec_swizzle<float, 4, 3, 0, 0, 1> xxy; __vec_swizzle<float, 4, 3, 0, 0, 2> xxz; __vec_swizzle<float, 4, 3, 0, 0, 3> xxw;
            __vec_swizzle<float, 4, 3, 0, 1, 0> xyx; __vec_swizzle<float, 4, 3, 0, 1, 1> xyy; __vec_swizzle<float, 4, 3, 0, 1, 2> xyz; __vec_swizzle<float, 4, 3, 0, 1, 3> xyw;
            __vec_swizzle<float, 4, 3, 0, 2, 0> xzx; __vec_swizzle<float, 4, 3, 0, 2, 1> xzy; __vec_swizzle<float, 4, 3, 0, 2, 2> xzz; __vec_swizzle<float, 4, 3, 0, 2, 3> xzw;
            __vec_swizzle<float, 4, 3, 0, 3, 0> xwx; __vec_swizzle<float, 4, 3, 0, 3, 1> xwy; __vec_swizzle<float, 4, 3, 0, 3, 2> xwz; __vec_swizzle<float, 4, 3, 0, 3, 3> xww;
            __vec_swizzle<float, 4, 3, 1, 0, 0> yxx; __vec_swizzle<float, 4, 3, 1, 0, 1> yxy; __vec_swizzle<float, 4, 3, 1, 0, 2> yxz; __vec_swizzle<float, 4, 3, 1, 0, 3> yxw;
            __vec_swizzle<float, 4, 3, 1, 1, 0> yyx; __vec_swizzle<float, 4, 3, 1, 1, 1> yyy; __vec_swizzle<float, 4, 3, 1, 1, 2> yyz; __vec_swizzle<float, 4, 3, 1, 1, 3> yyw;
            __vec_swizzle<float, 4, 3, 1, 2, 0> yzx; __vec_swizzle<float, 4, 3, 1, 2, 1> yzy; __vec_swizzle<float, 4, 3, 1, 2, 2> yzz; __vec_swizzle<float, 4, 3, 1, 2, 3> yzw;
            __vec_swizzle<float, 4, 3, 1, 3, 0> ywx; __vec_swizzle<float, 4, 3, 1, 3, 1> ywy; __vec_swizzle<float, 4, 3, 1, 3, 2> ywz; __vec_swizzle<float, 4, 3, 1, 3, 3> yww;
            __vec_swizzle<float, 4, 3, 2, 0, 0> zxx; __vec_swizzle<float, 4, 3, 2, 0, 1> zxy; __vec_swizzle<float, 4, 3, 2, 0, 2> zxz; __vec_swizzle<float, 4, 3, 2, 0, 3> zxw;
            __vec_swizzle<float, 4, 3, 2, 1, 0> zyx; __vec_swizzle<float, 4, 3, 2, 1, 1> zyy; __vec_swizzle<float, 4, 3, 2, 1, 2> zyz; __vec_swizzle<float, 4, 3, 2, 1, 3> zyw;
            __vec_swizzle<float, 4, 3, 2, 2, 0> zzx; __vec_swizzle<float, 4, 3, 2, 2, 1> zzy; __vec_swizzle<float, 4, 3, 2, 2, 2> zzz; __vec_swizzle<float, 4, 3, 2, 2, 3> zzw;
            __vec_swizzle<float, 4, 3, 2, 3, 0> zwx; __vec_swizzle<float, 4, 3, 2, 3, 1> zwy; __vec_swizzle<float, 4, 3, 2, 3, 2> zwz; __vec_swizzle<float, 4, 3, 2, 3, 3> zww;
            __vec_swizzle<float, 4, 3, 3, 0, 0> wxx; __vec_swizzle<float, 4, 3, 3, 0, 1> wxy; __vec_swizzle<float, 4, 3, 3, 0, 2> wxz; __vec_swizzle<float, 4, 3, 3, 0, 3> wxw;
            __vec_swizzle<float, 4, 3, 3, 1, 0> wyx; __vec_swizzle<float, 4, 3, 3, 1, 1> wyy; __vec_swizzle<float, 4, 3, 3, 1, 2> wyz; __vec_swizzle<float, 4, 3, 3, 1, 3> wyw;
            __vec_swizzle<float, 4, 3, 3, 2, 0> wzx; __vec_swizzle<float, 4, 3, 3, 2, 1> wzy; __vec_swizzle<float, 4, 3, 3, 2, 2> wzz; __vec_swizzle<float, 4, 3, 3, 2, 3> wzw;
            __vec_swizzle<float, 4, 3, 3, 3, 0> wwx; __vec_swizzle<float, 4, 3, 3, 3, 1> wwy; __vec_swizzle<float, 4, 3, 3, 3, 2> wwz; __vec_swizzle<float, 4, 3, 3, 3, 3> www;

            __vec_swizzle<float, 4, 2, 0, 0> xx; __vec_swizzle<float, 4, 2, 0, 1> xy; __vec_swizzle<float, 4, 2, 0, 2> xz; __vec_swizzle<float, 4, 2, 0, 3> xw;
            __vec_swizzle<float, 4, 2, 1, 0> yx; __vec_swizzle<float, 4, 2, 1, 1> yy; __vec_swizzle<float, 4, 2, 1, 2> yz; __vec_swizzle<float, 4, 2, 1, 3> yw;
            __vec_swizzle<float, 4, 2, 2, 0> zx; __vec_swizzle<float, 4, 2, 2, 1> zy; __vec_swizzle<float, 4, 2, 2, 2> zz; __vec_swizzle<float, 4, 2, 2, 3> zw;
            __vec_swizzle<float, 4, 2, 3, 0> wx; __vec_swizzle<float, 4, 2, 3, 1> wy; __vec_swizzle<float, 4, 2, 3, 2> wz; __vec_swizzle<float, 4, 2, 3, 3> ww;
		};

		vec4() = default;
        vec4(const vec4&) = default;
		__host__ __device__ explicit vec4(const float v) : x(v), y(v), z(v), w(v) {}
		__host__ __device__ vec4(const float& x_, const float& y_, const float& z_, const float& w_) : x(x_), y(y_), z(z_), w(w_) {}
		__host__ __device__ vec4(const float& x_, const vec3& v) : x(x_), y(v.x), z(v.y), w(v.z) {}
		__host__ __device__ vec4(const vec2& v, const float& z_, const float& w_) : x(v.x), y(v.y), z(z_), w(w_) {}
        __host__ __device__ vec4(const vec3& v, const float& w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
		template<typename T, typename = std::enable_if<std::is_base_of<VecBase<4>, T>::value>::type> 
		__host__ __device__ vec4(const T& other) : x(float(other.x)), y(float(other.y)), z(float(other.z)), w(float(other.w)) {}

        template<int... In>
        __host__ __device__ inline vec4(const __vec_swizzle<float, 4, 4, In...>& swizzled)
        {
            swizzled.unpack(data);
        }

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }
		__host__ __device__ inline vec4& operator=(const float v) { x = y = z = w = v;  return *this; }

		__host__ inline std::string format() const { return tfm::format("{%.10f, %.10f, %.10f, %.10f}", x, y, z, w); }
	};

	__host__ __device__ inline vec4 operator +(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
	__host__ __device__ inline vec4 operator +(const vec4& lhs, const float& rhs) { return vec4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs, const float& rhs) { return vec4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs) { return vec4(-lhs.x, -lhs.y, -lhs.z, -lhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const float& rhs) { return vec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
	__host__ __device__ inline vec4 operator *(const float& lhs, const vec4& rhs) { return vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w); }
    __host__ __device__ inline vec4 operator /(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }
    __host__ __device__ inline vec4 operator /(const vec4& lhs, const float& rhs) { return vec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs); }
	
    __host__ __device__ inline vec4& operator +=(vec4& lhs, const vec4& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }
	__host__ __device__ inline vec4& operator -=(vec4& lhs, const vec4& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }
	__host__ __device__ inline vec4& operator *=(vec4& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; lhs.w *= rhs; return lhs; }
	__host__ __device__ inline vec4& operator /=(vec4& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; lhs.w /= rhs; return lhs; }
    
    template<int X, int Y, int Z, int W>
    __host__ __device__ inline  __vec_swizzle<float, 4, 4, X, Y, Z, W>& operator +=(__vec_swizzle<float, 4, 4, X, Y, Z, W>& lhs, const vec4& rhs)
    {
        lhs.data[X] += rhs.x; lhs.data[Y] += rhs.y; lhs.data[Z] += rhs.z; lhs.data[W] += rhs.w; return lhs;
    }
    template<int X, int Y, int Z, int W>
    __host__ __device__ inline  __vec_swizzle<float, 4, 4, X, Y, Z, W>& operator -=(__vec_swizzle<float, 4, 4, X, Y, Z, W>& lhs, const vec4& rhs)
    {
        lhs.data[X] -= rhs.x; lhs.data[Y] -= rhs.y; lhs.data[Z] -= rhs.z; lhs.data[W] -= rhs.w; return lhs;
    } 

	__host__ __device__ inline float dot(const vec4& lhs, const vec4& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }
	__host__ __device__ inline float length2(const vec4& v) { return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
	__host__ __device__ inline float length(const vec4& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
	__host__ __device__ inline vec4 normalize(const vec4& v) { return v / length(v); }
	__host__ __device__ inline vec4 fmod(const vec4& a, const vec4& b) { return vec4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }
	__host__ __device__ inline vec4 pow(const vec4& a, const vec4& b) { return vec4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w)); }
	__host__ __device__ inline vec4 exp(const vec4& a) { return vec4(expf(a.x), expf(a.y), expf(a.z), expf(a.w)); }
	__host__ __device__ inline vec4 log(const vec4& a) { return vec4(logf(a.x), logf(a.y), logf(a.z), logf(a.w)); }
	__host__ __device__ inline vec4 log10(const vec4& a) { return vec4(log10f(a.x), log10f(a.y), log10f(a.z), log10f(a.w)); }
	__host__ __device__ inline vec4 log2(const vec4& a) { return vec4(log2f(a.x), log2f(a.y), log2f(a.z), log2f(a.w)); }
	__host__ __device__ inline vec4 clamp(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }
	__host__ __device__ inline vec4 saturate(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f), clamp(v.w, 0.0f, 1.0f)); }
	__host__ __device__ inline vec4 abs(const vec4& a) { return vec4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w)); }
	__host__ __device__ inline float sum(const vec4& a) { return a.x + a.y + a.z + a.w; }
	__host__ __device__ inline vec4 ceil(const vec4& v) { return vec4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w)); }
	__host__ __device__ inline vec4 floor(const vec4& v) { return vec4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)); }
	__host__ __device__ inline vec4 sign(const vec4& v) { return vec4(sign(v.x), sign(v.y), sign(v.z), sign(v.w)); }

    __host__ __device__ inline bool operator==(const vec4& lhs, const vec4& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w; }
    __host__ __device__ inline bool operator!=(const vec4& lhs, const vec4& rhs) { return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w; }

	__host__ __device__ inline float cwiseMax(const vec4& v) 
	{ 
		float m = -FLT_MAX;
		for (int i = 0; i < 4; i++) { m = fmax(m, v[i]); }
		return m;
	}
	__host__ __device__ inline float cwiseMin(const vec4& v)
	{
		float m = FLT_MAX;
		for (int i = 0; i < 4; i++) { m = fmin(m, v[i]); }
		return m;
	}

	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	template<typename T>
	__host__ __device__ inline T cast(const vec4& v) { T r; r.x = v.x; r.y = v.y; r.z = v.z; r.w = v.w; return r; }
}