/*
	distfun.hpp - v0.01 - Vojtech Krs 2019
	Implements signed distance functions of basic primitives
	Allows conversion a CSG-like tree of primitives to a program 
	Implements distance program evaluation on CPU and GPU (CUDA)

	INSTALLATION:		
		Add
			#define DISTFUN_IMPLEMENTATION
		before #include

		Add
			#define DISTFUN_ENABLE_CUDA
		before #include in .cu/.cuh files

	USAGE:
		1. Construct a tree out of TreeNode, combining parametrized Primitives
		2. Call compileProgram on root TreeNode and get DistProgram
		3. Allocate DistProgram::staticSize() bytes (CPU or GPU)
		4. call commitProgramCPU/GPU to copy program into a previously allocated byte array
		5. Use distanceAtPos, distNormal or getNearestPoint to evaluate program at given position		

*/
#ifndef DISTFUN_HEADER
#define DISTFUN_HEADER

#ifdef DISTFUN_ENABLE_CUDA
	#include <cuda_runtime.h>
	#define __DISTFUN__ inline __host__ __device__
	#define __DISTFUN_T_ __host__ __device__
	#define GLM_FORCE_ALIGNED_GENTYPES
#else
	#define __DISTFUN__ inline
	#define __DISTFUN_T_
#endif

#define DISTFUN_ARRAY_PLACEHOLDER 1

/*
#ifdef __CUDA_ARCH__
	#pragma push
	#pragma diag_suppress 2886
	#include <glm/glm.hpp>
	#include <glm/gtx/norm.hpp>
	#pragma pop
#else*/
#pragma warning(push, 0)        
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#pragma warning(pop)
//#endif


#include <array>
#include <memory>
#include <vector>
#include <limits>


namespace distfun {

/*////////////////////////////////////////////////////////////////////////////////////
	Math definitions
////////////////////////////////////////////////////////////////////////////////////*/
	using vec4 = glm::vec4;
	using vec3 = glm::vec3;
	using vec2 = glm::vec2;
	using mat4 = glm::mat4;
	using mat3 = glm::mat3;
	using ivec3 = glm::ivec3;

	struct sdRay {
		vec3 origin;
		vec3 dir;
	};

	struct sdAABB {
		__DISTFUN__ sdAABB(vec3 minimum = vec3(std::numeric_limits<float>::max()), vec3 maximum = vec3(std::numeric_limits<float>::lowest())) :
			min(minimum),
			max(maximum) {
		}
		vec3 min;// = vec3(std::numeric_limits<float>::max());
		vec3 max;// = vec3(std::numeric_limits<float>::lowest());
		__DISTFUN__ vec3 diagonal() const { return max - min; }
		__DISTFUN__ vec3 center() const { return (min + max) * 0.5f; }

		__DISTFUN__ vec3 sideX() const { return vec3(max.x - min.x, 0, 0); }
		__DISTFUN__ vec3 sideY() const { return vec3(0, max.y - min.y, 0); }
		__DISTFUN__ vec3 sideZ() const { return vec3(0, 0, max.z - min.z); }

		__DISTFUN__ float volume() const {
			const auto diag = diagonal();
			return diag.x * diag.y * diag.z;
		}

		__DISTFUN__ vec3 corner(int index) const {
			switch (index) {
			case 0: return min;
			case 1: return min + sideX();
			case 2: return min + sideY();
			case 3: return min + sideZ();
			case 4: return max;
			case 5: return max - sideX();
			case 6: return max - sideY();
			case 7: return max - sideZ();
			}
			return center();
		}

		__DISTFUN__ sdAABB getOctant(int octant) const
		{
			switch (octant) {
			case 0: return { min, center() };
			case 1: return { min + sideX()*0.5f, center() + sideX()*0.5f };
			case 2: return { min + sideY()*0.5f, center() + sideY()*0.5f };
			case 3: return { min + sideZ()*0.5f, center() + sideZ()*0.5f };
			case 4: return { center(), max };
			case 5: return { center() - sideX()*0.5f, max - sideX()*0.5f };
			case 6: return { center() - sideY()*0.5f, max - sideY()*0.5f };
			case 7: return { center() - sideZ()*0.5f, max - sideZ()*0.5f };
			}
			return sdAABB();
		}

		__DISTFUN__ sdAABB getSubGrid(const ivec3 & gridSize, const ivec3 & gridIndex) const {
			const auto diag = diagonal();
			const vec3 cellSize = vec3(diag.x / gridSize.x, diag.y / gridSize.y, diag.z / gridSize.z);

			vec3 subgridmin = min + vec3(
				gridIndex.x * cellSize.x,
				gridIndex.y * cellSize.y,
				gridIndex.z * cellSize.z);
			
			return { 
				subgridmin,
				subgridmin + cellSize
			};
		}

		__DISTFUN__ sdAABB intersect(const sdAABB & b) const {
			return { glm::max(min, b.min), glm::min(max, b.max) };
		}

		__DISTFUN__ bool isValid() const {
			return min.x < max.x && min.y < max.y && min.z < max.z;
		}

		__DISTFUN__ bool isInside(vec3 pt) const
		{
			return pt.x >= min.x && pt.y > min.y && pt.z > min.z  &&
				pt.x < max.x && pt.y < max.y && pt.z < max.z;
		}
	};

	struct sdBoundingSphere {
		vec3 pos;
		float radius;
	};

/*////////////////////////////////////////////////////////////////////////////////////
	Primitive parameters		
////////////////////////////////////////////////////////////////////////////////////*/
	struct sdPlaneParams { char _empty; };
	struct sdSphereParams { float radius; };
	struct sdBoxParams { vec3 size; };
	struct sdCylinderParams { float radius; float height; };
	struct sdBlendParams { float k; };
	struct sdEllipsoidParams { vec3 size; };
	struct sdConeParam { float h; float r1; float r2; };
	struct sdGridParam { const void * ptr; ivec3 size; sdAABB bounds; };
	
/*////////////////////////////////////////////////////////////////////////////////////
	Primitive 
////////////////////////////////////////////////////////////////////////////////////*/
	struct sdPrimitive {

		enum Type {
			SD_PLANE,
			SD_SPHERE,
			SD_BOX,
			SD_CYLINDER,
			SD_ELLIPSOID,
			SD_CONE,
			SD_GRID,
			SD_OP_UNION,
			SD_OP_INTERSECT,
			SD_OP_DIFFERENCE,
			SD_OP_BLEND
		};

		union Params {
			sdPlaneParams plane;
			sdSphereParams sphere;
			sdBoxParams box;
			sdCylinderParams cylinder;
			sdBlendParams blend;
			sdEllipsoidParams ellipsoid;
			sdConeParam cone;
			sdGridParam grid;
		};

		float rounding;
		Type type;
		mat4 invTransform;
		Params params;
		sdBoundingSphere bsphere;

		sdPrimitive() : 
			rounding(0), 
			type(SD_PLANE), 
			params{ sdPlaneParams() }, 
			invTransform(mat4(1.0f)), 
			bsphere{vec3(0),0.0f} {
		}
	};	

	

/*////////////////////////////////////////////////////////////////////////////////////
	Primitive Distance Functions
////////////////////////////////////////////////////////////////////////////////////*/
	
	__DISTFUN__ float sdPlane(const vec3 & p, const sdPlaneParams & param)
	{
		return p.y;
	}

	__DISTFUN__ float sdSphere(const vec3 & p, const sdSphereParams & param)
	{
		return glm::length(p) - param.radius;
	}

	__DISTFUN__ float sdBox(const vec3 & p, const sdBoxParams & param)
	{
		vec3 d = glm::abs(p) - param.size;
		return glm::min(glm::max(d.x, glm::max(d.y, d.z)), 0.0f) + glm::length(glm::max(d, vec3(0.0f)));
	}

	__DISTFUN__ float sdCylinder(const vec3 & p, const sdCylinderParams & param)
	{
		vec2 d = abs(vec2(glm::length(vec2(p.x, p.z)), p.y)) - vec2(param.radius, param.height);
		return glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, 0.0f));
	}

	__DISTFUN__ float sdEllipsoid(const vec3 & p, const sdEllipsoidParams & param)
	{
		const auto & r = param.size;
		float k0 = glm::length(vec3(p.x / r.x, p.y / r.y, p.z / r.z));
		if (k0 == 0.0f)
			return glm::min(r.x, glm::min(r.y, r.z));

		float k1 = glm::length(vec3(p.x / (r.x*r.x), p.y / (r.y*r.y), p.z / (r.z*r.z)));
		return k0*(k0 - 1.0f) / k1;
	}

	__DISTFUN__ float sdCone(const vec3 & p, const sdConeParam & param)
	{
		vec2 q = vec2(glm::length(vec2(p.x, p.z)), p.y);

		vec2 k1 = vec2(param.r2, param.h);
		vec2 k2 = vec2(param.r2 - param.r1, 2.0f*param.h);
		vec2 ca = vec2(q.x - glm::min(q.x, (q.y < 0.0f) ? param.r1 : param.r2), abs(q.y) - param.h);
		vec2 cb = q - k1 + k2*glm::clamp(glm::dot(k1 - q, k2) / glm::dot(k2, k2), 0.0f, 1.0f);
		float s = (cb.x < 0.0f && ca.y < 0.0f) ? -1.0f : 1.0f;
		return s*glm::sqrt(glm::min(glm::dot(ca, ca), glm::dot(cb, cb)));

	}

	__DISTFUN__ size_t sdLinearIndex(const ivec3 & stride, const ivec3 & pos) {
		return stride.x * pos.x + stride.y * pos.y + stride.z * pos.z;
	}

	/*
		posInside must be relative to param.bounds.min (in the space of the grid)
	*/
	__DISTFUN__ float sdGridInside(const vec3 & posInside, const sdGridParam & param) {
		vec3 diag = param.bounds.diagonal();
		vec3 cellSize = { diag.x / param.size.x,diag.y / param.size.y, diag.z / param.size.z };
		ivec3 ipos = posInside * vec3(1.0f / cellSize.x, 1.0f / cellSize.y, 1.0f / cellSize.z);
		ipos = glm::clamp(ipos, ivec3(0), param.size);
		vec3 fract = posInside - vec3(ipos) * cellSize;

		const ivec3 s = ivec3(1, param.size.x, param.size.x * param.size.z);
		const size_t i = sdLinearIndex(s, ipos);
		const ivec3 maxPos = param.size - ivec3(1);

		const float * voxels = reinterpret_cast<const float*>(param.ptr);

		float value[8] = {
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(0,0,0), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(1,0,0), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(0,1,0), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(1,1,0), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(0,0,1), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(1,0,1), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(0,1,1), maxPos))],
			voxels[sdLinearIndex(s,glm::min(ipos + ivec3(1,1,1), maxPos))]
		};

		float front = glm::mix(
			glm::mix(value[0], value[1], fract.x),
			glm::mix(value[2], value[3], fract.x),
			fract.y
		);

		float back = glm::mix(
			glm::mix(value[4], value[5], fract.x),
			glm::mix(value[6], value[7], fract.x),
			fract.y
		);

		return glm::mix(front, back, fract.z);
	}

	__DISTFUN__ float sdGrid(const vec3 & p, const sdGridParam & param) {
		vec3 clamped = p;		
		
		const float EPS = 1e-6f;		
		bool inside = true;
		if (p.x <= param.bounds.min.x) { clamped.x = param.bounds.min.x + EPS; inside = false; }
		if (p.y <= param.bounds.min.y) { clamped.y = param.bounds.min.y + EPS; inside = false; }
		if (p.z <= param.bounds.min.z) { clamped.z = param.bounds.min.z + EPS; inside = false; }
		if (p.x >= param.bounds.max.x) { clamped.x = param.bounds.max.x - EPS; inside = false; }
		if (p.y >= param.bounds.max.y) { clamped.y = param.bounds.max.y - EPS; inside = false; }
		if (p.z >= param.bounds.max.z) { clamped.z = param.bounds.max.z - EPS; inside = false; }

		
		if (inside) {			
			return sdGridInside(p - param.bounds.min, param);
		}
		else {		
			vec3 posInside = (clamped - param.bounds.min);

#ifdef DISTFUN_GRID_OUTSIDE_GRADIENT
			vec3 res = (clamped - p) + sdGradient(posInside, EPS, sdGridInside, param);
			return glm::length(res);
#else
			return sdGridInside(posInside, param) + glm::length(clamped - p);
#endif
		}	
	}

	__DISTFUN__ float sdUnion(float a, float b) {
		return glm::min(a, b);
	}

	__DISTFUN__ float sdIntersection(float a, float b) {
		return glm::max(a, b);
	}

	__DISTFUN__ float sdDifference(float a, float b) {
		return glm::max(-a, b);
	}

	__DISTFUN__ float sdRound(float dist, float r) {
		return dist - r;
	}

	__DISTFUN__ float sdSmoothmin(float a, float b, float k) {
		float h = glm::clamp(0.5f + 0.5f*(a - b) / k, 0.0f, 1.0f);
		return glm::mix(a, b, h) - k*h*(1.0f - h);
	}


	



/*////////////////////////////////////////////////////////////////////////////////////
	Transformations and primitive distance
////////////////////////////////////////////////////////////////////////////////////*/
	
	template <class Func, class ... Args>
	__DISTFUN_T_ float sdTransform(
		const vec3 & p,
		const mat4 & invTransform,
		Func f,
		Args ... args
	) {
		return f(vec3(invTransform * vec4(p.x, p.y, p.z, 1.0f)), args ...);
	}

	__DISTFUN__ vec3 sdTransformPos(
		const vec3 & p,
		const mat4 & m
	) {
		return vec3(
			m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
			m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
			m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2]
		);

	}

	template <class Func, class ... Args>
	__DISTFUN_T_ vec3 sdGradient(vec3 pos, float eps, Func f, Args ... args)
	{
		return vec3(
			f(pos + vec3(eps, 0, 0), args ...) - f(pos - vec3(eps, 0, 0), args ...),
			f(pos + vec3(0, eps, 0), args ...) - f(pos - vec3(0, eps, 0), args ...),
			f(pos + vec3(0, 0, eps), args ...) - f(pos - vec3(0, 0, eps), args ...));

	}

	template <class Func, class ... Args>
	__DISTFUN_T_ vec3 sdNormal(vec3 pos, float eps, Func f, Args ... args)
	{
		return glm::normalize(sdGradient(pos, eps, f, args...));
	}
	

	__DISTFUN__  float sdPrimitiveDistance(
		const vec3 & pos,
		const sdPrimitive & prim
	) {

		const vec3 tpos = sdTransformPos(pos, prim.invTransform);

		switch (prim.type) {
		case sdPrimitive::SD_SPHERE:
			return  sdSphere(tpos, prim.params.sphere);
		case sdPrimitive::SD_ELLIPSOID:
			return  sdEllipsoid(tpos, prim.params.ellipsoid);
		case sdPrimitive::SD_CONE:
			return  sdCone(tpos, prim.params.cone);
		case sdPrimitive::SD_BOX:
			return sdRound(sdBox(tpos, prim.params.box), prim.rounding);
		case sdPrimitive::SD_CYLINDER:
			return 	sdRound(sdCylinder(tpos, prim.params.cylinder), prim.rounding);
		case sdPrimitive::SD_PLANE:
			return sdPlane(tpos, prim.params.plane);
		case sdPrimitive::SD_GRID:
			return sdGrid(tpos, prim.params.grid);
		}
		return FLT_MAX;
	}


	__DISTFUN__ float sdPrimitiveDifference(const vec3 & pos, const sdPrimitive & a, const sdPrimitive & b) {
		return sdDifference(sdPrimitiveDistance(pos, a), sdPrimitiveDistance(pos, b));
	}
	
	__DISTFUN__ vec3 sdGetNearestPoint(const vec3 & pos, const sdPrimitive & prim, float dx = 0.001f) {
		float d = sdPrimitiveDistance(pos, prim);
		vec3 N = sdNormal(pos, dx, sdPrimitiveDistance, prim);
		return pos - d*N;
	}

	__DISTFUN__  sdAABB sdPrimitiveBounds(		
		const sdPrimitive & prim,
		float pollDistance,
		float dx = 0.001f
	){
		
		mat4 transform = glm::inverse(prim.invTransform);		
		vec3 pos = vec3(transform * vec4(vec3(0.0f), 1.0f));


		vec3 exPt[6] = {
			pos + pollDistance * vec3(-1,0,0),
			pos + pollDistance * vec3(0,-1,0),
			pos + pollDistance * vec3(0,0,-1),
			pos + pollDistance * vec3(1,0,0),
			pos + pollDistance * vec3(0,1,0),
			pos + pollDistance * vec3(0,0,1)
		};

		sdAABB boundingbox;

		boundingbox.min = {
			sdGetNearestPoint(exPt[0], prim, dx).x,
			sdGetNearestPoint(exPt[1], prim, dx).y,
			sdGetNearestPoint(exPt[2], prim, dx).z
		};

		boundingbox.max = {
			sdGetNearestPoint(exPt[3], prim, dx).x,
			sdGetNearestPoint(exPt[4], prim, dx).y,
			sdGetNearestPoint(exPt[5], prim, dx).z
		};


		return boundingbox;

	}

	
/*////////////////////////////////////////////////////////////////////////////////////
	Tree  (CPU only)
////////////////////////////////////////////////////////////////////////////////////*/
	struct sdTreeNode {
		sdPrimitive primitive;
		std::array<std::shared_ptr<sdTreeNode>, 2> children;

		sdTreeNode() : children{ nullptr,nullptr }{
		}
		bool isLeaf() const {
			return !children[0] && !children[1];
		}

	};

	bool sdIsLeaf(const sdTreeNode & node);
	int sdTreeDepth(const sdTreeNode & node);	

/*////////////////////////////////////////////////////////////////////////////////////
	Program
////////////////////////////////////////////////////////////////////////////////////*/
	

	//Evaluation order
	struct sdInstruction {
		enum Type {
			REG_OBJ,
			REG_REG,
			OBJ
		};
		
		sdInstruction(Type type = OBJ) :
			itype(type)
		{}
		using RegIndex = char;
				
		/*	
			Register DEST receives result of op on register reg and Primitive prim.
			Op is parametrized by _p0
			DEST <- reg op(_p0) prim			
		*/
		struct AddrRegObj {
			RegIndex reg;
			sdPrimitive prim;
			float _p0;
		};

		/*
			Register DEST receives result of Primitive prim.			
			DEST <- prim
		*/
		struct AddrObj {
			sdPrimitive prim;
		};

		/*
			Register DEST receives result of op on register reg[0] and reg[1].
			Op is parametrized by _p0
			DEST <- reg[0] op(_p0) reg[1]
		*/
		struct AddrRegReg {
			RegIndex reg[2];
			float _p0;
		};

		union Addr {		
			Addr() { memset(this, 0, sizeof(Addr)); }
			AddrRegObj regobj;
			AddrObj obj;
			AddrRegReg regreg;
		};

		sdPrimitive::Type optype;
		Type itype;
		RegIndex regTarget;
		Addr addr;

	};

	struct sdProgram {
		int instructionCount;
		int registers;
		std::vector<sdInstruction> instructions;

		sdProgram() : instructionCount(0), registers(0) {
		}

		size_t staticSize() const {
			return 2 * sizeof(int) + sizeof(sdInstruction)*instructions.size();
		}
	};
	
	struct sdProgramStatic{
		sdProgramStatic(const sdProgramStatic &) = delete;
		sdProgramStatic & operator=(const sdProgramStatic &) = delete;

		__DISTFUN__ size_t staticSize() const {
			return 2 * sizeof(int) + sizeof(sdInstruction)*instructionCount;
		}

		int instructionCount;
		int registers;
		sdInstruction instructions[DISTFUN_ARRAY_PLACEHOLDER]; //In-place pointer for variable size
		
	};

	template <typename T>
	__DISTFUN_T_ const sdProgramStatic * sdCastProgramStatic(const T * ptr) {
		return reinterpret_cast<const sdProgramStatic *>(ptr);
	}

	template <typename T>
	__DISTFUN_T_ sdProgramStatic * sdCastProgramStatic(T * ptr) {
		return reinterpret_cast<sdProgramStatic *>(ptr);
	}




/*////////////////////////////////////////////////////////////////////////////////////
	Tree To Program conversion
////////////////////////////////////////////////////////////////////////////////////*/
	
	sdProgram sdCompile(const sdTreeNode & node);

	
	template <class CopyFun>
	void sdCommit(void * destination, const sdProgram & program, CopyFun copyFunction){
		sdProgramStatic *dst = reinterpret_cast<sdProgramStatic*>(destination);		
		copyFunction(&dst->instructionCount, &program.instructionCount, sizeof(int));
		copyFunction(&dst->registers, &program.registers, sizeof(int));
		copyFunction(&dst->instructions, program.instructions.data(), sizeof(sdInstruction) * program.instructionCount);
	}

	const sdProgramStatic * sdCommitCPU(void * destination, const sdProgram & program);

#ifdef DISTFUN_ENABLE_CUDA
	const sdProgramStatic * sdCommitGPU(void * destination, const sdProgram & program);
#endif

	

/*////////////////////////////////////////////////////////////////////////////////////
	Program evaluation
////////////////////////////////////////////////////////////////////////////////////*/

	template <size_t regNum = 4>
	__DISTFUN_T_ float sdDistanceAtPos(const vec3 & pos, const sdProgramStatic * programPtr) {

		//Registers
		float r[regNum];		

		if (programPtr->instructionCount == 0)
			return FLT_MAX;
		
		//Step through each instruction
		for (auto pc = 0; pc < programPtr->instructionCount; pc++) {

			const sdInstruction & i = programPtr->instructions[pc];
			float & dest = r[i.regTarget];

			if (i.itype == sdInstruction::OBJ) {
				dest = sdPrimitiveDistance(pos, i.addr.obj.prim);
			}
			else if (i.itype == sdInstruction::REG_REG) {
				switch (i.optype) {
				case sdPrimitive::SD_OP_UNION:
					dest = sdUnion(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;
				case sdPrimitive::SD_OP_BLEND:
					dest = sdSmoothmin(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]], i.addr.regreg._p0);
					break;
				case sdPrimitive::SD_OP_INTERSECT:
					dest = sdIntersection(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;
				case sdPrimitive::SD_OP_DIFFERENCE:
					dest = sdDifference(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;				
				}				
			}
			else {
				switch (i.optype) {
				case sdPrimitive::SD_OP_UNION:
					dest = sdUnion(r[i.addr.regobj.reg], sdPrimitiveDistance(pos, i.addr.regobj.prim));
					break;
				case sdPrimitive::SD_OP_BLEND:
					dest = sdSmoothmin(r[i.addr.regobj.reg], sdPrimitiveDistance(pos, i.addr.regobj.prim), i.addr.regobj._p0);
					break;
				case sdPrimitive::SD_OP_INTERSECT:
					dest = sdIntersection(r[i.addr.regobj.reg], sdPrimitiveDistance(pos, i.addr.regobj.prim));
					break;
				case sdPrimitive::SD_OP_DIFFERENCE:
					dest = sdDifference(r[i.addr.regobj.reg], sdPrimitiveDistance(pos, i.addr.regobj.prim));
					break;
				}				
			}
		}

		return r[0];
	}	

	
	
	template <size_t regNum = 4>
	__DISTFUN_T_ vec3 sdGetNearestPoint(const vec3 & pos, const sdProgramStatic * programPtr, float dx = 0.001f) {
		float d = sdDistanceAtPos<regNum>(pos, programPtr);
		vec3 N = sdNormal(pos, dx, sdDistanceAtPos<regNum>, programPtr);
		return pos - d*N;
	}
		

	

	template <size_t regNum = 4>
	__DISTFUN_T_ float sdVolumeInBounds(
		const sdAABB & bounds,
		const sdProgramStatic * programPtr,
		int curDepth,
		int maxDepth
	) {
		const vec3 pt = bounds.center();
		const float d = sdDistanceAtPos<regNum>(pt, programPtr);


		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return 0.0f;
						
			//Cell completely inside, return volume of bounds	
			return bounds.volume();
		}

		//Nearest surface is within bounds, subdivide
		float volume = 0.0f;
		for (auto i = 0; i < 8; i++) {
			volume += sdVolumeInBounds<regNum>(bounds.getOctant(i), programPtr, curDepth + 1, maxDepth);
		}

		return volume;
	}

	
	__DISTFUN__ float sdIntersectionVolume(
		const sdAABB & bounds,
		const sdPrimitive & a,
		const sdPrimitive & b,
		int curDepth,
		int maxDepth
	) {
		const vec3 pt = bounds.center();
		const float da = sdPrimitiveDistance(pt, a);
		const float db = sdPrimitiveDistance(pt, b);
		const float d = sdIntersection(da, db);


		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return 0.0f;

			//Cell completely inside, return volume of bounds	
			return bounds.volume();
		}

		//Nearest surface is within bounds, subdivide
		float volume = 0.0f;
		for (auto i = 0; i < 8; i++) {
			volume += sdIntersectionVolume(bounds.getOctant(i), a,b, curDepth + 1, maxDepth);
		}

		return volume;
	}


	
	__DISTFUN__ vec4 sdElasticity(
		const sdAABB & bounds,
		const sdPrimitive & a, 
		const sdPrimitive & b,
		float k,
		int curDepth,
		int maxDepth
	){
		const vec3 pt = bounds.center();
		const float da = sdPrimitiveDistance(pt, a);
		const float db = sdPrimitiveDistance(pt, b);
		const float d = sdIntersection(db, da);

		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return vec4(0.0f);

			//Cell completely inside			

			//Distance to nearest non-penetrating surface of a
			const float L = sdDifference(da, db);								

			//Normal to nearest non-penetrating surface of a
			const vec3 N = sdNormal(pt, 0.0001f, sdPrimitiveDifference, a, b);
			const float magnitude = 0.5f * k * (L*L);
			const vec3 U = magnitude * N;
			return vec4(U, magnitude);
		}

		//Nearest surface is within bounds, subdivide
		vec4 elasticity = vec4(0.0f);
		float childK = k / 5.0f; 
		for (auto i = 0; i < 8; i++) {
			elasticity += sdElasticity(bounds.getOctant(i), a, b, childK, curDepth + 1, maxDepth);
		}

		return elasticity;



	}


/*////////////////////////////////////////////////////////////////////////////////////
	Raymarching
////////////////////////////////////////////////////////////////////////////////////*/

	struct sdMarchState {
		bool hit;
		vec3 pos;
		vec3 normal;		
		float dist;
	};

	__DISTFUN__ sdMarchState sdMarch(		
		const sdProgramStatic * programPtr,
		const sdRay & ray,
		float precision,
		float maxDist
	) {
		float curDist = 0.0f;
		float px = precision;
		
		
		vec3 curPos = ray.origin;

		while (curDist < maxDist) {			
			
			float t = sdDistanceAtPos<4>(curPos, programPtr);
			if (t == FLT_MAX)
				break;
			
			if (t < precision) {
				sdMarchState mstate;
				mstate.hit = true;
				mstate.pos = curPos;
				mstate.normal = sdNormal(curPos, 2 * px, sdDistanceAtPos<4>, programPtr);
				mstate.dist = curDist;
				return mstate;
			}

			curDist += t;
			curPos += t*ray.dir;
		};

		sdMarchState mstate;
		mstate.hit = false;
		mstate.pos = curPos;

		return mstate;

	}



/*////////////////////////////////////////////////////////////////////////////////////
	Integration
////////////////////////////////////////////////////////////////////////////////////*/

template <typename ResType>
using sdIntegrateFunc = ResType(*)(const vec3 &pt, float d, const sdAABB & bounds);


__DISTFUN__ float sdIntegrateVolume(const vec3 & pt, float d, const sdAABB & bounds){
	if(d > 0.0f) return 0.0f;
	return bounds.volume();
}


__DISTFUN__ vec4 sdIntegrateCenterOfMass(const vec3 & pt, float d, const sdAABB & bounds){
	if(d > 0.0f) return vec4(0.0f);
	float V = bounds.volume();
	return vec4(V * pt, V);
}


struct sdIntertiaTensor{
	float s1;
	float sx,sy,sz;
	float sxy,syz,sxz;
	float sxx,syy,szz;

	__DISTFUN__ sdIntertiaTensor operator + (const sdIntertiaTensor & o) const{
		sdIntertiaTensor r;
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for(auto i =0 ; i < 10; i++)
			((float*)&r)[i] = ((const float*)this)[i] + ((const float*)&o)[i];		

		return r;
	}

	__DISTFUN__ sdIntertiaTensor(float val = 0.0f) :
		s1(val), 
		sx(val), sy(val), sz(val),
		sxy(val), syz(val), sxz(val),
		sxx(val), syy(val), szz(val)
	{		
	}

};

__DISTFUN__ sdIntertiaTensor sdIntegrateIntertia(const vec3 & pt, float d, const sdAABB & bounds){
	if (d > 0.0f) return sdIntertiaTensor(0.0f);

	sdIntertiaTensor s;

	vec3 V = bounds.diagonal();
	vec3 V2max = bounds.max * bounds.max;
	vec3 V2min = bounds.min * bounds.min;
	vec3 V2 = V2max - V2min;

	vec3 V3max = bounds.max * V2max;
	vec3 V3min = bounds.min * V2min;
	vec3 V3 = V3max - V3min;
	
	s.s1 = V.x * V.y * V.z;	
	s.sx = 0.5f * V2.x * V.y * V.z;
	s.sy = 0.5f * V.x * V2.y * V.z;
	s.sz = 0.5f * V.x * V.y * V2.z;

	s.sxy = 0.25f * V2.x * V2.y * V.z;
	s.syz = 0.25f * V.x * V2.y * V2.z;
	s.sxz = 0.25f * V2.x * V.y * V2.z;

	s.sxx = (1.0f / 3.0f) * V3.x * V.y * V.z;
	s.syy = (1.0f / 3.0f) * V.x * V3.y * V.z;
	s.szz = (1.0f / 3.0f) * V.x * V.y * V3.z;

	return s;
}





template <typename ResType, class F>
__DISTFUN_T_ ResType sdIntegrateProgramRecursive(
	const sdProgramStatic * programPtr,
	const sdAABB & bounds,
	F func,
	int maxDepth = 5,
	int curDepth = 0
){

	vec3 pt = bounds.center();
	float d = sdDistanceAtPos(pt, programPtr);

	if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(bounds.diagonal())) {
		return func(pt, d, bounds);
	}


	ResType r = ResType(0);
	for (auto i = 0; i < 8; i++) {
		r = r + sdIntegrateProgramRecursive<ResType>(
			programPtr,
			bounds.getOctant(i),
			func,
			maxDepth,
			curDepth + 1
		);
	}

	return r;
}

/*
	Same functionality as sdIntegrateProgramRecursive
	but with explicit stack.
	Faster on CUDA as it reduces thread divergence.
*/
template <int stackSize = 8, typename ResType, class F>
__DISTFUN_T_ ResType sdIntegrateProgramRecursiveExplicit(
	const sdProgramStatic * programPtr,
	const sdAABB & bounds,
	F func,
	int maxDepth = 5,
	int curDepth = 0
) {

	ResType result = ResType(0);

	struct StackVal {
		sdAABB bounds;
		unsigned char i;
	};

	int stackDepth = 1;
	StackVal _stack[stackSize];
	StackVal * stackTop = &_stack[0];
	stackTop->bounds = bounds;
	stackTop->i = 0;

	while (stackDepth > 0) {

		StackVal & val = *stackTop;

		if (val.i == 8) {
			//Pop
			stackTop--;
			stackDepth--;
			continue;
		}

		const sdAABB curBounds = val.bounds.getOctant(val.i);
		val.i++;

		const vec3 pt = curBounds.center();
		const float d = sdDistanceAtPos<4>(pt, programPtr);

		if (curDepth + stackDepth > maxDepth || d * d >= 0.5f * 0.5f * glm::length2(curBounds.diagonal())) {
			result = result + func(pt, d, curBounds);
		}
		else {
			//Push			
			StackVal & newVal = *(++stackTop);
			newVal.bounds = curBounds;
			newVal.i = 0;

			stackDepth++;
		}
	}

	return result;
}



template <typename ResType, class F, class ... Args>
__DISTFUN_T_ ResType sdIntegrateProgramGrid(
	const sdProgramStatic * programPtr,
	const sdAABB & bounds,	
	ivec3 gridSpec,
	const F & func,
	Args ... args
){
	ResType r = ResType(0.0f);
	for (auto z = 0; z < gridSpec.z; z++) {
		for (auto y = 0; y < gridSpec.y; y++) {
			for (auto x = 0; x < gridSpec.x; x++) {
				const sdAABB cellBounds = bounds.getSubGrid(gridSpec, { x,y,z });
				r = r + func(programPtr, cellBounds, args ...);				
			}
		}
	}
	return r;
}



/*////////////////////////////////////////////////////////////////////////////////////
	CUDA Kernels
////////////////////////////////////////////////////////////////////////////////////*/
#ifdef DISTFUN_ENABLE_CUDA

#define DISTFUN_VOLUME_VOX								\
ivec3 vox = ivec3(								\
		blockIdx.x * blockDim.x + threadIdx.x,	\
		blockIdx.y * blockDim.y + threadIdx.y,	\
		blockIdx.z * blockDim.z + threadIdx.z	\
	);	
#define DISTFUN_VOLUME_VOX_GUARD(res)					\
	DISTFUN_VOLUME_VOX									\
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z)	\
	return;		

/*
	Kernel integrating over bounds using func, with base resolution given by gridSpec.
	Integrates from curDepth recursively until maxDepth 		
	Each grid cell result is saved into result, which should be an array allocated to
		gridSpec.x*gridSpec.y*gridSpec.z size
*/
template <int stackSize = 8, typename ResType, sdIntegrateFunc<ResType> func>
__global__ void sdIntegrateKernel(
	const sdProgramStatic * programPtr,
	sdAABB bounds,
	ivec3 gridSpec,
	ResType * result,
	int maxDepth,
	int curDepth
) {
	DISTFUN_VOLUME_VOX_GUARD(gridSpec);	

	const ResType res = sdIntegrateProgramRecursiveExplicit<stackSize, ResType>(
		programPtr,
		bounds.getSubGrid(gridSpec, vox),
		func,
		maxDepth,
		curDepth
		);

	const ivec3 stride = { 1, gridSpec.x, gridSpec.x * gridSpec.y };
	const size_t index = sdLinearIndex(stride, vox);
	result[index] = res;
}


#endif



}
/*////////////////////////////////////////////////////////////////////////////////////
	CPU code definition
////////////////////////////////////////////////////////////////////////////////////*/


#ifdef DISTFUN_IMPLEMENTATION

#include <unordered_map>
#include <stack>
#include <functional>
#include <cstring>


namespace distfun {

	bool sdIsLeaf(const sdTreeNode & node) {
		return !node.children[0] && !node.children[1];
	}

	int sdTreeDepth(const sdTreeNode & node) {
		int depth = 1;
		if (node.children[0])
			depth = sdTreeDepth(*node.children[0]);
		if (node.children[1])
			depth = glm::max(depth, sdTreeDepth(*node.children[1]));
		return depth;
	}



#define LABEL_LEFT_SIDE 0
#define LABEL_RIGHT_SIDE 1
	int labelTreeNode(const sdTreeNode * node, int side, std::unordered_map<const sdTreeNode*, int> & labels) {

		if (!node) return 0;

		if (sdIsLeaf(*node)) {
			//Left node (0), label 1; Right side (1) -> label 0
			int newLabel = 1 - side;
			labels[node] = newLabel;
			return newLabel;
		}

		int labelLeft = labelTreeNode(node->children[0].get(), LABEL_LEFT_SIDE, labels);
		int labelRight = labelTreeNode(node->children[1].get(), LABEL_RIGHT_SIDE, labels);

		if (labelLeft == labelRight) {
			labels[node] = labelLeft + 1;
		}
		else {
			labels[node] = glm::max(labelLeft, labelRight);
		}
		return labels[node];
	}


	sdProgram sdCompile(const sdTreeNode & node) {

		//Sethi-Ullman algorithm
		//https://www.cse.iitk.ac.in/users/karkare/cs335/lectures/19SethiUllman.pdf


		std::vector<sdInstruction> instructions;
		std::unordered_map<const sdTreeNode*, int> labels;


		int regs = labelTreeNode(&node, 0, labels);		
		int N = regs;

		std::stack<sdInstruction::RegIndex> rstack;
		std::stack<sdInstruction::RegIndex> tstack;

		auto swapTop = [](std::stack<sdInstruction::RegIndex> & stack) {
			int top = stack.top();
			stack.pop();
			int top2 = stack.top();
			stack.pop();
			stack.push(top);
			stack.push(top2);
		};

		for (int i = 0; i < regs; i++) {
			rstack.push(regs - i - 1);
			tstack.push(regs - i - 1);
		}

		std::function<void(const sdTreeNode * node, int side)> genCode;

		genCode = [&](const sdTreeNode * node, int side) -> void {
			assert(node);

			auto & leftChild = node->children[LABEL_LEFT_SIDE];
			auto & rightChild = node->children[LABEL_RIGHT_SIDE];


			if (sdIsLeaf(*node) && side == LABEL_LEFT_SIDE) {
				sdInstruction i(sdInstruction::OBJ);
				i.optype = node->primitive.type;
				i.addr.obj.prim = node->primitive;
				i.regTarget = rstack.top();
				instructions.push_back(i);
			}
			else if (sdIsLeaf(*rightChild)) {
				//Generate instructions for left subtree first
				genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

				sdInstruction i(sdInstruction::REG_OBJ);
				i.optype = node->primitive.type;

				//special case for blend param
				if (i.optype == sdPrimitive::SD_OP_BLEND) {
					i.addr.regobj._p0 = node->primitive.params.blend.k;
				}

				i.addr.regobj.reg = rstack.top();
				i.addr.regobj.prim = node->children[LABEL_RIGHT_SIDE]->primitive;
				i.regTarget = rstack.top();
				instructions.push_back(i);
			}
			//Case 3. left child requires less than N registers
			//else if (labels[leftChild.get()] < N) {
			else if (labels[leftChild.get()] < labels[rightChild.get()] && labels[leftChild.get()] < N) {
				// Right child goes into next to top register
				swapTop(rstack);
				//Evaluate right child
				genCode(node->children[LABEL_RIGHT_SIDE].get(), LABEL_RIGHT_SIDE);

				sdInstruction::RegIndex R = rstack.top();
				rstack.pop();

				//Evaluate left child
				genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

				sdInstruction i(sdInstruction::REG_REG);
				i.optype = node->primitive.type;
				i.addr.regreg.reg[0] = rstack.top();
				i.addr.regreg.reg[1] = R;

				//special case for blend param
				if (i.optype == sdPrimitive::SD_OP_BLEND) {
					i.addr.regreg._p0 = node->primitive.params.blend.k;
				}

				i.regTarget = rstack.top();
				instructions.push_back(i);

				rstack.push(R);
				swapTop(rstack);
			}
			//else if (labels[rightChild.get()] < N) {
			//Case 4
			else if (labels[rightChild.get()] <= labels[leftChild.get()] && labels[rightChild.get()] < N) {
				//Evaluate left child
				genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

				sdInstruction::RegIndex R = rstack.top();
				rstack.pop();

				//Evaluate right child
				genCode(node->children[LABEL_RIGHT_SIDE].get(), LABEL_RIGHT_SIDE);

				sdInstruction i(sdInstruction::REG_REG);
				i.optype = node->primitive.type;
				i.addr.regreg.reg[0] = R;
				i.addr.regreg.reg[1] = rstack.top();
				i.regTarget = R;
				instructions.push_back(i);

				rstack.push(R);
			}
			//Shouldn't happen, uses temporary stack (in gmem)
			else {
				
			}

		};


		genCode(&node, 0);

		sdProgram p;
		p.instructions = std::move(instructions);
		p.registers = regs;
		p.instructionCount = static_cast<int>(p.instructions.size());

		return p;
	}

	

	const sdProgramStatic * sdCommitCPU(void * destination, const sdProgram & program) {
		sdCommit(destination, program, std::memcpy);
		return sdCastProgramStatic(destination);
	}


#ifdef DISTFUN_ENABLE_CUDA
	
	const sdProgramStatic * sdCommitGPU(void * destination, const sdProgram & program) {
		const auto cpyGlobal = [](void * dest, const void * src, size_t size) {
			cudaMemcpy(dest, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
		};
		sdCommit(destination, program, cpyGlobal);
		return sdCastProgramStatic(destination);
	}
#endif



}

#endif

#endif