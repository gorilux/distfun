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
	#define __DISTFUN__ __host__ __device__
#else
	#define __DISTFUN__ inline
#endif

#define DISTFUN_ARRAY_PLACEHOLDER 1

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <array>
#include <memory>
#include <vector>

#ifdef DISTFUN_IMPLEMENTATION
#include <unordered_map>
#include <stack>
#include <functional>
#include <cstring>
#endif

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

	struct Ray {
		vec3 origin;
		vec3 dir;
	};

	struct AABB {
		vec3 min;
		vec3 max;
		vec3 diagonal() const { return max - min; }
		vec3 center() const { return (min + max) * 0.5f; }

		vec3 sideX() const { return vec3(max.x - min.x, 0, 0); }
		vec3 sideY() const { return vec3(0, max.y - min.y, 0); }
		vec3 sideZ() const { return vec3(0, 0, max.z - min.z); }

		vec3 corner(int index) const {
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

		AABB getOctant(int octant) const
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
			return AABB();
		}

		AABB getSubGrid(const ivec3 & gridSize, const ivec3 & gridIndex) const {
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

	};


/*////////////////////////////////////////////////////////////////////////////////////
	Primitive parameters		
////////////////////////////////////////////////////////////////////////////////////*/
	struct PlaneParams { char _empty; };
	struct SphereParams { float radius; };
	struct BoxParams { vec3 size; };
	struct CylinderParams { float radius; float height; };
	struct BlendParams { float k; };
	struct EllipsoidParams { vec3 size; };
	struct ConeParam { float h; float r1; float r2; };
	struct GridParam { void * ptr; ivec3 size; };
	
/*////////////////////////////////////////////////////////////////////////////////////
	Primitive 
////////////////////////////////////////////////////////////////////////////////////*/
	struct Primitive {

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
			PlaneParams plane;
			SphereParams sphere;
			BoxParams box;
			CylinderParams cylinder;
			BlendParams blend;
			EllipsoidParams ellipsoid;
			ConeParam cone;
			GridParam grid;
		};

		float rounding;
		Type type;
		mat4 invTransform;
		Params params;
	};	

	

/*////////////////////////////////////////////////////////////////////////////////////
	Primitive Distance Functions
////////////////////////////////////////////////////////////////////////////////////*/
	
	__DISTFUN__ float distPlaneHorizontal(const vec3 & p, const PlaneParams & param)
	{
		return p.y;
	}

	__DISTFUN__ float distSphere(const vec3 & p, const SphereParams & param)
	{
		return glm::length(p) - param.radius;
	}

	__DISTFUN__ float distBox(const vec3 & p, const BoxParams & param)
	{
		vec3 d = glm::abs(p) - param.size;
		return glm::min(glm::max(d.x, glm::max(d.y, d.z)), 0.0f) + glm::length(glm::max(d, vec3(0.0f)));
	}

	__DISTFUN__ float distCylinder(const vec3 & p, const CylinderParams & param)
	{
		vec2 d = abs(vec2(glm::length(vec2(p.x, p.z)), p.y)) - vec2(param.radius, param.height);
		return glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, 0.0f));
	}

	__DISTFUN__ float distEllipsoid(const vec3 & p, const EllipsoidParams & param)
	{
		const auto & r = param.size;
		float k0 = glm::length(vec3(p.x / r.x, p.y / r.y, p.z / r.z));
		float k1 = glm::length(vec3(p.x / (r.x*r.x), p.y / (r.y*r.y), p.z / (r.z*r.z)));
		return k0*(k0 - 1.0f) / k1;
	}

	__DISTFUN__ float distCone(const vec3 & p, const ConeParam & param)
	{
		vec2 q = vec2(glm::length(vec2(p.x, p.z)), p.y);

		vec2 k1 = vec2(param.r2, param.h);
		vec2 k2 = vec2(param.r2 - param.r1, 2.0f*param.h);
		vec2 ca = vec2(q.x - glm::min(q.x, (q.y < 0.0f) ? param.r1 : param.r2), abs(q.y) - param.h);
		vec2 cb = q - k1 + k2*glm::clamp(glm::dot(k1 - q, k2) / glm::dot(k2, k2), 0.0f, 1.0f);
		float s = (cb.x < 0.0f && ca.y < 0.0f) ? -1.0f : 1.0f;
		return s*glm::sqrt(glm::min(glm::dot(ca, ca), glm::dot(cb, cb)));

	}

	__DISTFUN__ float distUnion(float a, float b) {
		return glm::min(a, b);
	}

	__DISTFUN__ float distIntersection(float a, float b) {
		return glm::max(a, b);
	}

	__DISTFUN__ float distDifference(float a, float b) {
		return glm::max(-a, b);
	}

	__DISTFUN__ float distRound(float dist, float r) {
		return dist - r;
	}

	__DISTFUN__ float distSmoothmin(float a, float b, float k) {
		float h = glm::clamp(0.5f + 0.5f*(a - b) / k, 0.0f, 1.0f);
		return glm::mix(a, b, h) - k*h*(1.0f - h);
	}




/*////////////////////////////////////////////////////////////////////////////////////
	Transformations and primitive distance
////////////////////////////////////////////////////////////////////////////////////*/
	
	template <class Func, class ... Args>
	__DISTFUN__ float transform(
		const vec3 & p,
		const mat4 & invTransform,
		Func f,
		Args ... args
	) {
		return f(vec3(invTransform * vec4(p.x, p.y, p.z, 1.0f)), args ...);
	}

	__DISTFUN__ vec3 transformPos(
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
	__DISTFUN__ vec3 distNormal(vec3 pos, float eps, Func f, Args ... args)
	{
		return normalize(vec3(
			f(pos + vec3(eps, 0, 0), args ...) - f(pos - vec3(eps, 0, 0), args ...),
			f(pos + vec3(0, eps, 0), args ...) - f(pos - vec3(0, eps, 0), args ...),
			f(pos + vec3(0, 0, eps), args ...) - f(pos - vec3(0, 0, eps), args ...)));

	}
	

	__DISTFUN__  float distPrimitive(
		const vec3 & pos,
		const Primitive & prim
	) {

		const vec3 tpos = transformPos(pos, prim.invTransform);

		switch (prim.type) {
		case Primitive::SD_SPHERE:
			return  distSphere(tpos, prim.params.sphere);
		case Primitive::SD_ELLIPSOID:
			return  distEllipsoid(tpos, prim.params.ellipsoid);
		case Primitive::SD_CONE:
			return  distCone(tpos, prim.params.cone);
		case Primitive::SD_BOX:
			return distRound(distBox(tpos, prim.params.box), prim.rounding);
		case Primitive::SD_CYLINDER:
			return 	distRound(distCylinder(tpos, prim.params.cylinder), prim.rounding);
		case Primitive::SD_PLANE:
			return distPlaneHorizontal(tpos, prim.params.plane);
		}
		return FLT_MAX;
	}


	__DISTFUN__ float distPrimitiveDifference(const vec3 & pos, const Primitive & a, const Primitive & b) {
		return distDifference(distPrimitive(pos, a), distPrimitive(pos, b));
	}
	
	__DISTFUN__ vec3 getNearestPoint(const vec3 & pos, const Primitive & prim, float dx = 0.001f) {
		float d = distPrimitive(pos, prim);
		vec3 N = distNormal(pos, dx, distPrimitive, prim);
		return pos - d*N;
	}

	__DISTFUN__  AABB primitiveBounds(		
		const Primitive & prim,
		float pollDistance,
		float dx = 0.001f
	){
		if (prim.type == Primitive::SD_ELLIPSOID) {
			vec3 radius = prim.params.ellipsoid.size;
			mat4 transform = glm::inverse(prim.invTransform);
			AABB bbOrig = {-radius, radius};
			AABB bb = { vec3(FLT_MAX), vec3(-FLT_MAX) };
			for (auto i = 0; i < 8; i++) {
				vec3 pt = bbOrig.corner(i);
				pt = vec3(transform * vec4(pt, 1.0f));
				bb.min = glm::min(pt, bb.min);
				bb.max = glm::max(pt, bb.max);
			}
			return bb;
		}
		else {
			return { vec3(0),vec3(0) };
		}
	}

	
/*////////////////////////////////////////////////////////////////////////////////////
	Tree  (CPU only)
////////////////////////////////////////////////////////////////////////////////////*/
	struct TreeNode {
		Primitive primitive;
		std::array<std::unique_ptr<TreeNode>, 2> children;						
	};

	bool isLeaf(const TreeNode & node);
	int treeDepth(const TreeNode & node);	

/*////////////////////////////////////////////////////////////////////////////////////
	Program
////////////////////////////////////////////////////////////////////////////////////*/
	

	//Evaluation order
	struct Instruction {
		enum Type {
			REG_OBJ,
			REG_REG,
			OBJ
		};
		
		Instruction(Type type = OBJ) :
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
			Primitive prim;
			float _p0;
		};

		/*
			Register DEST receives result of Primitive prim.			
			DEST <- prim
		*/
		struct AddrObj {
			Primitive prim;
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
			AddrRegObj regobj;
			AddrObj obj;
			AddrRegReg regreg;
		};

		Primitive::Type optype;
		Type itype;
		RegIndex regTarget;
		Addr addr;

	};

	struct DistProgram {
		int instructionCount;
		int registers;
		std::vector<Instruction> instructions;

		size_t staticSize() const {
			return 2 * sizeof(int) + sizeof(Instruction)*instructions.size();
		}
	};
	
	struct DistProgramStatic{
		DistProgramStatic(const DistProgramStatic &) = delete;
		DistProgramStatic & operator=(const DistProgramStatic &) = delete;

		int instructionCount;
		int registers;
		Instruction instructions[DISTFUN_ARRAY_PLACEHOLDER]; //In-place pointer for variable size
		
	};




/*////////////////////////////////////////////////////////////////////////////////////
	Tree To Program conversion
////////////////////////////////////////////////////////////////////////////////////*/
	
	DistProgram compileProgram(const TreeNode & node);

	
	template <class CopyFun>
	void commitProgram(void * destination, const DistProgram & program, CopyFun copyFunction){
		DistProgramStatic *dst = reinterpret_cast<DistProgramStatic*>(destination);		
		copyFunction(&dst->instructionCount, &program.instructionCount, sizeof(int));
		copyFunction(&dst->registers, &program.registers, sizeof(int));
		copyFunction(&dst->instructions, program.instructions.data(), sizeof(Instruction) * program.instructionCount);
	}

	void commitProgramCPU(void * destination, const DistProgram & program);

#ifdef DISTFUN_ENABLE_CUDA
	void commitProgramGPU(void * destination, const DistProgram & program);
#endif



/*////////////////////////////////////////////////////////////////////////////////////
	Program evaluation
////////////////////////////////////////////////////////////////////////////////////*/

	template <size_t regNum = 4>
	__DISTFUN__ float distanceAtPos(const vec3 & pos, const DistProgramStatic * programPtr) {

		//Registers
		float r[regNum];		

		if (programPtr->instructionCount == 0)
			return FLT_MAX;
		
		//Step through each instruction
		for (auto pc = 0; pc < programPtr->instructionCount; pc++) {

			const Instruction & i = programPtr->instructions[pc];
			float & dest = r[i.regTarget];

			if (i.itype == Instruction::OBJ) {
				dest = distPrimitive(pos, i.addr.obj.prim);
			}
			else if (i.itype == Instruction::REG_REG) {
				switch (i.optype) {
				case Primitive::SD_OP_UNION:
					dest = distUnion(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;
				case Primitive::SD_OP_BLEND:
					dest = distSmoothmin(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]], i.addr.regreg._p0);
					break;
				case Primitive::SD_OP_INTERSECT:
					dest = distIntersection(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;
				case Primitive::SD_OP_DIFFERENCE:
					dest = distDifference(r[i.addr.regreg.reg[0]], r[i.addr.regreg.reg[1]]);
					break;				
				}				
			}
			else {
				switch (i.optype) {
				case Primitive::SD_OP_UNION:
					dest = distUnion(r[i.addr.regobj.reg], distPrimitive(pos, i.addr.regobj.prim));
					break;
				case Primitive::SD_OP_BLEND:
					dest = distSmoothmin(r[i.addr.regobj.reg], distPrimitive(pos, i.addr.regobj.prim), i.addr.regobj._p0);
					break;
				case Primitive::SD_OP_INTERSECT:
					dest = distIntersection(r[i.addr.regobj.reg], distPrimitive(pos, i.addr.regobj.prim));
					break;
				case Primitive::SD_OP_DIFFERENCE:
					dest = distDifference(r[i.addr.regobj.reg], distPrimitive(pos, i.addr.regobj.prim));
					break;
				}				
			}
		}

		return r[0];
	}	
	
	template <size_t regNum = 4>
	__DISTFUN__ vec3 getNearestPoint(const vec3 & pos, const DistProgramStatic * programPtr, float dx = 0.001f) {
		float d = distanceAtPos<regNum>(pos, programPtr);
		vec3 N = distNormal(pos, dx, distanceAtPos<regNum>, programPtr);
		return pos - d*N;
	}
		

	

	template <size_t regNum = 4>
	float volumeInBounds(
		const AABB & bounds,
		const DistProgramStatic * programPtr,
		int curDepth,
		int maxDepth
	) {
		const vec3 pt = bounds.center();
		const float d = distanceAtPos<regNum>(pt, programPtr);


		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return 0.0f;
						
			//Cell completely inside, return volume of bounds	
			return diagonal.x * diagonal.y * diagonal.z; 
		}

		//Nearest surface is within bounds, subdivide
		float volume = 0.0f;
		for (auto i = 0; i < 8; i++) {
			volume += volumeInBounds(bounds.getOctant(i), programPtr, curDepth + 1, maxDepth);
		}

		return volume;
	}


	
	__DISTFUN__ vec3 primitiveElasticity(
		const AABB & bounds,
		const Primitive & a, 
		const Primitive & b,
		float k,
		int curDepth,
		int maxDepth
	){
		const vec3 pt = bounds.center();
		const float da = distPrimitive(pt, a);
		const float db = distPrimitive(pt, b);
		const float d = distIntersection(db, da);

		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return vec3(0.0f);

			//Cell completely inside			

			//Distance to nearest non-penetrating surface of a
			const float L = distDifference(da, db);								

			//Normal to nearest non-penetrating surface of a
			const vec3 N = distNormal(pt, 0.0001f, distPrimitiveDifference, a, b);
			const vec3 U = 0.5f * k * (L*L) * N;
			return U;
		}

		//Nearest surface is within bounds, subdivide
		vec3 elasticity = vec3(0.0f);
		float childK = k / 5.0f; 
		for (auto i = 0; i < 8; i++) {
			elasticity += primitiveElasticity(bounds.getOctant(i), a, b, childK, curDepth + 1, maxDepth);
		}

		return elasticity;



	}


/*////////////////////////////////////////////////////////////////////////////////////
	Raymarching
////////////////////////////////////////////////////////////////////////////////////*/

	struct MarchState {
		bool hit;
		vec3 pos;
		vec3 normal;		
		float dist;
	};

	__DISTFUN__ MarchState march(		
		const DistProgramStatic * programPtr,
		const Ray & ray,
		float precision,
		float maxDist
	) {
		float curDist = 0.0f;
		float px = precision;
		
		int depth = 0;

		vec3 curPos = ray.origin;

		while (curDist < maxDist) {			
			
			float t = distanceAtPos<4>(curPos, programPtr);
			if (t == FLT_MAX)
				break;
			
			if (t < precision) {
				MarchState mstate;
				mstate.hit = true;
				mstate.pos = curPos;
				mstate.normal = distNormal(curPos, 2 * px, distanceAtPos<4>, programPtr);
				mstate.dist = curDist;
				return mstate;
			}

			curDist += t;
			curPos += t*ray.dir;
		};

		MarchState mstate;
		mstate.hit = false;
		mstate.pos = curPos;

		return mstate;

	}
	

/*////////////////////////////////////////////////////////////////////////////////////
	CPU code definition
////////////////////////////////////////////////////////////////////////////////////*/


#ifdef DISTFUN_IMPLEMENTATION
bool isLeaf(const TreeNode & node) {
	return !node.children[0] && !node.children[1];
}

int treeDepth(const TreeNode & node) {
	int depth = 1;
	if (node.children[0])
		depth = treeDepth(*node.children[0]);
	if (node.children[1])
		depth = glm::max(depth, treeDepth(*node.children[1]));
	return depth;
}



#define LABEL_LEFT_SIDE 0
#define LABEL_RIGHT_SIDE 1
int labelTreeNode(const TreeNode * node, int side, std::unordered_map<const TreeNode*,int> & labels) {

	if (!node) return 0;

	if (isLeaf(*node)) {
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


DistProgram compileProgram(const TreeNode & node) {

	//Sethi-Ullman algorithm
	//https://www.cse.iitk.ac.in/users/karkare/cs335/lectures/19SethiUllman.pdf

		
	std::vector<Instruction> instructions;
	std::unordered_map<const TreeNode*, int> labels;


	int regs = labelTreeNode(&node, 0, labels);
	int k = treeDepth(node);
	int N = k;

	std::stack<Instruction::RegIndex> rstack;
	std::stack<Instruction::RegIndex> tstack;

	auto swapTop = [](std::stack<Instruction::RegIndex> & stack) {		
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

	std::function<void(const TreeNode * node, int side)> genCode;

	genCode = [&](const TreeNode * node, int side) -> void {
		assert(node);

		auto & leftChild = node->children[LABEL_LEFT_SIDE];
		auto & rightChild = node->children[LABEL_RIGHT_SIDE];


		if (isLeaf(*node) && side == LABEL_LEFT_SIDE) {
			Instruction i(Instruction::OBJ);
			i.optype = node->primitive.type;
			i.addr.obj.prim = node->primitive;
			i.regTarget = rstack.top();
			instructions.push_back(i);
		}
		else if (isLeaf(*rightChild)) {
			//Generate instructions for left subtree first
			genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

			Instruction i(Instruction::REG_OBJ);
			i.optype = node->primitive.type;

			//special case for blend param
			if (i.optype == Primitive::SD_OP_BLEND) {
				i.addr.regobj._p0 = node->primitive.params.blend.k;
			}

			i.addr.regobj.reg = rstack.top();
			i.addr.regobj.prim = node->children[LABEL_RIGHT_SIDE]->primitive;
			i.regTarget = rstack.top();
			instructions.push_back(i);
		}
		//Case 3. left child requires less than N registers
		else if (labels[leftChild.get()] < N) {
			// Right child goes into next to top register
			swapTop(rstack);
			//Evaluate right child
			genCode(node->children[LABEL_RIGHT_SIDE].get(), LABEL_RIGHT_SIDE);

			Instruction::RegIndex R = rstack.top();
			rstack.pop();

			//Evaluate left child
			genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

			Instruction i(Instruction::REG_REG);
			i.optype = node->primitive.type;
			i.addr.regreg.reg[0] = rstack.top();
			i.addr.regreg.reg[1] = R;

			//special case for blend param
			if (i.optype == Primitive::SD_OP_BLEND) {
				i.addr.regreg._p0 = node->primitive.params.blend.k;
			}

			i.regTarget = rstack.top();
			instructions.push_back(i);

			rstack.push(R);
			swapTop(rstack);
		}
		else if (labels[rightChild.get()] <= N) {
			//Evaluate right child
			genCode(node->children[LABEL_LEFT_SIDE].get(), LABEL_LEFT_SIDE);

			Instruction::RegIndex R = rstack.top();
			rstack.pop();

			//Evaluate left child
			genCode(node->children[LABEL_RIGHT_SIDE].get(), LABEL_RIGHT_SIDE);

			Instruction i(Instruction::REG_REG);
			i.optype = node->primitive.type;
			i.addr.regreg.reg[0] = R;
			i.addr.regreg.reg[1] = rstack.top();
			i.regTarget = R;
			instructions.push_back(i);
		}
		//Shouldn't happen, uses temporary stack (in gmem)
		else {	

		}

	};


	genCode(&node, 0);

	DistProgram p;
	p.instructions = std::move(instructions);
	p.registers = regs;
	p.instructionCount = p.instructions.size();

	return p;	
}

void commitProgramCPU(void * destination, const DistProgram & program) {
	commitProgram(destination, program, std::memcpy);
}


#ifdef DISTFUN_ENABLE_CUDA
void commitProgramGPU(void * destination, const DistProgram & program) {
	const auto cpyGlobal = [](void * dest, void * src, size_t size) {
		cudaMemcpy(dest, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	};
	commitProgram(destination, program, cpyGlobal);
}
#endif

#endif

}


#endif