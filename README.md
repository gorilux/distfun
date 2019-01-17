# distfun

Header-only signed distance field/function library.
Inspired by raymarching work by [Inigo Quilez](https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm)

### Features:
  * Distance functions for basic primitives
    * Sphere, Box, Cylinder, etc.
  * Constructive Solid Geometry (CSG) tree
    * Operations: Union, Intersection, Difference, Smooth Union
  * Conversion of CSG tree to a custom program representation
    * Set of instructions operating on registers and distance functions
    * Uses the Sethi-Ullman algorithm    
    * Avoids expensive CSG tree traversal and minimizes required stack memory
  * CUDA compatible
    * use `#define DISTFUN_ENABLE_CUDA`
    * programs can often fit into `__constant__` memory, yielding significant speedup
  * Supports evaluation of distance, normal and nearest point at any 3D location
  * Supports raymarching
   
   
### TODO:
  - [ ] Volume calculation by octree subdivision
  - [ ] 3D Grid SDF for custom objects
  - [ ] Example code
  - [ ] Add primitives: torus, prism, capsule, 
  - [ ] Add operations: elongation, revolution, extrusion, displacement, twist, bend
  - [ ] primitive identification, materials
  - [ ] rendering related functions: shadow, ambient occlusion
    



