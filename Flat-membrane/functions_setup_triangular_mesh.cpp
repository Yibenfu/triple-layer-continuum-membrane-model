#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <algorithm> // : random_shuffle
#include <armadillo>
#include <omp.h>
#pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )

using namespace std;
using namespace arma;

struct Vertex3Layers{
    mat inlayer;
    mat midlayer;
    mat outlayer;
};
struct Force3Layers{
    mat inlayer;
    mat midlayer;
    mat outlayer;
};

/*
struct Face3Layers{
    Mat<int> inlayer;
    Mat<int> midlayer;
    Mat<int> outlayer;
};
*/

struct SpontCurv3Layers{
    rowvec inlayer;
    rowvec outlayer;
};

struct IsInsertionpatch3Layers{
    vector<bool> inlayer;
    vector<bool> outlayer;
};

struct ElementS03Layers{
    rowvec inlayer;
    rowvec midlayer;
    rowvec outlayer;
};
struct ElementV03Layers{
    rowvec inlayer;
    rowvec midlayer;
    rowvec outlayer;
};

struct Param{
    bool   isFlatMembrane = true;
    int    currentStep;
    bool   includeDivTilt = false;
    // mesh parameters:
    double l;                                       // triangular side length, nm
    double sideX;                                   // rectangle sidelength x, nm
    double sideY;                                   // rectangle sidelength y, nm
    int    numface;                                 // number of faces on each layer of meshes.
    int    numvertex;                               // number of vertices on each layer of meshes.
    bool   isBoundaryFixed = false;                 // boundaries
    bool   isBoundaryPeriodic = true;
    bool   isBoundaryFree = false;
    Row<int> isBoundaryVertex;
    Row<int> isBoundaryFace;
    Row<int> isGhostVertex;
    Row<int> isGhostFace;
    vector<int> ghostPartner; 
    vector<vector<int>> freePartners;
    // mesh regularization
    bool   isNCGstucked = false;
    bool   usingNCG = true;
    bool   usingRpi = true;
    double k_regularization;                         // coefficient of the regulerization constraint, 
    double gama_shape = 0.2;                         // factor of shape deformation
    double gama_area = 0.2;                          // factor of size deformation
    bool   deformAreaOrShape = false;
    int    subDivideTimes = 5;                       // subdivision times for the irregular patches
    int    GaussQuadratureN = 2;                     // Gaussian quadrature integral 

    // membrane parameters: 
    bool   isBilayerModel = true;
    bool   isGlobalAreaConstraint = true;            // whether to use Global constraints for the area elasticity
    // out-layer (top layer)
    double kc_out  = 20.0*4.17/2.0;                  // pN.nm. bending modulus, out-monolayer 
    double kst_out  = 0.0;                           // pN.nm. splay-tilt modulus, out-layer
    double us_out  = 250.0/2.0;                      // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
    double Ktilt_out  = 90.0;                        // pN/nm = mN/m. tilt modulus
    double Kthick_out = 3.0*Ktilt_out;      // pN/nm, coefficient of the membrane thickness. penalty term
    double thickness_out = 4.0/2.0;                  // nm, out-monolayer thickness. 
    double c0out = 0.0;                              // spontaneous curvature of membrane, out-layer. Convex is positive
    double S0out;                                    //target area
    double Sout;                                     // area, total area 
    double H0C_out;                                  // curvature-modified height; target height
    double V0out;                                    // target volume of the sphere
    double Vout;                                     // volume of the sphere      


    // in-layer (bottom layer)
    double kc_in  = kc_out;                          // pN.nm. bending modulus, in-monolayer 
    double kst_in  = 0.0;                            // pN.nm. splay-tilt modulus, in-layer
    double us_in  = us_out;                          // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
    double Ktilt_in  = Ktilt_out;                    // pN/nm = mN/m. tilt modulus
    double Kthick_in = 3.0*Ktilt_in;
    double thickness_in = thickness_out;             // nm, in-monolayer thickness. 
    double c0in  = c0out;                            // spontaneous curvature of membrane, in-layer. Concave is positive
    double S0in;                                     //target area
    double Sin;                                      // area, total area 
    double H0C_in;                                  // curvature-modified height; target height
    double V0in;                                     // target volume of the sphere
    double Vin;                                      // volume of the sphere      
    double uv;                                       // coefficeint of the volume constraint, 0.5*uv*(dv)^2/v0;
    
    double Kthick_constraint;

    // insertion parameters:
    Mat<int> insertionpatch;
    IsInsertionpatch3Layers isInsertionPatch;
    double c0out_ins = 0.3;                          // spontaneous curvature of insertion, outer layer
    double c0in_ins = 0.3;
    double s_insert  = 2.0;                          // insertion area
    double insert_dH0 = 0.3;                         // equilibrium value of thickness decrease induced by the insertion, nm
    double K_insertShape   = 10.0*us_out;            // spring constant for insertion zones, to constraint the insertion shape
    double insertionShapeEdgeLength; 
    Row<int> IsinertionpatchAdjacent;
    double K_adjacentPatch = 0.0;                    // constant for the shape constraint on the patches adjacent around insertion 
    double Kthick_insertion;

    // system setup
    bool   duringStepsToIncreaseInsertDepth = false;
    vector<bool> isLocallyFinerFace;
    vector<bool> isLocallyFinerFaceMostEdge;
}; 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for mesh setup.

Mat<double> setvertex_Loop_scheme(double sidex, double sidey, double l){ // vertex position
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3.0)/2.0 * a; 
    int m = round(sidey/dy);      // y axis division
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    double lx = n * dx;
    double ly = m * dy;

    int nodenum = (n+1)*(m+1);
    mat vertex(nodenum,3); vertex.fill(0.0);
    #pragma omp parallel for
    for (int j = 0; j < m+1; j++){
        bool isEvenJ = false;
        if ( pow(-1.0,j) > 0.0 ){
            isEvenJ = true;
        }
        for (int i = 0; i < n+1; i++){
            int index = (n+1)*j + i;
            double x = i*dx;
            if ( isEvenJ == true ){
                x = x + a/2.0;
            }
            double y = j*dy;
            vertex(index,0) = x - lx/2.0; 
            vertex(index,1) = y - ly/2.0;
        }
    }
    return vertex;
}
Mat<int> setface_Loop_scheme(double sidex, double sidey, double l){ // face and its surrounding vertex
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3)/2 * a; 
    int m = round(sidey/dy);      // y axis division
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int facenum = m*n*2;
    Mat<int> face(facenum,3);
    #pragma omp parallel for
    for (int j = 0; j < m; j++){
        bool isEvenJ = false;
        if ( pow(-1.0,j) > 0.0 ){
            isEvenJ = true;
        }
        for (int i = 0; i < n; i++){
            int index = 2*n*j + i*2;
            int node1, node2, node3, node4;
            if ( isEvenJ == false ){
                node1 = (n+1)*j + i;
                node2 = (n+1)*(j+1) + i;
                node3 = (n+1)*j + (i+1);
                node4 = (n+1)*(j+1) + (i+1);
            }else{
                node1 = (n+1)*(j+1) + i;
                node2 = (n+1)*(j+1) + (i+1);
                node3 = (n+1)*j + i;
                node4 = (n+1)*j + (i+1);
            }
            face(index,0) = node1; face(index,1) = node2; face(index,2) = node3;
            face(index+1,0) = node2; face(index+1,1) = node4; face(index+1,2) = node3;
        }
    }
    return face;
}

mat setup_inner_mesh(mat vertex, double h){
    int vertexnum = vertex.n_rows;
    mat vertexin(vertexnum,3);
    rowvec zdirect(3); zdirect << 0.0 << 0.0 << 1.0 << endr;
    for (int i = 0; i < vertexnum; i++){
        vertexin.row(i) = vertex.row(i) - h*zdirect;
    }
    return vertexin;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for neighor vertex or faces

// find the faces around one vertex
// each vertex have 6 or 5 or 7 neighbor faces that have this vertex. 
// For the regular vertex, it has 6 neighbors; for the irregular one, it has 5 neighbors;
Mat<int> faces_with_vertexi(mat vertex, Mat<int> face){
    int vertexnum = vertex.n_rows;
    Mat<int> face_with_nodei(vertexnum,7); face_with_nodei.fill(-1);
    #pragma omp parallel for 
     for (int i=0; i < vertex.n_rows; i++){
        int facenumber = -1;
        for (int j=0; j < face.n_rows; j++){
            for (int k=0; k<face.n_cols; k++){
                if ( i == face(j,k) ){
                    facenumber = facenumber + 1;
                    face_with_nodei(i,facenumber) = j;
                }
            }
        }  
     }
     return face_with_nodei;
}

// valence of each vertex. it could be 5, 6, 7. 
// it is the closest vertices around the vertex_i
Mat<int> vertex_valence(Mat<int> faces_with_nodei, Mat<int> face){   
    int vertexnum = faces_with_nodei.n_rows;
    Mat<int> closest_nodes(vertexnum,7); closest_nodes.fill(-1); 
    #pragma omp parallel for
    for (int i = 0; i < vertexnum; i++){
        //how many neighbor vertices
        int N = 7;
        if (faces_with_nodei(i,6) == -1 && faces_with_nodei(i,5) != -1){
            N = 6;
        }else if (faces_with_nodei(i,6) == -1 && faces_with_nodei(i,5) == -1){
            N = 5;
        }
        Row<int> A(N); A.fill(-1);
        int shu = -1;
        for (int j = 0; j < N; j++){
            int Numface = faces_with_nodei(i,j); 
            if (Numface == -1) continue;
            for (int k = 0; k < 3; k++){
                if ( face(Numface,k) != i ){
                    bool islisted = false;
                    for (int m = 0; m < N; m++){
                        if (face(Numface,k) == A(m)){
                            islisted = true;
                        }
                    }
                    if ( islisted == false ){
                         shu = shu + 1;
                         A(shu) = face(Numface,k);
                    }
                }
            }
        }
        for (int j = 0; j < N; j++){
            closest_nodes(i,j) = A(j);
        }
    }
    return closest_nodes;
}

int find_valence_number(Row<int> shu){
    int number = shu.n_cols;
    for (int i = 0; i < shu.n_cols; i++){
        if (shu(i) < 0){
            number = number -1;
        }else{
            continue;
        }
    }
    return number;
}

// The first column is the valence vector of the face_i, which is the valence of its 3 vertices. 
// The second column is the vertex-position in face.row(i)
Mat<int> determine_valence_vector(Mat<int> face, Mat<int> closest_nodes, int i){
    Mat<int> matrixtmp(3,2); 

    int valence0 = find_valence_number(closest_nodes.row(face(i,0)));
    int valence1 = find_valence_number(closest_nodes.row(face(i,1)));
    int valence2 = find_valence_number(closest_nodes.row(face(i,2)));
    Row<int> valenceVec(3); valenceVec << valence0 << valence1 << valence2 << endr;

    // regular patch
    if ( min(valenceVec) == 6 &&  max(valenceVec) == 6 ){ 
        matrixtmp(0,0) = 0; matrixtmp(0,1) =  6;
        matrixtmp(1,0) = 1; matrixtmp(1,1) =  6;
        matrixtmp(2,0) = 2; matrixtmp(2,1) =  6;
    }
    // irregular patch
    if ( min(valenceVec) == 5 &&  max(valenceVec) == 6  ){
        if ( valenceVec(0) == 5 ){
            matrixtmp(0,0) = 0; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 1; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 2; matrixtmp(2,1) =  6;
        }else if ( valenceVec(1) == 5 ){
            matrixtmp(0,0) = 1; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 2; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 0; matrixtmp(2,1) =  6;
        }else if ( valenceVec(2) == 5 ){
            matrixtmp(0,0) = 2; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 0; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 1; matrixtmp(2,1) =  6;
        }
    }
    // complex patch
    if ( min(valenceVec) == 6 &&  max(valenceVec) == 7  ){
        if ( valenceVec(0) == 7 ){
            matrixtmp(0,0) = 1; matrixtmp(0,1) =  6;
            matrixtmp(1,0) = 2; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 0; matrixtmp(2,1) =  7;
        }else if ( valenceVec(1) == 7 ){
            matrixtmp(0,0) = 2; matrixtmp(0,1) =  6;
            matrixtmp(1,0) = 0; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 1; matrixtmp(2,1) =  7;
        }else if ( valenceVec(2) == 7 ){
            matrixtmp(0,0) = 0; matrixtmp(0,1) =  6;
            matrixtmp(1,0) = 1; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 2; matrixtmp(2,1) =  7;
        }
    }
    // sudo-regular patch: 1
    if ( min(valenceVec) == 5 &&  max(valenceVec) == 7  ){
        if ( valenceVec(0) == 5 && valenceVec(1) == 7 ){
            matrixtmp(0,0) = 0; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 1; matrixtmp(1,1) =  7;
            matrixtmp(2,0) = 2; matrixtmp(2,1) =  6;
        }else if ( valenceVec(1) == 5 && valenceVec(2) == 7 ){
            matrixtmp(0,0) = 1; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 2; matrixtmp(1,1) =  7;
            matrixtmp(2,0) = 0; matrixtmp(2,1) =  6;
        }else if ( valenceVec(2) == 5 && valenceVec(0) == 7 ){
            matrixtmp(0,0) = 2; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 0; matrixtmp(1,1) =  7;
            matrixtmp(2,0) = 1; matrixtmp(2,1) =  6;
        }
    }
    // sudo-regular patch: 2
    if ( min(valenceVec) == 5 &&  max(valenceVec) == 7  ){
        if ( valenceVec(0) == 5 && valenceVec(1) == 6 ){
            matrixtmp(0,0) = 0; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 1; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 2; matrixtmp(2,1) =  7;
        }else if ( valenceVec(1) == 5 && valenceVec(2) == 6 ){
            matrixtmp(0,0) = 1; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 2; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 0; matrixtmp(2,1) =  7;
        }else if ( valenceVec(2) == 5 && valenceVec(0) == 6 ){
            matrixtmp(0,0) = 2; matrixtmp(0,1) =  5;
            matrixtmp(1,0) = 0; matrixtmp(1,1) =  6;
            matrixtmp(2,0) = 1; matrixtmp(2,1) =  7;
        }
    }

    return matrixtmp;
}

// find the node that is mirror to node3, symetry of the line node1-node2
int find_nodeindex(int node1, int node2, int node3, Mat<int> closest_nodes){
    int node;
    for (int i = 0; i < closest_nodes.n_cols; i++){
        int nodetmp1 = closest_nodes(node1,i);
        for (int j = 0; j < closest_nodes.n_cols; j++){
            int nodetmp2 = closest_nodes(node2,j);
            if ( nodetmp1 == nodetmp2 && nodetmp1 != -1 && nodetmp1 != node3 ){
                node = nodetmp1;
            }
        }
    }
    return node;
}

// one_ring_vertices. It store the one_ring neighbor vertices around each face. The one-ring-vertices are stored in specific order! They will combine the box-spline for differential geometry.
Mat<int> one_ring_vertices(Mat<int> face,mat vertex, Mat<int> closest_nodes, Row<int> isBoundaryFace){   
    int facenum = face.n_rows;
    Mat<int> ring_vertices(facenum,13); ring_vertices.fill(-1);
    // three types of patch: 1. regular patch with 12 one-ring vertices, each vertex has 6 closest nodes. valence vector = [6,6,6];
    //                       2. irregular-minus patch with 11 one-ring vertices, one vertex has 5 closest-nodes. valence vector = [5,6,6];
    //                       3. irregular-plus patch with 13 one-ring vertices, one vertex has 7 closest-nodes. valence vector = [6,6,7];
    //                       4. pseudo-regular patch1 with 12 one-ring vertice,but one vertex has 7 closest-nodes, and one vertex has 5 closest-nodes. valence vector = [5,7,6];
    //                       5. pseudo-regular patch2. chiral symetry to case 4. valence vector = [5,6,7];
    // the one-ring-vertex last element shows out the case index as -1, -2, >0, -4, -5. !!!!!!
    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        if (i < isBoundaryFace.n_cols){
            if (isBoundaryFace(i) == 1) 
                continue;
        }
        //int d1, d2, d3, d5, d6, d9, d10, d11, d12, d13;
        // note the order of the vertex
        Mat<int> neighbornum = determine_valence_vector(face, closest_nodes, i);   
        // case 1: regular patch
        if ( neighbornum(0,1) == 6 && neighbornum(1,1) == 6 && neighbornum(2,1) == 6 ){ 
            int d4 = face(i,0); int d7 = face(i,1); int d8 = face(i,2);
            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to +z axis or outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center(3); center << 0.0 << 0.0 << 1.0 << endr;
            mat shu = cross(node7-node4,node8-node4)*strans(center);
            if (shu(0,0) < 0){
                d7 = face(i,2);
                d8 = face(i,1);
            }
            //
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes);
            int d11 = find_nodeindex(d7, d8, d4, closest_nodes);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes);
            int d10 = find_nodeindex(d7, d11, d8, closest_nodes);
            int d12 = find_nodeindex(d8, d11, d7, closest_nodes);
            int d13 = -1;
            Row<int> v(13); v << d1 << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << endr;
            ring_vertices.row(i) = v;

        // case 2: irregular-minus patch
        }else if ( neighbornum(0,1) == 5 && neighbornum(1,1) == 6 && neighbornum(2,1) == 6 ){
            int d4 = face(i,neighbornum(0,0)); 
            int d7 = face(i,neighbornum(1,0));
            int d8 = face(i,neighbornum(2,0));
            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to +z axis or outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center(3); center << 0.0 << 0.0 << 1.0 << endr;
            mat shu = cross(node7-node4,node8-node4)*strans(center);
            if (shu(0,0) < 0){
                d7 = face(i,neighbornum(2,0));
                d8 = face(i,neighbornum(1,0));
            }
            // d4 is the vertex with valence = 5
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes);
            int d11 = find_nodeindex(d7, d8, d4, closest_nodes);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes);
            int d10 = find_nodeindex(d7, d11, d8, closest_nodes);
            int d12 = find_nodeindex(d8, d11, d7, closest_nodes);
            int d13 = -2;
            Row<int> v(13); v << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << d13 << endr;
            ring_vertices.row(i) = v;

        // case 3: irregular-plus patch
        }else if ( neighbornum(0,1) == 6 && neighbornum(1,1) == 6 && neighbornum(2,1) == 7 ){
            int d4 = face(i,neighbornum(0,0)); 
            int d7 = face(i,neighbornum(1,0));
            int d8 = face(i,neighbornum(2,0));
            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to +z axis or outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center(3); center << 0.0 << 0.0 << 1.0 << endr;
            mat shu = cross(node4-node8,node7-node8)*strans(center);
            if (shu(0,0) < 0){
                d4 = face(i,neighbornum(1,0));
                d7 = face(i,neighbornum(0,0));
            }
            // d8 is the vertex with valence = 7
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes);
            int d11 = find_nodeindex(d7, d8, d4, closest_nodes);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes);
            int d10 = find_nodeindex(d7, d11, d8, closest_nodes);
            int d12 = find_nodeindex(d8, d11, d7, closest_nodes);
            int d13 = find_nodeindex(d8, d9, d5, closest_nodes);
            Row<int> v(13); v << d1 << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << endr;
            ring_vertices.row(i) = v;
      
        // case 4: pseudo-regular patch 1
        }else if ( neighbornum(0,1) == 5 && neighbornum(1,1) == 7 && neighbornum(2,1) == 6 ){
            int d4 = face(i,neighbornum(0,0)); 
            int d7 = face(i,neighbornum(1,0));
            int d8 = face(i,neighbornum(2,0));
            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to +z axis or outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center(3); center << 0.0 << 0.0 << 1.0 << endr;
            mat shu = cross(node4-node8,node7-node8)*strans(center);
            if (shu(0,0) < 0){
                d7 = face(i,neighbornum(2,0));
                d8 = face(i,neighbornum(1,0));
            }
            // d4 valence = 5; d7 valence = 7; d8 valence = 6;
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes);
            int d12 = find_nodeindex(d7, d8, d4, closest_nodes);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes);
            int d10 = find_nodeindex(d7, d6, d3, closest_nodes);
            int d11 = find_nodeindex(d7, d12, d8, closest_nodes);
            int d13 = find_nodeindex(d8, d9, d5, closest_nodes);
            int dend = -4;
            Row<int> v(13); v << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << dend << endr;
            ring_vertices.row(i) = v;

        // case 5: pseudo-regular patch 2
        }else if ( neighbornum(0,1) == 5 && neighbornum(1,1) == 6 && neighbornum(2,1) == 7 ){
            int d4 = face(i,neighbornum(0,0)); 
            int d7 = face(i,neighbornum(1,0));
            int d8 = face(i,neighbornum(2,0));
            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to +z axis or outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center(3); center << 0.0 << 0.0 << 1.0 << endr;
            mat shu = cross(node4-node8,node7-node8)*strans(center);
            if (shu(0,0) < 0){
                d7 = face(i,neighbornum(2,0));
                d8 = face(i,neighbornum(1,0));
            }
            // d4 valence = 5; d7 valence = 6; d8 valence = 7;
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes);
            int d11 = find_nodeindex(d7, d8, d4, closest_nodes);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes);
            int d10 = find_nodeindex(d7, d11, d8, closest_nodes);
            int d12 = find_nodeindex(d8, d11, d7, closest_nodes);
            int d13 = find_nodeindex(d8, d9, d5, closest_nodes);
            int dend = -5;
            Row<int> v(13); v << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << dend << endr;
            ring_vertices.row(i) = v;
        }
    }
    return ring_vertices;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about the boundaries

// boundary vertex is located on the most edge.
Row<int> determine_BoundaryVertex(double sidex, double sidey, double l){
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3)/2 * a; 
    int m = round(sidey/dy);      // y axis division
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    Row<int> isBoundaryNode(vertexnum); isBoundaryNode.fill(-1);
    #pragma omp parallel for
    for (int j = 0; j < m+1; j++){
        for (int i = 0; i < n+1; i++){
            int index = (n+1)*j + i;
            if ( j == 0 || j == m || i == 0 || i == n ){
                isBoundaryNode(index) = 1; // if element is 1, then this vertex is on boundary
            }
        }
    }
    return isBoundaryNode;
}

// boundary face is located on the most edge
Row<int> determine_Boundaryface(Mat<int> face, Row<int> isBoundaryNode){
    int facenum = face.n_rows;
    Row<int> isBoundaryFace(facenum); isBoundaryFace.fill(-1);
    #pragma omp parallel for 
    for (int i = 0; i < facenum; i++){
        int node1 = face(i,0);
        int node2 = face(i,1);
        int node3 = face(i,2);
        bool isboundaryface = false;
        if ( isBoundaryNode(node1) == 1 || isBoundaryNode(node2) == 1 || isBoundaryNode(node3) == 1 ){
            isboundaryface = true;
            isBoundaryFace(i) = 1; // if element is 1, then this face is on boundary
        }
    }
    return isBoundaryFace;
}

// ghost vertex exists for free or periodic boundary condition 
// Free boundary condition: the most edge vertex is ghost vertex
// Periodic boundary condition: the three edge vertices are all ghost. 
Row<int> determine_GhostVertex(double sidex, double sidey, double l, bool isBoundaryFixed, bool isBoundaryPeriodic, bool isBoundaryFree){
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3)/2 * a; 
    int m = round(sidey/dy);      // y axis division
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    Row<int> isGhostNode(vertexnum); isGhostNode.fill(-1);

    Row<int> TopBottom; 
    Row<int> LeftRight; 
    if (isBoundaryPeriodic == true && isBoundaryFree == false && isBoundaryFixed == false){
        TopBottom << 0 << 1 << 2 << m-2 << m-1 << m << endr;
        LeftRight << 0 << 1 << 2 << n-2 << n-1 << n << endr;
    }
    if (isBoundaryFree == true && isBoundaryPeriodic == false && isBoundaryFixed == false){
        TopBottom << 0 << m << endr;
        LeftRight << 0 << n << endr;
    }
    int number = TopBottom.n_cols;
    for (int k = 0; k < number; k++){
        int j = TopBottom(k);
        #pragma omp parallel for
        for (int i = 0; i < n+1; i++){
            int index = (n+1)*j + i;
            isGhostNode(index) = 1; // if element is 1, then this vertex is ghost
        }
    }
    for (int k = 0; k < number; k++){
        int i = LeftRight(k);
        #pragma omp parallel for
        for (int j = 0; j < m+1; j++){
            int index = (n+1)*j + i;
            isGhostNode(index) = 1; // if element is 1, then this vertex is ghost
        }
    }
    return isGhostNode;
}

Row<int> determine_GhostFace(double sidex, double sidey, double l, bool isBoundaryFixed, bool isBoundaryPeriodic, bool isBoundaryFree){
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3)/2 * a; 
    int m = round(sidey/dy);      // y axis division
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int facenum = m*n*2;
    Row<int> isGhostFace(facenum); isGhostFace.fill(-1);

    Row<int> TopBottom; 
    Row<int> LeftRight; 
    if (isBoundaryPeriodic == true && isBoundaryFree == false && isBoundaryFixed == false){
        TopBottom << 0 << 1 << 2 << m-3 << m-2 << m-1 << endr;
        LeftRight << 0 << 1 << 2 << n-3 << n-2 << n-1 << endr;
    }
    if (isBoundaryFree == true && isBoundaryPeriodic == false && isBoundaryFixed == false){
        TopBottom << 0 << m-1 << endr;
        LeftRight << 0 << n-1 << endr;
    }
    int number = TopBottom.n_cols;
    for (int k = 0; k < number; k++){
        int j = TopBottom(k);
        #pragma omp parallel for
        for (int i = 0; i < n; i++){
            int index = 2*n*j + i*2;
            isGhostFace(index) = 1; // if element is 1, then this face is ghost
            isGhostFace(index+1) = 1;
        }
    }
    for (int k = 0; k < number; k++){
        int i = LeftRight(k);
        #pragma omp parallel for
        for (int j = 0; j < m; j++){
            int index = 2*n*j + i*2;
            isGhostFace(index) = 1; // if element is 1, then this face is ghost
            isGhostFace(index+1) = 1;
        }
    }

    return isGhostFace;
}

vector<int> determine_GhostPartner(double sidex, double sidey, double l){   
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double aa = dx;
    double dy = sqrt(3.0)/2.0 * aa; 
    int m = round(sidey/dy);      // y axis division 
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    //nint facenum = m*n*2;

    vector<int> ghostPartner(vertexnum, -1); 

    //#pragma omp parallel for 
    for (int i = 0; i < n+1; i++){ // for top and bottom ghost vertices
        int index0 = (n+1)*0 + i;      // bottom ghost vertices
        int index1 = (n+1)*1 + i;
        int index2 = (n+1)*2 + i;
        //int index00 = (n+1)*(m-6) + i; // top real vertices
        //int index11 = (n+1)*(m-5) + i;
        //int index22 = (n+1)*(m-4) + i;
        int index00 = (n+1)*(m-5) + i; // top real vertices
        int index11 = (n+1)*(m-4) + i;
        int index22 = (n+1)*(m-3) + i;
        ghostPartner[index0] = index00;
        ghostPartner[index1] = index11;
        ghostPartner[index2] = index22;

        index0 = (n+1)*(m-2) + i; // top ghost vertices
        index1 = (n+1)*(m-1) + i;
        index2 = (n+1)*(m-0) + i;
        //index00 = (n+1)*4 + i;    // bottom real vertices
        //index11 = (n+1)*5 + i;
        //index22 = (n+1)*6 + i;
        index00 = (n+1)*3 + i;    // bottom real vertices
        index11 = (n+1)*4 + i;
        index22 = (n+1)*5 + i;
        ghostPartner[index0] = index00;
        ghostPartner[index1] = index11;
        ghostPartner[index2] = index22;
    }
    //#pragma omp parallel for 
    for (int j = 0; j < m+1; j++){ // for left and right ghost vertices
        int index0 = (n+1)*j + 0; // left ghost
        int index1 = (n+1)*j + 1;
        int index2 = (n+1)*j + 2;
        //int index00 = (n+1)*j + n-6; // right real
        //int index11 = (n+1)*j + n-5;
        //int index22 = (n+1)*j + n-4;
        int index00 = (n+1)*j + n-5; // right real
        int index11 = (n+1)*j + n-4;
        int index22 = (n+1)*j + n-3;
        ghostPartner[index0] = index00;
        ghostPartner[index1] = index11;
        ghostPartner[index2] = index22;

        index0 = (n+1)*j + n; // right ghost
        index1 = (n+1)*j + n-1;
        index2 = (n+1)*j + n-2;
        //index00 = (n+1)*j + 6; // left real
        //index11 = (n+1)*j + 5;
        //index22 = (n+1)*j + 4;
        index00 = (n+1)*j + 5; // left real
        index11 = (n+1)*j + 4;
        index22 = (n+1)*j + 3;
        ghostPartner[index0] = index00;
        ghostPartner[index1] = index11;
        ghostPartner[index2] = index22;
    }

    return ghostPartner;
}

vector<vector<int>> determine_freePartners(double sidex, double sidey, double l){
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double aa = dx;
    double dy = sqrt(3)/2 * aa; 
    int m = round(sidey/dy);      // y axis division 
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    //nint facenum = m*n*2;

    vector<vector<int>> freePartners(vertexnum, vector<int>( 3, -1) ); 

    int index1, index2, index3, index4;
    // left side
    for (int j = 2; j < m ; j++ ){
        if ( pow(-1.0,j) > 0.0 ){
            index1 = (n+1)*j + 0;
            index2 = (n+1)*j + 1;
            index3 = (n+1)*(j-1) + 1;
            index4 = (n+1)*(j-1) + 2;
        }else{
            index1 = (n+1)*j + 0;
            index2 = (n+1)*j + 1;
            index3 = (n+1)*(j-1) + 0;
            index4 = (n+1)*(j-1) + 1;
        }
        freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    }
    index1 = (n+1)*1+0; 
    index2 = (n+1)*2+0;
    index3 = (n+1)*1+1;
    index4 = (n+1)*2+1;
    freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    // right side
    index1 = (n+1)*1 + n; 
    index2 = (n+1)*2 + n-1;
    index3 = (n+1)*1 + n-1;
    index4 = (n+1)*2 + n-2;
    freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    for (int j = 2; j < m ; j++ ){
        if ( pow(-1.0,j) > 0.0 ){
            index1 = (n+1)*j + n;
            index2 = (n+1)*j + n-1;
            index3 = (n+1)*(j-1) + n;
            index4 = (n+1)*(j-1) + n-1;
        }else{
            index1 = (n+1)*j + n;
            index2 = (n+1)*j + n-1;
            index3 = (n+1)*(j-1) + n-1;
            index4 = (n+1)*(j-1) + n-2;
        }
        freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    }
    // bottom 
    for (int i = 0; i < n; i++){
        index1 = (n+1)*0 + i;
        index2 = (n+1)*1 + i;
        index3 = (n+1)*1 + i+1;
        index4 = (n+1)*2 + i;
        freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    }
    index1 = (n+1)*0 + n; 
    index2 = (n+1)*0 + n-1;
    index3 = (n+1)*1 + n;
    index4 = (n+1)*1 + n-1;
    freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    // top 
    for (int i = 1; i < n+1; i++){
        index1 = (n+1)*m + i;
        index2 = (n+1)*(m-1) + i;
        index3 = (n+1)*(m-1) + i+1;
        index4 = (n+1)*(m-2) + i;
        freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;
    }
    index1 = (n+1)*m + n; 
    index2 = (n+1)*m + n-1;
    index3 = (n+1)*(m-1) + n;
    index4 = (n+1)*(m-1) + n-1;
    freePartners[index1][0] = index2; freePartners[index1][1] = index3; freePartners[index1][2] = index4;

    return freePartners;    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about inserting triangles to the mesh 
/*
double get_magnitude(rowvec targvec){
    double out;
    for (int i = 0; i < targvec.n_cols; i++){
        out = out + targvec(i)*targvec(i);
    }
    return sqrt(out);
}

Mat<int> adjust_vertex_dueTo_insertion(mat& vertexnew, Mat<int>& facenew, Mat<int> faces_with_nodei, Mat<int> insertionVertex, double meanL){
    mat vertex = vertexnew;
    Mat<int> face = facenew;

    int insertrownum = insertionVertex.n_rows;
    int insertcolnum = insertionVertex.n_cols;
    if (insertrownum < 1 || insertcolnum < 1){
        Mat<int> out;
        return out;
    }

    for (int i = 0; i < insertrownum; i++){
        // define the direction of the insertion;
        int vertex1 = insertionVertex(i,0);
        int vertex2 = insertionVertex(i,insertcolnum-1);
        rowvec insertdirect = vertex.row(vertex1) - vertex.row(vertex2); 
        insertdirect = insertdirect / get_magnitude(insertdirect);
        //rowvec radiusdirect = 0.5*( vertex.row(vertex1) + vertex.row(vertex2) );
        //radiusdirect = radiusdirect / get_magnitude(radiusdirect);
        rowvec radiusdirect(3); radiusdirect << 0.0 << 0.0 << 1.0 << endr;
        rowvec leftmove = cross(radiusdirect,insertdirect); 
        //leftmove = leftmove * sqrt(3.0)/4.0*meanL/2.0;
        leftmove = leftmove * meanL * 0.05;
        // insert new vertex 
        int newvertexnum = insertcolnum-2;
        mat newvertex(newvertexnum,3);
        for (int j = 1; j < insertcolnum-1; j++){
            int nodeindex = insertionVertex(i,j);
            vertexnew.row(nodeindex) = vertex.row(nodeindex) + leftmove;
            newvertex.row(j-1) = vertex.row(nodeindex) - leftmove;
        }
        int vertexnumold = vertexnew.n_rows;
        vertexnew.insert_rows(vertexnumold,newvertex);
        // insert new faces
        int newfacenum = 2*(insertcolnum-3) + 2;
        Mat<int> newface(newfacenum,3);
        newface(0,0) = insertionVertex(i,0); newface(0,1) = insertionVertex(i,1); newface(0,2) = vertexnumold + 0;
        newface(newfacenum-1,0) = insertionVertex(i,insertcolnum-1); newface(newfacenum-1,1) = insertionVertex(i,insertcolnum-2); newface(newfacenum-1,2) = vertexnumold + insertcolnum-3;
        if (insertcolnum > 3){
            for (int j = 1; j < insertcolnum -2; j++){
                int node1 = insertionVertex(i,j);
                int node2 = insertionVertex(i,j+1);
                int node3 = vertexnumold + (j-1);
                int node4 = vertexnumold + j;
                newface(2*(j-1)+1,0) = node1; newface(2*(j-1)+1,1) = node2; newface(2*(j-1)+1,2) = node4; 
                newface(2*(j-1)+2,0) = node1; newface(2*(j-1)+2,1) = node3; newface(2*(j-1)+2,2) = node4; 
            }
        }
        int facenumold = facenew.n_rows;
        facenew.insert_rows(facenumold,newface);
        // update the right-side faces that has the insertVertex, replace with new vertex index.
        for (int j = 1; j < insertionVertex.n_cols-1; j++){
            int nodeindex = insertionVertex(i,j);
            int newnodeindex = vertexnumold + (j-1);
            for (int k = 0; k < faces_with_nodei.n_cols; k++){
                if ( faces_with_nodei(nodeindex,k) == -1 ) continue;
                int faceindex = faces_with_nodei(nodeindex,k);
                int node1 = face(faceindex,0);
                int node2 = face(faceindex,1);
                int node3 = face(faceindex,2);
                rowvec center = 1.0/3.0*( vertex.row(node1) + vertex.row(node2) + vertex.row(node3) );
                rowvec tempdirect = center - vertex.row(vertex1);
                mat shu = cross(tempdirect,radiusdirect)*strans(insertdirect);
                bool isleft = false;
                if (shu(0,0) > 0.0 ) isleft = true;
                if (isleft == true) continue;
                for (int m = 0; m < 3; m++){
                    if ( face(faceindex,m) == nodeindex ){
                        facenew(faceindex,m) = newnodeindex;
                    }
                }
            }
        }
    }
    // store the insertion faces.
    int facenumPerInsertion = (facenew.n_rows - face.n_rows)/insertrownum;
    Mat<int> insertionpatch(insertrownum,facenumPerInsertion);
    for (int i = 0; i < insertrownum; i++){
        for (int j = 0; j < facenumPerInsertion; j++){
            insertionpatch(i,j) = face.n_rows + facenumPerInsertion*i + j;
        }
    }
    return insertionpatch;
}
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about box spline 

mat shapefunctions(rowvec vwu){
    // 12 shape functions and their differential equations; shape_functions(:,1), shape functions;  
    // shape_functions(:,2), differential to v; shape_functions(:,3), differential to w; 
    // shape_functions(:,4), double differential to v; shape_functions(:,5), double differential to w;
    // shape_functions(:,6), differential to v and w; shape_functions(:,7), differential to w and v;
    double v = vwu(0); double w = vwu(1); double u = vwu(2);
    mat sf(12,7); sf.fill(0.0);
    sf(0,0) = 1.0/12.0*(pow(u,4.0) + 2.0*pow(u,3.0)*v);
    sf(0,1) = 1.0/12.0*(-2.0*pow(u,3.0) - 6.0*pow(u,2.0)*v); 
    sf(0,2) = 1.0/12.0*(-4.0*pow(u,3.0) - 6.0*pow(u,2.0)*v); 
    sf(0,3) = u*v; 
    sf(0,4) = pow(u,2.0) + u*v; 
    sf(0,5) = 1.0/2.0*(pow(u,2.0) + 2.0*u*v); 
    sf(0,6) = 1.0/2.0*(pow(u,2.0) + 2.0*u*v); 
    sf(1,0) = 1.0/12.0*(pow(u,4.0) + 2.0*pow(u,3.0)*w); 
    sf(1,1) = 1.0/12.0*(-4.0*pow(u,3.0) - 6.0*pow(u,2.0)*w); 
    sf(1,2) = 1.0/12.0*(-2.0*pow(u,3.0) - 6.0*pow(u,2.0)*w);
    sf(1,3) = pow(u,2.0) + u*w; 
    sf(1,4) = u*w;
    sf(1,5) = 1.0/2.0*(pow(u,2.0) + 2.0*u*w); 
    sf(1,6) = 1.0/2.0*(pow(u,2.0) + 2.0*u*w); 
    sf(2,0) = 1.0/12.0*(pow(u,4.0) + 2.0*pow(u,3.0)*w + 6.0*pow(u,3.0)*v + 6.0*pow(u,2.0)*v*w + 12.0*pow(u,2.0)*pow(v,2.0) + 6.0*u*pow(v,2.0)*w + 6.0*u*pow(v,3.0) + 2.0*pow(v,3.0)*w + pow(v,4.0));
    sf(2,1) = 1.0/12.0*(2.0*pow(u,3.0) + 6.0*pow(u,2.0)*v - 6.0*u*pow(v,2.0) - 2.0*pow(v,3.0));
    sf(2,2) = 1.0/12.0*(-2.0*pow(u,3.0) - 6.0*pow(u,2.0)*w - 12.0*pow(u,2)*v - 12.0*u*v*w - 18.0*u*pow(v,2.0) - 6.0*pow(v,2.0)*w - 4.0*pow(v,3.0));
    sf(2,3) = -2.0*u*v;
    sf(2,4) = u*w + v*w + u*v + pow(v,2.0);
    sf(2,5) = 1.0/2.0*(-pow(u,2.0) - 2.0*u*v + pow(v,2.0));
    sf(2,6) = 1.0/2.0*(-pow(u,2.0) - 2.0*u*v + pow(v,2.0));
    sf(3,0) = 1.0/12.0*(6.0*pow(u,4.0) + 24.0*pow(u,3.0)*w + 24.0*pow(u,2.0)*pow(w,2.0) + 8.0*u*pow(w,3.0) + pow(w,4.0) + 24.0*pow(u,3.0)*v + 60.0*pow(u,2.0)*v*w + 36.0*u*v*pow(w,2.0) + 6.0*v*pow(w,3.0) + 24.0*pow(u,2.0)*pow(v,2.0) + 36.0*u*pow(v,2.0)*w + 12.0*pow(v,2.0)*pow(w,2.0) + 8.0*u*pow(v,3.0) + 6.0*pow(v,3.0)*w + pow(v,4.0));
    sf(3,1) = 1.0/12.0*(-12.0*pow(u,2.0)*w - 12.0*u*pow(w,2.0) - 2.0*pow(w,3.0) - 24.0*pow(u,2.0)*v - 48.0*u*v*w - 12.0*v*pow(w,2.0) -24.0*u*pow(v,2.0) - 18.0*pow(v,2.0)*w - 4.0*pow(v,3.0));
    sf(3,2) = 1.0/12.0*(-24.0*pow(u,2.0)*w - 24.0*u*pow(w,2.0) - 4.0*pow(w,3.0) - 12.0*pow(u,2.0)*v - 48.0*u*v*w - 18.0*v*pow(w,2.0) - 12.0*u*pow(v,2.0) - 12.0*pow(v,2.0)*w - 2.0*pow(v,3.0));
    sf(3,3) = -2.0*u*w - 2.0*pow(u,2.0) + v*w + pow(v,2.0);
    sf(3,4) = -2.0*pow(u,2.0) + pow(w,2.0) - 2.0*u*v + v*w;
    sf(3,5) = 1.0/2.0*(-2.0*pow(u,2.0) + pow(w,2.0) + 4.0*v*w + pow(v,2.0));
    sf(3,6) = 1.0/2.0*(-2.0*pow(u,2.0) + pow(w,2.0) + 4.0*v*w + pow(v,2.0));
    sf(4,0) = 1.0/12.0*(pow(u,4.0) + 6.0*pow(u,3.0)*w + 12.0*pow(u,2.0)*pow(w,2.0) + 6.0*u*pow(w,3.0) + pow(w,4.0) + 2.0*pow(u,3.0)*v + 6.0*pow(u,2.0)*v*w + 6.0*u*v*pow(w,2.0) + 2.0*v*pow(w,3.0));
    sf(4,1) = 1.0/12.0*(-2.0*pow(u,3.0) - 12.0*pow(u,2.0)*w - 18.0*u*pow(w,2.0) - 4.0*pow(w,3.0) - 6.0*pow(u,2.0)*v - 12.0*u*v*w - 6.0*v*pow(w,2.0));
    sf(4,2) = 1.0/12.0*(2.0*pow(u,3.0) + 6.0*pow(u,2.0)*w - 6.0*u*pow(w,2.0) - 2.0*pow(w,3.0));
    sf(4,3) = u*w + pow(w,2.0) + u*v + v*w;
    sf(4,4) = -2.0*u*w;
    sf(4,5) = 1.0/2.0*(-pow(u,2.0) - 2.0*u*w + pow(w,2.0));
    sf(4,6) = 1.0/2.0*(-pow(u,2.0) - 2.0*u*w + pow(w,2.0));
    sf(5,0) = 1.0/12.0*(2.0*u*pow(v,3.0) + pow(v,4.0)); 
    sf(5,1) = 1.0/12.0*(6.0*u*pow(v,2.0) + 2.0*pow(v,3.0)); 
    sf(5,2) = -1.0/6.0*pow(v,3.0);
    sf(5,3) = u*v; 
    sf(5,4) = 0.0;
    sf(5,5) = -1.0/2.0*pow(v,2.0); 
    sf(5,6) = -1.0/2.0*pow(v,2.0);
    sf(6,0) = 1.0/12.0*(pow(u,4.0) + 6.0*pow(u,3.0)*w + 12.0*pow(u,2.0)*pow(w,2.0) + 6.0*u*pow(w,3.0)+ pow(w,4.0) + 8.0*pow(u,3.0)*v + 36.0*pow(u,2.0)*v*w + 36.0*u*v*pow(w,2.0) + 8.0*v*pow(w,3.0) + 24.0*pow(u,2.0)*pow(v,2.0) + 60.0*u*pow(v,2.0)*w + 24.0*pow(v,2.0)*pow(w,2.0) + 24.0*u*pow(v,3.0) + 24.0*pow(v,3.0)*w + 6.0*pow(v,4.0));
   sf(6,1) = 1.0/12.0*(4.0*pow(u,3.0) + 18.0*pow(u,2.0)*w + 12.0*u*pow(w,2.0) + 2.0*pow(w,3.0) + 24.0*pow(u,2.0)*v + 48.0*u*v*w + 12.0*v*pow(w,2.0) + 24.0*u*pow(v,2.0) + 12.0*pow(v,2.0)*w);
   sf(6,2) = 1.0/12.0*(2.0*pow(u,3.0) + 6.0*pow(u,2.0)*w - 6.0*u*pow(w,2.0) - 2.0*pow(w,3.0) + 12.0*pow(u,2.0)*v - 12.0*v*pow(w,2.0) + 12.0*u*pow(v,2.0) - 12.0*pow(v,2.0)*w);
   sf(6,3) = pow(u,2.0) + u*w - 2.0*v*w - 2.0*pow(v,2.0);
   sf(6,4) = -2.0*u*w - 2.0*u*v - 2.0*v*w - 2.0*pow(v,2.0);
   sf(6,5) = 1.0/2.0*(pow(u,2.0) - 2.0*u*w - pow(w,2.0) - 4.0*v*w - 2.0*pow(v,2.0));
   sf(6,6) = 1.0/2.0*(pow(u,2.0) - 2.0*u*w - pow(w,2.0) - 4.0*v*w - 2.0*pow(v,2.0));
   sf(7,0) = 1.0/12.0*(pow(u,4.0) + 8.0*pow(u,3.0)*w + 24.0*pow(u,2.0)*pow(w,2.0) + 24.0*u*pow(w,3.0) + 6.0*pow(w,4.0) + 6.0*pow(u,3.0)*v + 36.0*pow(u,2.0)*v*w + 60.0*u*v*pow(w,2.0) + 24.0*v*pow(w,3.0) + 12.0*pow(u,2.0)*pow(v,2.0) + 36.0*u*pow(v,2.0)*w + 24.0*pow(v,2.0)*pow(w,2.0) + 6.0*u*pow(v,3.0) + 8.0*pow(v,3.0)*w + pow(v,4.0));
   sf(7,1) = 1.0/12.0*(2.0*pow(u,3.0) + 12.0*pow(u,2.0)*w + 12.0*u*pow(w,2.0) + 6.0*pow(u,2.0)*v - 12.0*v*pow(w,2.0) - 6.0*u*pow(v,2.0) - 12.0*pow(v,2.0)*w - 2.0*pow(v,3.0));
   sf(7,2) = 1.0/12.0*(4.0*pow(u,3.0) + 24.0*pow(u,2.0)*w + 24.0*u*pow(w,2.0) + 18.0*pow(u,2.0)*v + 48.0*u*v*w + 12.0*v*pow(w,2.0) + 12.0*u*pow(v,2.0) + 12.0*pow(v,2.0)*w + 2.0*pow(v,3.0));
   sf(7,3) = -2.0*u*w - 2.0*pow(w,2.0) - 2.0*u*v - 2.0*v*w;
   sf(7,4) = pow(u,2.0) - 2.0*pow(w,2.0) + u*v - 2.0*v*w;
   sf(7,5) = 1.0/2.0*(pow(u,2.0) - 2.0*pow(w,2.0) - 2.0*u*v - 4.0*v*w - pow(v,2.0));
   sf(7,6) = 1.0/2.0*(pow(u,2.0) - 2.0*pow(w,2.0) - 2.0*u*v - 4.0*v*w - pow(v,2.0));
   sf(8,0) = 1.0/12.0*(2.0*u*pow(w,3.0) + pow(w,4.0)); 
   sf(8,1) = -1.0/6.0*pow(w,3.0); 
   sf(8,2) = 1.0/12.0*(6.0*u*pow(w,2.0) + 2.0*pow(w,3.0));
   sf(8,3) = 0.0; 
   sf(8,4) = u*w;
   sf(8,5) = -1.0/2.0*pow(w,2.0); 
   sf(8,6) = -1.0/2.0*pow(w,2.0);
   sf(9,0) = 1.0/12.0*(2.0*pow(v,3.0)*w + pow(v,4.0));
   sf(9,1) = 1.0/12.0*(6.0*pow(v,2.0)*w + 4.0*pow(v,3.0)); 
   sf(9,2) = 1.0/6.0*pow(v,3.0);
   sf(9,3) = v*w + pow(v,2.0); 
   sf(9,4) = 0.0;
   sf(9,5) = 1.0/2.0*pow(v,2.0); 
   sf(9,6) = 1.0/2.0*pow(v,2.0);
   sf(10,0) = 1.0/12.0*(2.0*u*pow(w,3.0) + pow(w,4.0) + 6.0*u*v*pow(w,2.0) + 6.0*v*pow(w,3.0) + 6.0*u*pow(v,2.0)*w + 12.0*pow(v,2.0)*pow(w,2.0) + 2.0*u*pow(v,3.0) + 6.0*pow(v,3.0)*w + pow(v,4.0));
   sf(10,1) = 1.0/12.0*(4.0*pow(w,3.0) + 18.0*v*pow(w,2.0) + 6.0*u*pow(w,2.0) + 12.0*pow(v,2.0)*w + 12.0*u*v*w + 2.0*pow(v,3.0) + 6.0*u*pow(v,2.0));
   sf(10,2) = 1.0/12.0*(2.0*pow(w,3.0) + 6.0*u*pow(w,2.0) + 12.0*v*pow(w,2.0) + 12.0*u*v*w + 18.0*pow(v,2.0)*w + 6.0*u*pow(v,2.0) + 4.0*pow(v,3.0));
   sf(10,3) = pow(w,2.0) + v*w + u*w + u*v;
   sf(10,4) = u*w + v*w + u*v + pow(v,2.0);
   sf(10,5) = 1.0/2.0*(pow(w,2.0) + 4.0*v*w + 2.0*u*w + pow(v,2.0) + 2.0*u*v);
   sf(10,6) = 1.0/2.0*(pow(w,2.0) + 4.0*v*w + 2.0*u*w + pow(v,2.0) + 2.0*u*v);
   sf(11,0) = 1.0/12.0*(pow(w,4.0) + 2.0*v*pow(w,3.0)); 
   sf(11,1) = 1.0/6.0*pow(w,3.0); 
   sf(11,2) = 1.0/12.0*(4.0*pow(w,3.0) + 6.0*v*pow(w,2.0));
   sf(11,3) = 0.0; 
   sf(11,4) = pow(w,2.0) + v*w;
   sf(11,5) = 1.0/2.0*pow(w,2.0); 
   sf(11,6) = 1.0/2.0*pow(w,2.0);
   return sf;
}

mat setVMU(int n){
    int dotsnumber = 1;
    if (n==1){
        dotsnumber = 1;
        mat vmu(dotsnumber,3);
        vmu << 1.0/3.0 << 1.0/3.0 << 1.0/3.0 << endr;
        return vmu;
    }else if(n==2){
        dotsnumber = 3;
        mat vmu(dotsnumber,3);
        vmu << 1.0/6.0 << 1.0/6.0 << 4.0/6.0 << endr
            << 1.0/6.0 << 4.0/6.0 << 1.0/6.0 << endr
            << 4.0/6.0 << 1.0/6.0 << 1.0/6.0 << endr;
        return vmu;
    }else if(n==3){
        dotsnumber = 4;
        mat vmu(dotsnumber,3);
        vmu << 1.0/3.0 << 1.0/3.0 << 1.0/3.0 << endr
            << 1.0/5.0 << 1.0/5.0 << 3.0/5.0 << endr
            << 1.0/5.0 << 3.0/5.0 << 1.0/5.0 << endr
            << 3.0/5.0 << 1.0/5.0 << 1.0/5.0 << endr;
        /*
        vmu << 1.0/3.0 << 1.0/3.0 << 1.0/3.0 << endr
            << 2.0/15.0 << 11.0/15.0 << 2.0/15.0 << endr
            << 2.0/15.0 << 2.0/15.0 << 11.0/15.0 << endr
            << 11.0/15.0 << 2.0/15.0 << 2.0/15.0 << endr;
        */
        return vmu;
    }else if(n==4){
        dotsnumber = 6;
        mat vmu(dotsnumber,3);
        vmu << 0.44594849091597 << 0.44594849091597 << 0.10810301816807 << endr
            << 0.44594849091597 << 0.10810301816807 << 0.44594849091597 << endr
            << 0.10810301816807 << 0.44594849091597 << 0.44594849091597 << endr
            << 0.09157621350977 << 0.09157621350977 << 0.81684757298046 << endr
            << 0.09157621350977 << 0.81684757298046 << 0.09157621350977 << endr
            << 0.81684757298046 << 0.09157621350977 << 0.09157621350977 << endr;
        return vmu;
    }else if(n==5){
        dotsnumber = 7;
        mat vmu(dotsnumber,3);
        vmu << 0.33333333333333 << 0.33333333333333 << 0.33333333333333 << endr
            << 0.47014206410511 << 0.47014206410511 << 0.05971587178977 << endr
            << 0.47014206410511 << 0.05971587178977 << 0.47014206410511 << endr
            << 0.05971587178977 << 0.47014206410511 << 0.47014206410511 << endr
            << 0.10128650732346 << 0.10128650732346 << 0.79742698535309 << endr
            << 0.10128650732346 << 0.79742698535309 << 0.10128650732346 << endr
            << 0.79742698535309 << 0.10128650732346 << 0.10128650732346 << endr;
        return vmu;
    }else if(n==6){
        dotsnumber = 12;
        mat vmu(dotsnumber,3);
        vmu << 0.24928674517091 << 0.24928674517091 << 0.50142650965818 << endr
            << 0.24928674517091 << 0.50142650965818 << 0.24928674517091 << endr
            << 0.50142650965818 << 0.24928674517091 << 0.24928674517091 << endr
            << 0.06308901449150 << 0.06308901449150 << 0.87382197101700 << endr
            << 0.06308901449150 << 0.87382197101700 << 0.06308901449150 << endr
            << 0.87382197101700 << 0.06308901449150 << 0.06308901449150 << endr
            << 0.31035245103378 << 0.63650249912140 << 0.05314504984482 << endr
            << 0.63650249912140 << 0.05314504984482 << 0.31035245103378 << endr
            << 0.05314504984482 << 0.31035245103378 << 0.63650249912140 << endr
            << 0.63650249912140 << 0.31035245103378 << 0.05314504984482 << endr
            << 0.31035245103378 << 0.05314504984482 << 0.63650249912140 << endr
            << 0.05314504984482 << 0.63650249912140 << 0.31035245103378 << endr;
        return vmu;
    }
}

rowvec setVMUcoefficient(int n){
    int dotsnumber = 1;
    if (n==1){
        dotsnumber = 1;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << 1.0 << endr;
        return vmucoeff;
    }else if(n==2){
        dotsnumber = 3;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << 1.0/3.0 << 1.0/3.0 << 1.0/3.0 << endr;
        return vmucoeff;
    }else if(n==3){
        dotsnumber = 4;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << -0.56250000000000 << 0.52083333333333 << 0.52083333333333 << 0.52083333333333 << endr;
        return vmucoeff;
    }else if(n==4){
        dotsnumber = 6;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << 0.22338158967801 << 0.22338158967801 << 0.22338158967801 << 0.10995174365532 << 0.10995174365532 << 0.10995174365532 << endr;
        return vmucoeff;
    }else if(n==5){
        dotsnumber = 7;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << 0.22500000000000 << 0.13239415278851 << 0.13239415278851 << 0.13239415278851 << 0.12593918054483 << 0.12593918054483 << 0.12593918054483 << endr;
        return vmucoeff;
    }else if(n==6){
        dotsnumber = 12;
        rowvec vmucoeff(dotsnumber);
        vmucoeff << 0.11678627572638 << 0.11678627572638 << 0.11678627572638 << 0.05084490637021 << 0.05084490637021 << 0.05084490637021 << 0.08285107561837 << 0.08285107561837 << 0.08285107561837 << 0.08285107561837 << 0.08285107561837 << 0.08285107561837 << endr;
        return vmucoeff;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about subdivision for regular and irregular patches

struct SubMatrix {
    mat regM; mat regM1; mat regM2; mat regM3; mat regM4;
    mat irregM; mat irregM1; mat irregM2; mat irregM3; mat irregM4;
    mat comregM; mat comregM1; mat comregM2; mat comregM3; mat comregM4;
    mat sudoreg1M; mat sudoreg1M1; mat sudoreg1M2; mat sudoreg1M3; mat sudoreg1M4;
    mat sudoreg2M; mat sudoreg2M1; mat sudoreg2M2; mat sudoreg2M3; mat sudoreg2M4;
};

// for irregular patch, more subdivision is needed. 
// for different sub-element, different new nodes are selected, select-matrix (SM)

// vertex here is the original 12 vertice, so vertex is 12*3 matrix
// M(18,12); mat M1(12,18); mat M2(12,18); mat M3(12,18); mat M4(12,18);
void subdivision_matrix_regular(mat& M, mat& SM1, mat& SM2, mat& SM3, mat& SM4){
    int N = 6; double w=3.0/8.0/N; 
    double a=3.0/8.0; double b=1.0/8.0;
    M = zeros(18,12);
    M(0,0) = a; M(0,1) = b; M(0,2) = b; M(0,3) = a;
    M(1,0) = b; M(1,1) = a; M(1,3) = a; M(1,4) = b;
    M(2,0) = b; M(2,2) = a; M(2,3) = a; M(2,6) = b; 
    M(3,0) = w; M(3,1) = w; M(3,2) = w; M(3,3) = 1.0-N*w; M(3,4) = w; M(3,6) = w; M(3,7) = w;
    M(4,1) = b; M(4,3) = a; M(4,4) = a; M(4,7) = b;
    M(5,2) = a; M(5,3) = b; M(5,5) = b; M(5,6) = a;
    M(6,2) = b; M(6,3) = a; M(6,6) = a; M(6,7) = b;
    M(7,3) = a; M(7,4) = b; M(7,6) = b; M(7,7) = a;
    M(8,3) = b; M(8,4) = a; M(8,7) = a; M(8,8) = b;
    M(9,2) = b; M(9,5) = a; M(9,6) = a; M(9,9) = b;
    M(10,2) = w; M(10,3) = w; M(10,5) = w; M(10,6) = 1.0-N*w; M(10,7) = w; M(10,9) = w; M(10,10) = w; 
    M(11,3) = b; M(11,6) = a; M(11,7) = a; M(11,10) = b;
    M(12,3) = w; M(12,4) = w; M(12,6) = w; M(12,7) = 1.0-N*w; M(12,8) = w; M(12,10) = w; M(12,11) = w; 
    M(13,4) = b; M(13,7) = a; M(13,8) = a; M(13,11) = b;
    M(14,5) = b; M(14,6) = a; M(14,9) = a; M(14,10) = b;
    M(15,6) = a; M(15,7) = b; M(15,9) = b; M(15,10) = a;
    M(16,6) = b; M(16,7) = a; M(16,10) = a; M(16,11) = b;
    M(17,7) = a; M(17,8) = b; M(17,10) = b; M(17,11) = a;
    // To be continued
    SM1 = zeros(12,18);
    Row<int> element1; element1 << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 10 << 11 << 12 << endr;
    for (int i = 0; i < 12; i++){
        SM1(i,element1(i)) = 1.0;
    }
    SM2 = zeros(12,18);
    Row<int> element2; element2 << 5 << 2 << 10 << 6 << 3 << 15 << 11 << 7 << 4 << 16 << 12 << 8 << endr;
    for (int i = 0; i < 12; i++){
        SM2(i,element2(i)) = 1.0;
    }
    SM3 = zeros(12,18);
    Row<int> element3; element3 << 2 << 3 << 5 << 6 << 7 << 9 << 10 << 11 << 12 << 14 << 15 << 16 << endr;
    for (int i = 0; i < 12; i++){ 
        SM3(i,element3(i)) = 1.0;
    }
    SM4 = zeros(12,18);
    Row<int> element4; element4 << 3 << 4 << 6 << 7 << 8 << 10 << 11 << 12 << 13 << 15 << 16 << 17 << endr;
    for (int i = 0; i < 12; i++){ 
        SM4(i,element4(i)) = 1.0;
    }
}

// vertex here is the original 11 vertice, so vertex is 11*3 matrix
// M(17,11); mat M1(12,17); mat M2(12,17); mat M3(12,17); mat M4(11,17);
// sub-element 1 is still irregular patch
void subdivision_matrix_irregular(mat& M, mat& SM1, mat& SM2, mat& SM3, mat& SM4){
    int N=6; double w=3.0/8.0/N; // w=1/N*(5/8-(3/8+1/4*cos(2*pi/N))^2);
    int N1=5; double w1=3.0/8.0/N1; // w1=1/N1*(5/8-(3/8+1/4*cos(2*pi/N1))^2);
    double a=3.0/8.0; double b=1.0/8.0;
    M = zeros(17,11);
    M << a << b << a << b << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr
      << b << a << a << 0 << 0 << b << 0 << 0 << 0 << 0 << 0 << endr
      << w1 << w1 << 1.0-N1*w1 << w1 << 0 << w1 << w1 << 0 << 0 << 0 << 0 << endr
      << b << 0 << a << a << 0 << 0 << b << 0 << 0 << 0 << 0 << endr
      << 0 << a << b << 0 << b << a << 0 << 0 << 0 << 0 << 0 << endr
      << 0 << b << a << 0 << 0 << a << b << 0 << 0 << 0 << 0 << endr
      << 0 << 0 << a << b << 0 << b << a << 0 << 0 << 0 << 0 << endr
      << 0 << 0 << b << a << 0 << 0 << a << b << 0 << 0 << 0 << endr
      << 0 << b << 0 << 0 << a << a << 0 << 0 << b << 0 << 0 << endr
      << 0 << w << w << 0 << w << 1.0-N*w <<  w << 0 << w << w << 0 << endr
      << 0 << 0 << b << 0 << 0 << a << a << 0 << 0 << b << 0 << endr
      << 0 << 0 << w << w << 0 << w << 1.0-N*w << w << 0 << w << w << endr
      << 0 << 0 << 0 << b << 0 << 0 << a << a << 0 << 0 << b << endr
      << 0 << 0 << 0 << 0 << b << a << 0 << 0 << a << b << 0 << endr
      << 0 << 0 << 0 << 0 << 0 << a << b << 0 << b << a << 0 << endr
      << 0 << 0 << 0 << 0 << 0 << b << a << 0 << 0 << a << b << endr
      << 0 << 0 << 0 << 0 << 0 << 0 << a << b << 0 << b << a << endr;
    SM1=zeros(11,17);
    Row<int> element1; element1 << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 9 << 10 << 11 << endr;
    for (int i = 0; i < 11; i++){ 
        SM1(i,element1(i)) = 1.0;
    }
    SM2=zeros(12,17);
    Row<int> element2; element2 << 4 << 1 << 9 << 5 << 2 << 14 << 10 << 6 << 3 << 15 << 11 << 7 << endr;
    for (int i = 0; i < 12; i++){
        SM2(i,element2(i)) = 1.0;
    }
    SM3=zeros(12,17);
    Row<int> element3; element3 << 1 << 2 << 4 << 5 << 6 << 8 << 9 << 10 << 11 << 13 << 14 << 15 << endr;
    for (int i = 0; i < 12; i++){ 
        SM3(i,element3(i)) = 1.0;
    }
    SM4=zeros(12,17);
    Row<int> element4; element4 << 2 << 3 << 5 << 6 << 7 << 9 << 10 << 11 << 12 << 14 << 15 << 16 << endr;
    for (int i=0; i<12; i++){
        SM4(i,element4(i)) = 1.0;
    }
}

// vertex here is the original 13 vertice, so vertex is 13*3 matrix
// M(19,13); mat M1(12,19); mat M2(12,19); mat M3(12,19); mat M4(13,19);
// sub-element 4 is still complex patch.
void subdivision_matrix_complex(mat& M, mat& SM1, mat& SM2, mat& SM3, mat& SM4){
    int N=6; double w=3.0/8.0/N; 
    int N1=7; double w1=3.0/8.0/N1; 
    double a=3.0/8.0; double b=1.0/8.0;
    M = zeros(19,13);
    M(0,0) = a; M(0,1) = b; M(0,2) = b; M(0,3) = a;
    M(1,0) = b; M(1,1) = a; M(1,3) = a; M(1,4) = b;
    M(2,0) = b; M(2,2) = a; M(2,3) = a; M(2,6) = b;
    M(3,0) = w; M(3,1) = w; M(3,2) = w; M(3,3) = 1.0-N*w; M(3,4) = w; M(3,6) = w; M(3,7) = w;
    M(4,1) = b; M(4,3) = a; M(4,4) = a; M(4,7) = b;
    M(5,2) = a; M(5,3) = b; M(5,5) = b; M(5,6) = a;
    M(6,2) = b; M(6,3) = a; M(6,6) = a; M(6,7) = b;
    M(7,3) = a; M(7,4) = b; M(7,6) = b; M(7,7) = a;
    M(8,3) = b; M(8,4) = a; M(8,7) = a; M(8,8) = b;
    M(9,2) = b; M(9,5) = a; M(9,6) = a; M(9,9) = b;
    M(10,2) = w; M(10,3) = w; M(10,5) = w; M(10,6) = 1.0-N*w; M(10,7) = w; M(10,9) = w; M(10,10) = w;
    M(11,3) = b; M(11,6) = a; M(11,7) = a; M(11,10) = b;
    M(12,3) = w1; M(12,4) = w1; M(12,6) = w1; M(12,7) = 1.0-N1*w1; M(12,8) = w1; M(12,10) = w1; M(12,11) = w1; M(12,12) = w1;
    M(13,4) = b; M(13,7) = a; M(13,8) = a; M(13,12) = b;
    M(14,5) = b; M(14,6) = a; M(14,9) = a; M(14,10) = b;
    M(15,6) = a; M(15,7) = b; M(15,9) = b; M(15,10) = a;
    M(16,6) = b; M(16,7) = a; M(16,10) = a; M(16,11) = b;
    M(17,7) = a; M(17,10) = b; M(17,11) = a; M(17,12) = b;
    M(18,7) = a; M(18,8) = b; M(18,11) = b; M(18,12) = a;
    SM1=zeros(12,19);
    Row<int> element1; element1 << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 10 << 11 << 12 << endr;
    for (int i=0; i<12; i++){
        SM1(i,element1(i)) = 1.0;
    }
    SM2=zeros(12,19);
    Row<int> element2; element2 << 5 << 2 << 10 << 6 << 3 << 15 << 11 << 7 << 4 << 16 << 12 << 8 << endr;
    for (int i = 0; i < 12; i++){
        SM2(i,element2(i)) = 1.0;
    }
    SM3=zeros(12,19);
    Row<int> element3; element3 << 2 << 3 << 5 << 6 << 7 << 9 << 10 << 11 << 12 << 14 << 15 << 16 << endr;
    for (int i = 0; i < 12; i++){ 
        SM3(i,element3(i)) = 1.0;
    }
    SM4=zeros(13,19);
    Row<int> element4; element4 << 3 << 4 << 6 << 7 << 8 << 10 << 11 << 12 << 13 << 15 << 16 << 17 << 18 << endr;
    for (int i = 0; i < 13; i++){ 
        SM4(i,element4(i)) = 1.0;
    }
}

// vertex here is the original 12 vertice, so vertex is 12*3 matrix
// M(18,12); mat M1(11,18); mat M2(12,18); mat M3(12,18); mat M4(13,18);
// sub-element 1 is irregular patch; sub-element 4 is complex patch. 
void subdivision_matrix_sudoregular1(mat& M, mat& SM1, mat& SM2, mat& SM3, mat& SM4){
    int N = 6; double w = 3.0/8.0/N; 
    int N1 = 5; double w1 = 3.0/8.0/N1;
    int N2 = 7; double w2 = 3.0/8.0/N2; 
    double a = 3.0/8.0; double b = 1.0/8.0;
    M = zeros(18,12);
    M(0,0) = a; M(0,1) = b; M(0,2) = a; M(0,3) = b;
    M(1,0) = b; M(1,1) = a; M(1,2) = a; M(1,5) = b;
    M(2,0) = w1; M(2,1) = w1; M(2,2) = 1.0-N1*w1; M(2,3) = w1; M(2,5) = w1; M(2,6) = w1; 
    M(3,0) = b; M(3,2) = a; M(3,3) = a; M(3,6) = b; 
    M(4,1) = a; M(4,2) = b; M(4,4) = b; M(4,5) = a;
    M(5,1) = b; M(5,2) = a; M(5,5) = a; M(5,6) = b;
    M(6,2) = a; M(6,3) = b; M(6,5) = b; M(6,6) = a;
    M(7,2) = b; M(7,3) = a; M(7,6) = a; M(7,7) = b;
    M(8,1) = b; M(8,4) = a; M(8,5) = a; M(8,8) = b;
    M(9,1) = w2; M(9,2) = w2; M(9,4) = w2; M(9,5) = 1.0-N2*w2; M(9,6) = w2; M(9,8) = w2; M(9,9) = w2; M(9,10) = w2;
    M(10,2) = b; M(10,5) = a; M(10,6) = a; M(10,10) = b; 
    M(11,2) = w; M(11,3) = w; M(11,5) = w; M(11,6) = 1.0-N*w; M(11,7) = w; M(11,10) = w; M(11,11) = w; 
    M(12,3) = b; M(12,6) = a; M(12,7) = a; M(12,11) = b; 
    M(13,4) = b; M(13,5) = a; M(13,8) = a; M(13,9) = b;
    M(14,5) = a; M(14,8) = b; M(14,9) = a; M(14,10) = b; 
    M(15,5) = a; M(15,6) = b; M(15,9) = b; M(15,10) = a;
    M(16,5) = b; M(16,6) = a; M(16,10) = a; M(16,11) = b;
    M(17,6) = a; M(17,7) = b; M(17,10) = b; M(17,11) = a;
    SM1 = zeros(11,18);
    Row<int> element1; element1 << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 9 << 10 << 11 << endr;
    for (int i = 0; i < 11; i++){
        SM1(i,element1(i)) = 1.0;
    }
    SM2 = zeros(12,18);
    Row<int> element2; element2 << 4 << 1 << 9 << 5 << 2 << 15 << 10 << 6 << 3 << 16 << 11 << 7 << endr;
    for (int i = 0; i < 12; i++){
        SM2(i,element2(i)) = 1.0;
    }
    SM3 = zeros(12,18);
    Row<int> element3; element3 << 2 << 3 << 5 << 6 << 7 << 9 << 10 << 11 << 12 << 15 << 16 << 17 << endr;
    for (int i = 0; i < 12; i++){ 
        SM3(i,element3(i)) = 1.0;
    }
    SM4 = zeros(13,18);
    Row<int> element4; element4 << 11 << 16 << 6 << 10 << 15 << 2 << 5 << 9 << 14 << 1 << 4 << 8 << 13 << endr;
    for (int i = 0; i < 13; i++){ 
        SM4(i,element4(i)) = 1.0;
    }
}

// vertex here is the original 12 vertice, so vertex is 12*3 matrix
// M(18,12); mat M1(11,18); mat M2(12,18); mat M3(12,18); mat M4(13,18);
// sub-element 1 is irregular patch; sub-element 4 is complex patch. 
void subdivision_matrix_sudoregular2(mat& M, mat& SM1, mat& SM2, mat& SM3, mat& SM4){
    int N = 6; double w=3.0/8.0/N; 
    int N1 = 5; double w1=3.0/8.0/N1;
    int N2 = 7; double w2 = 3.0/8.0/N2; 
    double a=3.0/8.0; double b=1.0/8.0;
    M = zeros(18,12);
    M(0,0) = a; M(0,1) = b; M(0,2) = a; M(0,3) = b;
    M(1,0) = b; M(1,1) = a; M(1,2) = a; M(1,5) = b;
    M(2,0) = w1; M(2,1) = w1; M(2,2) = 1.0-N1*w1; M(2,3) = w1; M(2,5) = w1; M(2,6) = w1; 
    M(3,0) = b; M(3,2) = a; M(3,3) = a; M(3,6) = b; 
    M(4,1) = a; M(4,2) = b; M(4,4) = b; M(4,5) = a;
    M(5,1) = b; M(5,2) = a; M(5,5) = a; M(5,6) = b;
    M(6,2) = a; M(6,3) = b; M(6,5) = b; M(6,6) = a;
    M(7,2) = b; M(7,3) = a; M(7,6) = a; M(7,7) = b;
    M(8,1) = b; M(8,4) = a; M(8,5) = a; M(8,8) = b;
    M(9,1) = w; M(9,2) = w; M(9,4) = w; M(9,5) = 1.0-N*w; M(9,6) = w; M(9,8) = w; M(9,9) = w;
    M(10,2) = b; M(10,5) = a; M(10,6) = a; M(10,9) = b; 
    M(11,2) = w2; M(11,3) = w2; M(11,5) = w2; M(11,6) = 1.0-N2*w2; M(11,7) = w2; M(11,9) = w2; M(11,10) = w2; M(11,11) = w2;
    M(12,3) = b; M(12,6) = a; M(12,7) = a; M(12,11) = b; 
    M(13,4) = b; M(13,5) = a; M(13,8) = a; M(13,9) = b;
    M(14,5) = a; M(14,6) = b; M(14,8) = b; M(14,9) = a;
    M(15,5) = b; M(15,6) = a; M(15,9) = a; M(15,10) = b;
    M(16,6) = a; M(16,9) = b; M(16,10) = a; M(16,11) = b;
    M(17,6) = a; M(17,7) = b; M(17,10) = b; M(17,11) = a;
    SM1 = zeros(11,18);
    Row<int> element1; element1 << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 9 << 10 << 11 << endr;
    for (int i = 0; i < 11; i++){
        SM1(i,element1(i)) = 1.0;
    }
    SM2 = zeros(12,18);
    Row<int> element2; element2 << 4 << 1 << 9 << 5 << 2 << 14 << 10 << 6 << 3 << 15 << 11 << 7 << endr;
    for (int i = 0; i < 12; i++){
        SM2(i,element2(i)) = 1.0;
    }
    SM3 = zeros(12,18);
    Row<int> element3; element3 << 1 << 2 << 4 << 5 << 6 << 8 << 9 << 10 << 11 << 13 << 14 << 15 << endr;
    for (int i = 0; i < 12; i++){ 
        SM3(i,element3(i)) = 1.0;
    }
    SM4 = zeros(13,18);
    Row<int> element4; element4 << 2 << 3 << 5 << 6 << 7 << 9 << 10 << 11 << 12 << 14 << 15 << 16 << 17 << endr;
    for (int i = 0; i < 13; i++){ 
        SM4(i,element4(i)) = 1.0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about algebra calculation

void trans_time(mat a, mat b, mat& out){
    int rowa = a.n_rows; int cola = a.n_cols;
    int rowb = b.n_rows; int colb = b.n_cols;
    //mat out(cola,colb);
    if (rowa != rowb){
        cout<<"wrong. matrix dimentions are not correct during trans_time!"<<endl;
        exit(1);
    }else{
        for (int i = 0; i < cola; i++){
            for (int j = 0; j < colb; j++){
                double temp = 0.0;
                for (int k = 0; k < rowa; k++){
                    temp = temp + a(k,i)*b(k,j);
                }
                out(i,j) = temp;
            }
        }
    }
    //return out;
}

vec force3_scale(Force3Layers force3, bool isBilayerModel){
    if ( isBilayerModel == false ){
        int numout = force3.outlayer.n_rows;
        vec out(numout);
        #pragma omp parallel for
        for (int i = 0; i < numout; i++){
            out(i) = norm(force3.outlayer.row(i),2);
        }
        return out;
    }else{
        int numin = force3.inlayer.n_rows;
        int nummid = force3.midlayer.n_rows;
        int numout = force3.outlayer.n_rows;
        vec out(numin + nummid + numout);
        #pragma omp parallel for
        for (int i = 0; i < numin; i++){
            out(i) = norm(force3.inlayer.row(i),2);
        }
        #pragma omp parallel for
        for (int i = numin; i < numin+nummid; i++){
            out(i) = norm(force3.midlayer.row(i-numin),2);
        }
        #pragma omp parallel for
        for (int i = numin+nummid; i < numin+nummid+numout; i++){
            out(i) = norm(force3.outlayer.row(i-numin-nummid),2);
        }
        return out;
    }
}

vec toscale(mat forcein){
    int num = forcein.n_rows;
    vec out(num);
    #pragma omp parallel for
    for (int i = 0; i < num; i++){
        out(i) = norm(forcein.row(i),2);
    }
    return out;
}

rowvec tovector(mat vertex){
    int nodenum = vertex.n_rows;
    rowvec vertex_row(3*nodenum);
    #pragma omp parallel for
    for (int i = 0; i < nodenum; i++){
        vertex_row(3*i+0) = vertex(i,0);
        vertex_row(3*i+1) = vertex(i,1);
        vertex_row(3*i+2) = vertex(i,2);
    }
    return vertex_row;
}

rowvec changeForce3ToVector(Force3Layers force3, bool isBilayerModel){
    int nodenum = force3.outlayer.n_rows;
    if ( isBilayerModel == false ){
        rowvec out(3*nodenum);
        #pragma omp parallel for
        for (int i = 0; i < nodenum; i++){
            out(3*i+0) = force3.outlayer(i,0);
            out(3*i+1) = force3.outlayer(i,1);
            out(3*i+2) = force3.outlayer(i,2);
        }
        return out;
    }else{
        rowvec out(3*nodenum * 3);
        #pragma omp parallel for
        for (int i = 0; i < nodenum; i++){
            out(3*i+0) = force3.inlayer(i,0);
            out(3*i+1) = force3.inlayer(i,1);
            out(3*i+2) = force3.inlayer(i,2);
        }
        #pragma omp parallel for
        for (int i = nodenum; i < nodenum*2; i++){
            out(3*i+0) = force3.midlayer(i-nodenum,0);
            out(3*i+1) = force3.midlayer(i-nodenum,1);
            out(3*i+2) = force3.midlayer(i-nodenum,2);
        }
        #pragma omp parallel for
        for (int i = nodenum*2; i < nodenum*3; i++){
            out(3*i+0) = force3.outlayer(i-nodenum*2,0);
            out(3*i+1) = force3.outlayer(i-nodenum*2,1);
            out(3*i+2) = force3.outlayer(i-nodenum*2,2);
        }
        return out;
    }
}

mat tomatrix(rowvec vertex_row){
    int nodenum = vertex_row.n_cols/3;
    mat vertex(nodenum,3);
    #pragma omp parallel for
    for (int i = 0; i < nodenum; i++){
        vertex(i,0) = vertex_row(3*i+0);
        vertex(i,1) = vertex_row(3*i+1);
        vertex(i,2) = vertex_row(3*i+2);
    }
    return vertex;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following are about updates of boundary vertices, including their nodal forces and positions.

mat manage_ghost_force(mat force, Mat<int> face, Param param){
    mat fxyz = force;
    if ( param.isFlatMembrane == false ) {
        return fxyz;
    }
    double sidex = param.sideX; double sidey = param.sideY; double l = param.l;
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double a = dx;
    double dy = sqrt(3)/2 * a; 
    int m = round(sidey/dy);      // y axis division 
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    int facenum = m*n*2;
    if ( param.isBoundaryFixed == true ){
        #pragma omp parallel for 
        for (int i = 0; i < facenum; i++){
            if ( i < param.isBoundaryFace.n_cols && param.isBoundaryFace(i) == 1 ){
                int node1 = face(i,0);
                int node2 = face(i,1);
                int node3 = face(i,2);
                fxyz.row(node1) = force.row(node1) * 0.0;
                fxyz.row(node2) = force.row(node2) * 0.0;
                fxyz.row(node3) = force.row(node3) * 0.0;
            }
        }
    }else if ( param.isBoundaryPeriodic == true ){
        #pragma omp parallel for 
        for (int i = 0; i < vertexnum; i++){
            if ( i < param.isGhostVertex.n_cols && param.isGhostVertex(i) == 1 )
                fxyz.row(i) = force.row(i) * 0.0;
        }
    }else if ( param.isBoundaryFree == true ){   
        #pragma omp parallel for 
        for ( int i = 0; i < vertexnum; i++ ){
            if ( param.isBoundaryVertex(i) == 1 ){
                fxyz.row(i) = force.row(i) * 0.0;
            }
        }
    }
    return fxyz;
}

mat update_vertex(mat vertex, Mat<int> face, double a, mat force, Param param){
    mat vertexnew = vertex + a * force;
    if ( param.isFlatMembrane == false ){
        return vertexnew;
    }

    double sidex = param.sideX; double sidey = param.sideY; double l = param.l;
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double aa = dx;
    double dy = sqrt(3)/2 * aa; 
    int m = round(sidey/dy);      // y axis division 
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    int facenum = m*n*2;

    if ( param.isBoundaryFixed == true ){
        Row<int> isBoundaryFace = param.isBoundaryFace;
        #pragma omp parallel for 
        for (int i = 0; i < facenum; i++){
            if ( isBoundaryFace(i) != 1 )  
                continue;
            int node1 = face(i,0);
            int node2 = face(i,1);
            int node3 = face(i,2);
            vertexnew.row(node1) = vertex.row(node1);
            vertexnew.row(node2) = vertex.row(node2);
            vertexnew.row(node3) = vertex.row(node3);
        }
    }else if ( param.isBoundaryPeriodic == true ){
        vector<int> ghostPartner = param.ghostPartner;
        #pragma omp parallel for 
        for ( int i = 0; i < vertexnum; i++ ){
            if ( ghostPartner[i] != -1 ){   
                vertexnew.row(i) = vertex.row(i) + ( vertexnew.row(ghostPartner[i]) - vertex.row(ghostPartner[i]) );
            }
        }
    }else if ( param.isBoundaryFree == true ){
        vector<vector<int>> freePartners = param.freePartners;
        for ( int i = 0; i < vertexnum; i++ ){ // Note: forbid to use parallelization here!
            if ( freePartners[i][0] != -1 ){
                int index1 = i; 
                int index2 = freePartners[i][0];
                int index3 = freePartners[i][1]; 
                int index4 = freePartners[i][2];
                vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
            }
        }
        
    }
    
    return vertexnew;
}
/*
mat update_vertex(mat vertex, Mat<int> face, double a, mat force, Param param, Row<int> isGhostFace, Row<int> isGhostVertex){
    double sidex = param.sideX; double sidey = param.sideY; double l = param.l;
    int n = round(sidex/l); double dx = sidex/n; // x axis division
    double aa = dx;
    double dy = sqrt(3)/2 * aa; 
    int m = round(sidey/dy);      // y axis division 
    if ( pow(-1.0,m) < 0.0 ) { m = m + 1; }
    
    int vertexnum = (n+1)*(m+1);
    int facenum = m*n*2;
    
    mat vertexnew = vertex + a * force;

    if ( param.isBoundaryFixed == true ){
        #pragma omp parallel for 
        for (int i = 0; i < facenum; i++){
            if ( isGhostFace(i) != 1 )
                continue;
            int node1 = face(i,0);
            int node2 = face(i,1);
            int node3 = face(i,2);
            vertexnew.row(node1) = vertex.row(node1);
            vertexnew.row(node2) = vertex.row(node2);
            vertexnew.row(node3) = vertex.row(node3);
        }
    }else if ( param.isBoundaryPeriodic == true ){
       #pragma omp parallel for 
       for (int i = 0; i < n+1; i++){ // for top and bottom ghost vertices
            int index0 = (n+1)*0 + i;      // bottom ghost vertices
            int index1 = (n+1)*1 + i;
            int index2 = (n+1)*2 + i;
            int index00 = (n+1)*(m-6) + i; // top real vertices
            int index11 = (n+1)*(m-5) + i;
            int index22 = (n+1)*(m-4) + i;
            vertexnew.row(index0) = vertex.row(index0) + ( vertexnew.row(index00)-vertex.row(index00) );
            vertexnew.row(index1) = vertex.row(index1) + ( vertexnew.row(index11)-vertex.row(index11) );
            vertexnew.row(index2) = vertex.row(index2) + ( vertexnew.row(index22)-vertex.row(index22) );
            index0 = (n+1)*(m-2) + i; // top ghost vertices
            index1 = (n+1)*(m-1) + i;
            index2 = (n+1)*(m-0) + i;
            index00 = (n+1)*4 + i;    // bottom real vertices
            index11 = (n+1)*5 + i;
            index22 = (n+1)*6 + i;
            vertexnew.row(index0) = vertex.row(index0) + ( vertexnew.row(index00)-vertex.row(index00) );
            vertexnew.row(index1) = vertex.row(index1) + ( vertexnew.row(index11)-vertex.row(index11) );
            vertexnew.row(index2) = vertex.row(index2) + ( vertexnew.row(index22)-vertex.row(index22) );
        }
        #pragma omp parallel for 
       for (int j = 0; j < m+1; j++){ // for left and right ghost vertices
            int index0 = (n+1)*j + 0; // left ghost
            int index1 = (n+1)*j + 1;
            int index2 = (n+1)*j + 2;
            int index00 = (n+1)*j + n-6; // right real
            int index11 = (n+1)*j + n-5;
            int index22 = (n+1)*j + n-4;
            vertexnew.row(index0) = vertex.row(index0) + ( vertexnew.row(index00)-vertex.row(index00) );
            vertexnew.row(index1) = vertex.row(index1) + ( vertexnew.row(index11)-vertex.row(index11) );
            vertexnew.row(index2) = vertex.row(index2) + ( vertexnew.row(index22)-vertex.row(index22) );
            index0 = (n+1)*j + n; // right ghost
            index1 = (n+1)*j + n-1;
            index2 = (n+1)*j + n-2;
            index00 = (n+1)*j + 6; // left real
            index11 = (n+1)*j + 5;
            index22 = (n+1)*j + 4;
            vertexnew.row(index0) = vertex.row(index0) + ( vertexnew.row(index00)-vertex.row(index00) );
            vertexnew.row(index1) = vertex.row(index1) + ( vertexnew.row(index11)-vertex.row(index11) );
            vertexnew.row(index2) = vertex.row(index2) + ( vertexnew.row(index22)-vertex.row(index22) );
        }
    }else if ( param.isBoundaryFree == true ){
        int index1, index2, index3, index4;
        // left side
        for (int j = 2; j < m ; j++ ){
            if ( pow(-1.0,j) > 0.0 ){
                index1 = (n+1)*j + 0;
                index2 = (n+1)*j + 1;
                index3 = (n+1)*(j-1) + 1;
                index4 = (n+1)*(j-1) + 2;
            }else{
                index1 = (n+1)*j + 0;
                index2 = (n+1)*j + 1;
                index3 = (n+1)*(j-1) + 0;
                index4 = (n+1)*(j-1) + 1;
            }
            vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        }
        index1 = (n+1)*1+0; 
        index2 = (n+1)*2+0;
        index3 = (n+1)*1+1;
        index4 = (n+1)*2+1;
        vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        // right side
        index1 = (n+1)*1 + n; 
        index2 = (n+1)*2 + n-1;
        index3 = (n+1)*1 + n-1;
        index4 = (n+1)*2 + n-2;
        vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        for (int j = 2; j < m ; j++ ){
            if ( pow(-1.0,j) > 0.0 ){
                index1 = (n+1)*j + n;
                index2 = (n+1)*j + n-1;
                index3 = (n+1)*(j-1) + n;
                index4 = (n+1)*(j-1) + n-1;
            }else{
                index1 = (n+1)*j + n;
                index2 = (n+1)*j + n-1;
                index3 = (n+1)*(j-1) + n-1;
                index4 = (n+1)*(j-1) + n-2;
            }
            vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        }
        // bottom 
        for (int i = 0; i < n; i++){
            index1 = (n+1)*0 + i;
            index2 = (n+1)*1 + i;
            index3 = (n+1)*1 + i+1;
            index4 = (n+1)*2 + i;
            vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        }
        index1 = (n+1)*0 + n; 
        index2 = (n+1)*0 + n-1;
        index3 = (n+1)*1 + n;
        index4 = (n+1)*1 + n-1;
        vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        // top 
        for (int i = 1; i < n+1; i++){
            index1 = (n+1)*m + i;
            index2 = (n+1)*(m-1) + i;
            index3 = (n+1)*(m-1) + i+1;
            index4 = (n+1)*(m-2) + i;
            vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
        }
        index1 = (n+1)*m + n; 
        index2 = (n+1)*m + n-1;
        index3 = (n+1)*(m-1) + n;
        index4 = (n+1)*(m-1) + n-1;
        vertexnew.row(index1) = vertexnew.row(index2) + vertexnew.row(index3) - vertexnew.row(index4);
    }
    
    return vertexnew;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for finer_local_mesh

struct LocalFinerMesh{
    bool buildLocalFinerMesh = false;
    int centerVertexIndex;
    Row<int> sixVerticeHexagon;
    Row<int> faceIndex;
    //Row<int> outBoundaryRingFaceIndex;
};

vector<bool> determine_isLocallyFinerFace(Mat<int> face, LocalFinerMesh localFinerMesh){
    int facenum = face.n_rows;
    vector<bool> isLocallyFinerFace(facenum, false);
    for ( int i = 0; i < localFinerMesh.faceIndex.n_cols; i++ ){
        int faceindex = localFinerMesh.faceIndex(i);
        isLocallyFinerFace[faceindex] = true;
    }
    return isLocallyFinerFace;
}

vector<bool> determine_isLocallyFinerFaceMostEdge(Mat<int> face, LocalFinerMesh localFinerMesh, Mat<int> closest_nodes){
    int facenum = face.n_rows;
    vector<bool> isLocallyFinerFaceMostEdge(facenum, false);
    for ( int i = 0; i < localFinerMesh.faceIndex.n_cols; i++ ){
        int faceindex = localFinerMesh.faceIndex(i);
        for ( int j = 0; j < 3; j++ ){
            int nodetmp = face(faceindex,j);
            if (closest_nodes(nodetmp,6) != -1 ){ // this node has 7 valence.
                isLocallyFinerFaceMostEdge[faceindex] = true;
            }
        }
    }
    return isLocallyFinerFaceMostEdge;
}

// subdivide one triangle for one time, so it is divided into four smaller triangles. Meanwhile the vertex and face matrix need to be updated! 
void subdivide_one_triangle(mat& vertex, Mat<int>& face, int originalVertexNumber, int faceTarget){
    int node0 = face(faceTarget,0);
    int node1 = face(faceTarget,1);
    int node2 = face(faceTarget,2);

    rowvec vertexNew0 = 1.0/2.0 * (vertex.row(node0) + vertex.row(node1));
    rowvec vertexNew1 = 1.0/2.0 * (vertex.row(node1) + vertex.row(node2));
    rowvec vertexNew2 = 1.0/2.0 * (vertex.row(node2) + vertex.row(node0));  
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // decide the new vertex index. Some may be already existed
    // new vertex 0
    int nodeNew0 = vertex.n_rows;
    bool isNodeExist = false;
    for (int i = originalVertexNumber-1; i < vertex.n_rows; i++ ){
        rowvec dist = vertex.row(i) - vertexNew0;
        if ( norm(dist,2) < 1.0e-9 ){ // nodeNew0 is already in the vertex matrix
            nodeNew0 = i; 
            isNodeExist = true;
        }
    }
    if ( isNodeExist == false ){
        vertex.resize(vertex.n_rows + 1, 3);
        vertex.row(vertex.n_rows - 1) = vertexNew0;
    }
    // new vertex 1
    int nodeNew1 = vertex.n_rows;
    isNodeExist = false;
    for (int i = originalVertexNumber-1; i < vertex.n_rows; i++ ){
        rowvec dist = vertex.row(i) - vertexNew1;
        if ( norm(dist,2) < 1.0e-9 ){ // nodeNew1 is already in the vertex matrix
            nodeNew1 = i;
            isNodeExist = true; 
        }
    }
    if ( isNodeExist == false ){
        vertex.resize(vertex.n_rows + 1, 3);
        vertex.row(vertex.n_rows - 1) = vertexNew1;
    } 
    // new vertex 2
    int nodeNew2 = vertex.n_rows;
    isNodeExist = false;
    for (int i = originalVertexNumber-1; i < vertex.n_rows; i++ ){
        rowvec dist = vertex.row(i) - vertexNew2;
        if ( norm(dist,2) < 1.0e-9 ){ // nodeNew2 is already in the vertex matrix
            nodeNew2 = i;
            isNodeExist = true; 
        }
    }
    if ( isNodeExist == false ){
        vertex.resize(vertex.n_rows + 1, 3);
        vertex.row(vertex.n_rows - 1) = vertexNew2;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // build the new faces
    Row<int> facetmp(3); 
    facetmp << nodeNew0 << nodeNew1 << nodeNew2 << endr;
    face.row(faceTarget) = facetmp;

    facetmp << node0 << nodeNew0 << nodeNew2 << endr;
    face.resize(face.n_rows + 1, 3);
    face.row(face.n_rows - 1) = facetmp;

    facetmp << node1 << nodeNew1 << nodeNew0 << endr;
    face.resize(face.n_rows + 1, 3);
    face.row(face.n_rows - 1) = facetmp;

    facetmp << node2 << nodeNew2 << nodeNew1 << endr;
    face.resize(face.n_rows + 1, 3);
    face.row(face.n_rows - 1) = facetmp;
}

// check whether a is one element of matrix A. Only works for int
bool isElement( Mat<int> A, int a ){
    bool iselement = false;
    for ( int i = 0; i < A.n_rows; i++ ){
        for ( int j = 0; j < A.n_cols; j++ ){
            if ( a == A(i,j) ){
                iselement = true;
            }
        }
    }
    return iselement;
}

// select the local faces around the centerVertexIndex for further subdivision or finer mesh. 
Mat<int> select_local_face_for_finer(Mat<int> face, mat vertex, Mat<int> faces_with_nodei, int centerVertexIndex, int localRounds){
    int matrixSize = ( 2 * (localRounds -1) + 1 ) * 6;
    Mat<int> selectFaces(localRounds, matrixSize); selectFaces.fill(-1);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // the first round should be the faces_with_centerVertexIndex
    for ( int i = 0; i < faces_with_nodei.n_cols; i++ ){
        selectFaces(0,i) = faces_with_nodei(centerVertexIndex, i);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // next rounds should be outside the previous rounds
    for ( int i = 1; i < localRounds; i++ ){
        int counts = -1;
        for (int j = 0; j < matrixSize; j++){
            int facetmp = selectFaces(i-1, j); // one face of previous round
            if ( facetmp == -1 ) continue;
            for ( int k = 0; k < 3; k++ ){ // for loop all 3 vertice of the facetmp
                int vertextmp = face(facetmp,k);
                for ( int n = 0; n < faces_with_nodei.n_cols; n++ ){ // check each neighbor face of the vertextmp
                    int faceTarget = faces_with_nodei(vertextmp, n);
                    if ( faceTarget == -1 ) continue;
                    bool isCounted = isElement( selectFaces, faceTarget ); // check whether it is already counted
                    if ( isCounted == false ){
                        counts = counts + 1;
                        selectFaces(i, counts) = faceTarget;
                    }
                }
            }
        }
    } 
    return selectFaces;
}

void finer_local_mesh(int centerVertexIndex, int localRounds, mat& vertex, Mat<int>& face, Mat<int>& faces_with_nodei, LocalFinerMesh& localFinerMesh, Row<int>& isBoundaryNode, Row<int>& isBoundaryFace, Row<int>& isGhostNode, Row<int>& isGhostFace, vector<int>& ghostPartner, vector<vector<int>>& freePartners ){
    // find a center vertex, then select several rounds of faces around it. Note, all the faces must be regular patches. 

    int vertexnum = vertex.n_rows;
    int facenum = face.n_rows;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find how many round of faces need to be subdivided around the centerVertex.
    if ( localRounds < 2 ){
        cout<< "Wrong: In finer_local_mesh, too small local area is selected for subdivision!" << endl;
        exit(0);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find each round of faces around the centerVertex.
    Mat<int> selectLocalFaces = select_local_face_for_finer(face, vertex, faces_with_nodei, centerVertexIndex, localRounds);
   
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // subdivide each select face except the last round! 
    // the subdivision makes each triangle into 4 smaller triangles! So the vertex and face matrix need to be updated at the same time!
    for ( int i = 0; i < selectLocalFaces.n_rows - 1; i++ ){
        for ( int j = 0; j < selectLocalFaces.n_cols; j++ ){
            int facetmp = selectLocalFaces(i,j); 
            if ( facetmp == -1 ) continue;
            subdivide_one_triangle(vertex, face, vertexnum, facetmp);
        }
    }
  
    // For faces of the last round, some of them need to be bisected, but some do not!
    for ( int i = 0; i < selectLocalFaces.n_cols; i++ ){
        int facetmp = selectLocalFaces(selectLocalFaces.n_rows-1, i);
        if ( facetmp == -1 ) continue;
        /////////////////////////////////////////////
        // check whether this face needs to be bisect! The criterion is: this face center to the centerVertexIndex is further than two of its 3 vertices 
        rowvec centertmp(3); centertmp.fill(0.0); // center of the face
        for (int j = 0; j < 3; j++){
            int node = face(facetmp, j);
            centertmp = centertmp + 1.0/3.0 * vertex.row(node);
        }
        double distCenter = norm(centertmp - vertex.row(centerVertexIndex), 2);
        int count = 0;
        Row<int> threeVertexIndex(3); threeVertexIndex.fill(-1);
        for (int j = 0; j < 3; j++){
            int node = face(facetmp, j);
            double disttmp = norm(vertex.row(node) - vertex.row(centerVertexIndex), 2);
            if ( distCenter > disttmp ){
                count = count + 1;
                threeVertexIndex(count-1) = node;
            }
        }

        /////////////////////////////////////////////
        // bisect the face
        if ( count == 2 ){
            // order the 3 vertice, make the first two on the bisection line
            int node0 = threeVertexIndex(0);
            int node1 = threeVertexIndex(1);
            int node2;
            for (int j = 0; j < 3; j++){
                int nodetmp = face(facetmp, j);
                if ( nodetmp != node0 && nodetmp != node1 ){
                    node2 = nodetmp;
                }
            }
            if ( node0 == face(facetmp,0) && node1 == face(facetmp,2) ){
                node0 = face(facetmp,2);
                node1 = face(facetmp,0);
                node2 = face(facetmp,1);
            }
            // bisect the line node0-node1
            rowvec bisectNode = 1.0/2.0 * ( vertex.row(node0) + vertex.row(node1) );
            int nodenew; // it should already exist during the previous subdivisions process. 
            for (int j = vertex.n_rows - 1; j > vertexnum - 2; j-- ){
                rowvec distVec = vertex.row(j) - bisectNode;
                if ( norm(distVec,2) < 1.0e-9 ){ // nodenew is already in the vertex matrix
                    nodenew = j;
                    break;
                }
            }
            // build the faces
            Row<int> facenew(3); 
            facenew << nodenew << node2 << node0 << endr; // valenceIndex = [5 7 6]
            face.row(facetmp) = facenew;

            facenew << nodenew << node1 << node2 << endr; // valenceIndex = [5 6 7];
            face.resize(face.n_rows + 1, 3);
            face.row(face.n_rows - 1) = facenew;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // update faces_with_nodei, because both vertex and face matrices are changed!
    faces_with_nodei = faces_with_vertexi(vertex, face); 
    // update the boundary for vertice and faces. The newly added vertices and faces are not boundary or ghost! 
    isBoundaryNode.resize(1, vertex.n_rows); 
    for ( int i = vertexnum; i < vertex.n_rows; i++ ) isBoundaryNode(i) = -1;
    isBoundaryFace.resize(1, face.n_rows); 
    for ( int i = facenum; i < face.n_rows; i++ ) isBoundaryFace(i) = -1;
    isGhostNode.resize(1, vertex.n_rows); 
    for ( int i = vertexnum; i < vertex.n_rows; i++ ) isGhostNode(i) = -1;
    isGhostFace.resize(1, face.n_rows); 
    for ( int i = facenum; i < face.n_rows; i++ ) isGhostFace(i) = -1;
    
    ghostPartner.resize(vertex.n_rows, -1);
    freePartners.resize(vertex.n_rows, vector<int>(3, -1));
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // update localFinerMesh.faceIndex
    Row<int> addLocalFaceIndex;
    for ( int i = 0; i < selectLocalFaces.n_rows; i++ ){
        for ( int j = 0; j < selectLocalFaces.n_cols; j++ ){
            int facetmp = selectLocalFaces(i,j); 
            if ( facetmp == -1 ) continue;
            bool isExisted = false;
            for ( int k = 0; k < localFinerMesh.faceIndex.n_cols; k++ ){
                if ( facetmp ==  localFinerMesh.faceIndex(k) ) 
                    isExisted = true;
            }
            if ( isExisted == false ){
                addLocalFaceIndex.resize(1, addLocalFaceIndex.n_cols+1);
                addLocalFaceIndex(addLocalFaceIndex.n_cols-1) = facetmp;
            }
        }
    } 
    for ( int i = facenum; i < face.n_rows; i++ ){
        addLocalFaceIndex.resize(1, addLocalFaceIndex.n_cols+1);
        addLocalFaceIndex(addLocalFaceIndex.n_cols-1) = i;
    }
    for ( int i = 0; i < addLocalFaceIndex.n_cols; i++ ){
        localFinerMesh.faceIndex.resize(1, localFinerMesh.faceIndex.n_cols+1);
        localFinerMesh.faceIndex(localFinerMesh.faceIndex.n_cols-1) = addLocalFaceIndex(i);
    }
}

void build_local_finer_mesh(mat& vertex, Mat<int>& face, double& finestL, double radiusForFinest, Mat<int>& faces_with_nodei, LocalFinerMesh& localFinerMesh, Row<int>& isBoundaryNode, Row<int>& isBoundaryFace, Row<int>& isGhostNode, Row<int>& isGhostFace, vector<int>& ghostPartner, vector<vector<int>>& freePartners ){
    // the smallest edge of the mesh, about half the lipid edge. The target edge length around the insertion is 0.5 nm. 
    double targetEdge = 0.5; 
    if ( radiusForFinest < finestL ){
        cout<<"   No need for local-finer, because the selected area is too small: Mesh finestL = " << finestL << " nm, RadiusForLocalFiner = " << radiusForFinest << " nm." <<endl;
        return;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find the center vertex. 
    int centerVertexIndex = 0;
    // for flat membrane, the centerVertex is the center of the membrane
    bool isFlat = true;
    if ( isFlat == true ){
        double x0 = ( max(vertex.col(0)) + min(vertex.col(0)) ) / 2.0;
        double y0 = ( max(vertex.col(1)) + min(vertex.col(1)) ) / 2.0;
        double z0 = vertex(0,2);
        rowvec centerTarget(3); centerTarget << x0 << y0 << z0 << endr;
        double distTarget = 1.0e8;
        for ( int i = 0; i < vertex.n_rows; i++ ){
            rowvec distVec = vertex.row(i) - centerTarget;
            double disttmp = norm(distVec,2);
            if ( disttmp < distTarget ){
                distTarget = disttmp;
                centerVertexIndex = i;
            }
        }
    }
    
    // determine how many times of local-finer are needed.
    int times = round(log2(finestL/targetEdge)) ; 
    
    if ( times < 1 ){
        cout<<"   No need for local-finer, because the mesh is fine enough!" <<endl;
        return;
    }

    double radiusForFinestAdjusted = ceil( (radiusForFinest + 2.0/2.0) / finestL) * finestL;
    double radiusForLocalFiner = radiusForFinestAdjusted / pow(0.8, times-1);

    int localRounds = round( radiusForLocalFiner/finestL );

    if ( times >= 1){
        for ( int i = 0; i < times; i++ ){
            finer_local_mesh(centerVertexIndex, localRounds, vertex, face, faces_with_nodei, localFinerMesh, isBoundaryNode, isBoundaryFace, isGhostNode, isGhostFace, ghostPartner, freePartners);
            finestL = finestL/2.0; // after the finer, the smallest edge length
            int localRoundsAfterFiner = (localRounds-1) * 2 + 1; // after the finer, the local area has this number rounds of triangles 
            localRounds = localRoundsAfterFiner - 2;  // next times, this rounds of triangles will be finer.
            
            if ( i == times - 1 ){
                if ( localRounds > round( radiusForFinest/finestL ) )
                    localRounds = round( radiusForFinest/finestL );
            }
            
            if ( localRounds < 2 )
                break;

            // find the six vertices of the hexagon, and store them to the localFinerMesh.sixVerticeHexagon 
            if ( i == 0 ){
                localFinerMesh.buildLocalFinerMesh = true;
                localFinerMesh.centerVertexIndex = centerVertexIndex;
                Row<int> localFaces = localFinerMesh.faceIndex;
                double distancemax = 0.0;
                int node0 = 0;
                for ( int j = 0; j < localFaces.n_cols; j++ ){
                    for ( int k = 0; k < 3; k++ ){
                        int nodeindex = face(localFaces(j),k);
                        rowvec distvec = vertex.row(nodeindex) - vertex.row(centerVertexIndex);
                        if ( norm(distvec,2) > distancemax ){
                            distancemax = norm(distvec,2);
                            node0 = nodeindex;
                        }
                    }
                }
                localFinerMesh.sixVerticeHexagon.resize(1,6);
                localFinerMesh.sixVerticeHexagon(0) = node0;
                for ( int n = 1; n < 6; n++ ){
                    rowvec vec1 = vertex.row(centerVertexIndex); vec1 = vec1 / norm(vec1,2);
                    rowvec vec2 = vertex.row(node0) - vertex.row(centerVertexIndex); vec2 = vec2 / norm(vec2,2);
                    rowvec vec3 = cross(vec1, vec2);
                    double angle = n * 60.0/180.0 * M_PI;
                    rowvec vectmp = cos(angle) * vec2 + sin(angle) * vec3;
                    int nodetmp = 0;
                    distancemax = 0.0;
                    for ( int j = 0; j < localFaces.n_cols; j++ ){
                        for ( int k = 0; k < 3; k++ ){
                            int nodeindex = face(localFaces(j),k);
                            rowvec nodevec = vertex.row(nodeindex) - vertex.row(centerVertexIndex);
                            mat dist = nodevec * strans(vectmp);
                            if ( dist(0,0) > distancemax ){
                                distancemax = dist(0,0);
                                nodetmp = nodeindex;
                            }
                        }
                    }
                    localFinerMesh.sixVerticeHexagon(n) = nodetmp;
                }
            } // end of the six vertices of the hexagon, localFinerMesh.sixVerticeHexagon    
        } // end For loop of local finer mesh. 

        // Warning! for the membrane with boundaries, random_shuffle will cause chaos for the boundary updates!
        /////////////////////////////////////////////////////////////////////////////////////
        // rearragne the face index, because the new-added faces are all listed at the bottom, which has more irregular patches and will be assigned to the same node in openmpi, time-consuming.
        vector<int> faceIndex(face.n_rows); iota(faceIndex.begin(), faceIndex.end(), 0); 
        srand(500); // set the seed of the random. 
        std::random_shuffle(faceIndex.begin(), faceIndex.end());
        Mat<int> faceRearranged(face.n_rows,3); 
        for ( int i = 0; i < face.n_rows; i++ ){
            int faceindex = faceIndex[i];
            faceRearranged.row(i) =  face.row(faceindex);
        }
        face = faceRearranged;

        // update faces_with_nodei
        faces_with_nodei = faces_with_vertexi(vertex, face); 

        // update boundary face and ghost face
        Row<int> isBoundaryFaceRearranged(face.n_rows); // isBoundaryFaceRearranged.fill(-1);
        Row<int> isGhostFaceRearranged(face.n_rows);
        for ( int i = 0; i < face.n_rows; i++ ){
            int faceindex = faceIndex[i];
            isBoundaryFaceRearranged(i) = isBoundaryFace(faceindex);
            isGhostFaceRearranged(i)    = isGhostFace(faceindex);
        }
        isBoundaryFace = isBoundaryFaceRearranged;
        isGhostFace    = isGhostFaceRearranged;

        // update localFinerMesh.faceIndex (the local finer elements)
        for ( int i = 0; i < localFinerMesh.faceIndex.n_cols; i++ ){
            int faceindex_previous = localFinerMesh.faceIndex(i);
            auto it = find(faceIndex.begin(), faceIndex.end(), faceindex_previous);
            if ( it == faceIndex.end() ){
                cout<<"Wrong! The local finer face is not found after the faces are rearranged in index."<<endl;
                exit(0);
            }else{
                int faceindex_present = it - faceIndex.begin();
                localFinerMesh.faceIndex(i) = faceindex_present;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for selection_insertionPatch

// find faceindex's mirror face, symetry along the line node1-node2
int find_face(int node1, int node2, int faceindex, Mat<int> vertexi_face){
    int facetarget = -1;
    for (int i = 0; i < 6; i ++){
        int facei = vertexi_face(node1,i);
        if ( facei == -1 ) continue;
        for (int j = 0; j < 6; j++){
            int facej = vertexi_face(node2,j);
            if ( facej == -1 ) continue;
            if ( facei == facej && facei != faceindex){
                facetarget = facei;
            }
        }
    }
    if (facetarget == -1 ){
        cout<<"Wrong: in find_two_faces, unsuccessful! "<<endl;
        exit(0);
    }
    return facetarget;
}

struct InsertionTemplate{
    rowvec center;
    rowvec direction;
};

// the templates include the insetionPatches' centers and directions
// templates are all located on x-y plane, with their geometry center being [0,0,0].
// the flat membrane: x is direction
vector<InsertionTemplate> build_insertion_templates(int insertionPatchNum, double distBetwn, bool isInsertionsParallel){
    vector<InsertionTemplate> templates(insertionPatchNum);
    if ( insertionPatchNum == 1 ){
        templates[0].center << 0.0 << 0.0 << 0.0 << endr;
        templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
    }else if ( insertionPatchNum == 2 ){
        templates[0].center << 0.0 << -0.5*distBetwn << 0.0 << endr; templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
        templates[1].center << 0.0 << +0.5*distBetwn << 0.0 << endr; templates[1].direction << 1.0 << 0.0 << 0.0 << endr;
    }else if ( insertionPatchNum == 3 ){
        if ( isInsertionsParallel == true ){
            templates[0].center << 0.0 << -distBetwn << 0.0 << endr; templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[1].center << 0.0 << 0.0 << 0.0 << endr;        templates[1].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[2].center << 0.0 << +distBetwn << 0.0 << endr; templates[2].direction << 1.0 << 0.0 << 0.0 << endr;
        }else{
            double rho = distBetwn / sqrt(3.0);
            double angle0 = M_PI/2.0; 
            double angle1 = 7.0/6.0*M_PI; 
            double angle2 = -1.0/6.0*M_PI;
            templates[0].center << rho*cos(angle0) << rho*sin(angle0) << 0.0 << endr; templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[1].center << rho*cos(angle1) << rho*sin(angle1) << 0.0 << endr; templates[1].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[2].center << rho*cos(angle2) << rho*sin(angle2) << 0.0 << endr; templates[2].direction << 1.0 << 0.0 << 0.0 << endr;
        }
    }else if ( insertionPatchNum == 4 ){
        if ( isInsertionsParallel == true ){
            templates[0].center << 0.0 << -1.5*distBetwn << 0.0 << endr; templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[1].center << 0.0 << -0.5*distBetwn << 0.0 << endr; templates[1].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[2].center << 0.0 << +0.5*distBetwn << 0.0 << endr; templates[2].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[3].center << 0.0 << +1.5*distBetwn << 0.0 << endr; templates[3].direction << 1.0 << 0.0 << 0.0 << endr;
        }else{
            double rho = distBetwn / sqrt(2.0);
            double angle0 = M_PI/4.0; 
            double angle1 = 3.0/4.0*M_PI; 
            double angle2 = 5.0/4.0*M_PI; 
            double angle3 = 7.0/4.0*M_PI;
            templates[0].center << rho*cos(angle0) << rho*sin(angle0) << 0.0 << endr; templates[0].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[1].center << rho*cos(angle1) << rho*sin(angle1) << 0.0 << endr; templates[1].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[2].center << rho*cos(angle2) << rho*sin(angle2) << 0.0 << endr; templates[2].direction << 1.0 << 0.0 << 0.0 << endr;
            templates[3].center << rho*cos(angle3) << rho*sin(angle3) << 0.0 << endr; templates[3].direction << 1.0 << 0.0 << 0.0 << endr;
        }
    }
    return templates;
}

vector<InsertionTemplate> generate_insertionCentersDirections_accordingTemplates(bool isFlatMembrane, vector<InsertionTemplate> templates, mat vertex, LocalFinerMesh localFinerMesh){
    vector<InsertionTemplate> insertions = templates;

    rowvec geometryCenterTemplates(3); geometryCenterTemplates << 0.0 << 0.0 << 0.0 << endr;
    rowvec geometryDirectionTemplates(3); geometryDirectionTemplates << 0.0 << 0.0 << 1.0 << endr;

    rowvec geometryCenterTarget = vertex.row(localFinerMesh.centerVertexIndex);

    rowvec geometryDirectionTarget = geometryCenterTarget / norm(geometryCenterTarget, 2);
    if (isFlatMembrane == true){ // specially for flat membrane.
        geometryDirectionTarget << 0.0 << 0.0 << 1.0 << endr;
    }
    
    // rotate templates from its direction to the target direction.
    // calculate the rotation angles along x, y, z axis: alpha, peta, gama. Actually, gama = 0. It means, only need to rotate along x and y axis.
    // rotation angle along x: alpha
    rowvec n1 = geometryDirectionTemplates;
    rowvec nt = geometryDirectionTarget;
    double alpha1 = 0.5 * M_PI;
    double alpha2 = acos( nt(1) / sqrt( nt(1)*nt(1) + nt(2)*nt(2) ) );
    if ( nt(2) < 0.0 ) alpha2 = 2.0*M_PI - alpha2;
    double alpha = alpha2 - alpha1;
    // rotation angle along y: peta
    double peta1 = 0.5 * M_PI;
    double peta2 = acos( nt(0) / sqrt( nt(0)*nt(0) + nt(2)*nt(2) ) );
    if ( nt(2) < 0.0 ) peta2 = 2.0*M_PI - peta2;
    double peta = peta2 - peta1;
    // rotation angle along z: gama
    double gama = 0.0;
    
    // rotate now
    mat Rx(3,3); Rx << 1.0 << 0.0 << 0.0 << endr
                    << 0.0 << cos(alpha) << -sin(alpha) << endr
                    << 0.0 << sin(alpha) << cos(alpha) << endr;
    mat Ry(3,3); Ry << cos(peta) << 0.0 << sin(peta) << endr
                    << 0.0 << 1.0 << 0.0 << endr
                    << -sin(peta) << 0.0 << cos(peta) << endr;

    for ( int i = 0; i < templates.size(); i++ ){
        // rotate the centers and then move to the target position
        mat tmp = Rx * strans(templates[i].center);
        tmp = Ry * tmp;
        insertions[i].center = strans(tmp.col(0));
        insertions[i].center = insertions[i].center + geometryCenterTarget;
        // rotate the directions
        tmp = Rx * strans(templates[i].direction);
        tmp = Ry * tmp;
        insertions[i].direction = strans(tmp.col(0));
    }

    // on sphere, the surface is curved. Adjust the centers to be on the surface
    //double radius = norm(vertex.row(0));
    //for (int i = 0; i < insertions.size(); i++){
    //}

    return insertions;
}

Row<int> find_oneInsertion_accordingToCenterAndDirection(bool isFlatMembrane, Mat<int> face, mat vertex, Mat<int> faces_with_nodei, LocalFinerMesh localFinerMesh, double meanL, double insertLength, double insertWidth, InsertionTemplate targetCenterDirection){
    double triangleS = sqrt(3.0)/4.0 * meanL * meanL;
    int n2 = round(insertWidth/(sqrt(3.0)/2.0*meanL));
    int n1 = round( insertLength * insertWidth / triangleS / n2 );
    // for each insertionpatch, we need n1 *n2 triangles
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // build insertionpatch matrix to store the face indexes as the insertionpatch
    Row<int> insertionpatch(n1 * n2); insertionpatch.fill(-1);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // to find the first triangle, set it as the 0
    rowvec direction = targetCenterDirection.direction; 
    rowvec center0 = targetCenterDirection.center - direction*(insertLength/2.0); 

    // find the face0 which is closest to the center0, as the first face of the insertion
    int face0; 
    if ( localFinerMesh.buildLocalFinerMesh == false ){
        double shu = 1.0e8;
        for ( int i = 0; i < face.n_rows; i++ ){
            int facetmp = i;
            int node0 = face(facetmp,0);
            int node1 = face(facetmp,1);
            int node2 = face(facetmp,2);
            rowvec centertmp = 1.0/3.0 * ( vertex.row(node0) + vertex.row(node1) + vertex.row(node2) );
            double disttmp = norm(centertmp - center0, 2); 
            if ( disttmp < shu ){
                shu = disttmp;
                face0 = facetmp;
            }
        }
    }else{
        double shu = 1.0e8;
        for ( int i = 0; i < localFinerMesh.faceIndex.n_cols; i++ ){ 
            int facetmp = localFinerMesh.faceIndex(i);
            int node0 = face(facetmp,0);
            int node1 = face(facetmp,1);
            int node2 = face(facetmp,2);
            rowvec centertmp = 1.0/3.0 * ( vertex.row(node0) + vertex.row(node1) + vertex.row(node2) );
            double disttmp = norm(centertmp - center0, 2); 
            if ( disttmp < shu ){
                shu = disttmp;
                face0 = facetmp;
            }
        }
    }
    insertionpatch(0) = face0;

    // define the direction1 and direction2
    rowvec direction1(3); 
    {
        double shu = 0.0;
        int node0 = -1; int node1 = -1;
        for ( int i = 0; i < 3; i++ ){
            int nodei = face(face0,i);
            int nodei1;
            if ( i+1 > 2){ 
                nodei1 = face(face0,0);
            }else{
                nodei1 = face(face0,i+1);
            }
            rowvec directtmp = vertex.row(nodei1) - vertex.row(nodei); directtmp = directtmp / norm(directtmp,2);
            mat shutmp = directtmp * strans(direction);
            if ( shutmp(0,0) > shu ){
                shu = shutmp(0,0);
                node0 = nodei;
                node1 = nodei1;
            }
            directtmp = - directtmp;
            shutmp = directtmp * strans(direction);
            if ( shutmp(0,0) > shu ){
                shu = shutmp(0,0);
                node0 = nodei1;
                node1 = nodei;
            }
        }
        direction1 = vertex.row(node1) - vertex.row(node0); 
        direction1 = direction1 / norm(direction1,2); 
    }
    rowvec direction2(3);
    {
        int node0 = face(face0,0);
        int node1 = face(face0,1);
        int node2 = face(face0,2);
        rowvec centertmp = 1.0/3.0 * ( vertex.row(node0) + vertex.row(node1) + vertex.row(node2) );
        if ( isFlatMembrane == true ){
            centertmp << 0.0 << 0.0 << 1.0 << endr;
        }
        direction2 = cross(centertmp/norm(centertmp,2), direction1);
        // make sure direction2 is not pointing towards the face0's center.
        centertmp = 1.0/3.0 * ( vertex.row(node0) + vertex.row(node1) + vertex.row(node2) );
        rowvec v0 = vertex.row(node0) - centertmp;
        rowvec v1 = vertex.row(node1) - centertmp;
        rowvec v2 = vertex.row(node2) - centertmp;
        mat tmp = ( direction2 * strans(v0) ) * ( direction2 * strans(v1) ) * ( direction2 * strans(v2) );
        if ( tmp(0,0) > 0 )
            direction2 = -direction2;

        direction2 = direction2 / norm(direction2,2); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // row0 triangles along n1
    for ( int i = 1; i < n1; i++ ){
        int facepre = insertionpatch(i-1); // prior face index
        rowvec centerpre = 1.0/3.0*( vertex.row(face(facepre,0)) + vertex.row(face(facepre,1)) + vertex.row(face(facepre,2)) );
        int facetarget = -1;
        double shu = 0.0;
        for ( int j = 0; j < 3; j++ ){
            int nodei = face(facepre,j);
            int nodei1;
            if ( j+1 > 2){ 
                nodei1 = face(facepre,0);
            }else{
                nodei1 = face(facepre,j+1);
            }
            int facetmp = find_face(nodei, nodei1, facepre, faces_with_nodei); // facepre's mirror face, symetry of the line nodei-nodei1
            rowvec centertmp = 1.0/3.0*( vertex.row(face(facetmp,0)) + vertex.row(face(facetmp,1)) + vertex.row(face(facetmp,2)) );
            rowvec directtmp = centertmp - centerpre; directtmp = directtmp / norm(directtmp,2);
            mat shutmp = directtmp * strans(direction1);
            if (  shutmp(0,0) > shu ){
                shu = shutmp(0,0);
                facetarget = facetmp;
            }
        }
        insertionpatch(i) = facetarget;
    }

    //////////////////////////////////////////////////
    // row0 patches along n2
    for ( int i = 1; i < n2; i++ ){
        for ( int j = 0; j < n1; j++ ){
            int indexnow = n1*i + j;
            int indexpre = n1*(i-1) + j;
            int faceref = insertionpatch(indexpre);
            rowvec centerref = 1.0/3.0*( vertex.row(face(faceref,0)) + vertex.row(face(faceref,1)) + vertex.row(face(faceref,2)) );
            int facetarget = -1;
            double shu = 0.0;
            for ( int k = 0; k < 3; k++ ){
                int nodek = face(faceref,k);
                for ( int l = 0; l < faces_with_nodei.n_cols; l++ ){
                    int facetmp = faces_with_nodei(nodek,l);
                    if ( facetmp == -1 || facetmp == faceref ) continue;
                    rowvec centertmp = 1.0/3.0*( vertex.row(face(facetmp,0)) + vertex.row(face(facetmp,1)) + vertex.row(face(facetmp,2)) );
                    rowvec directtmp = centertmp - centerref; directtmp = directtmp / norm(directtmp,2);
                    mat shutmp = directtmp * strans(direction2);
                    if (  shutmp(0,0) > shu ){
                        shu = shutmp(0,0);
                        facetarget = facetmp;
                    }
                }
            }
            insertionpatch(indexnow) = facetarget;
        }
    } 

    for ( int i = 0; i < insertionpatch.n_cols; i++ ){
        if (insertionpatch(i) == -1){
            cout << "Wrong: the insertionpatch is not found correctly! Happens in: find_oneInsertion_accordingToCenterAndDirection. " << endl;
            exit(0);
        }
    }

    return insertionpatch;
}

Mat<int> select_insertionPatch(bool isFlatMembrane, Mat<int> face, mat vertex, Mat<int> faces_with_nodei, LocalFinerMesh localFinerMesh, double meanL, double insertLength, double insertWidth, int insertionPatchNum, double distBetwn, bool isInsertionsParallel){
    // build insertionpatch matrix to store the face indexes as the insertionpatch
    Mat<int> insertionpatch;
    if ( insertionPatchNum == 0 ){
        return insertionpatch;
    }
    vector<InsertionTemplate> templates = build_insertion_templates(insertionPatchNum, distBetwn, isInsertionsParallel);
    vector<InsertionTemplate> insertionsCenterDirection = generate_insertionCentersDirections_accordingTemplates(isFlatMembrane, templates, vertex, localFinerMesh); 

    for ( int i = 0; i < insertionPatchNum; i++ ){
        Row<int> oneInsertion = find_oneInsertion_accordingToCenterAndDirection(isFlatMembrane, face, vertex, faces_with_nodei, localFinerMesh, meanL, insertLength, insertWidth, insertionsCenterDirection[i]);
        if ( i > 0 ){
            if ( insertionpatch.n_cols != oneInsertion.n_cols ){
                cout << "Wrong: insertions are not the same size! "<< endl;
                exit(0);
            }
        }
        insertionpatch.resize(insertionpatch.n_rows + 1, oneInsertion.n_cols);
        insertionpatch.row(i) = oneInsertion;
    }
    
    return insertionpatch;
}

Row<int> select_insertionPatchAdjacent(Mat<int> faceout, mat vertexout, Mat<int> faces_with_nodei, Mat<int> insertionpatch){
    Row<int> adjacent(faceout.n_rows);
    adjacent.fill(-1);
    int jishu = -1;
    for (int i = 0; i < insertionpatch.n_rows; i++){
        for (int j = 0; j < insertionpatch.n_cols; j++){
            int faceindex = insertionpatch(i,j);
            for (int k = 0; k < 3; k++){
                int nodeindex = faceout(faceindex,k);
                for ( int l = 0; l < faces_with_nodei.n_cols; l++ ){
                    int facetmp = faces_with_nodei(nodeindex,l);
                    if ( facetmp == -1 ) continue;
                    bool isfacetmpCounted = false;
                    for (int ii = 0; ii < insertionpatch.n_rows; ii++){
                        for (int jj = 0; jj < insertionpatch.n_cols; jj++){
                            if ( facetmp == insertionpatch(ii,jj)){
                                isfacetmpCounted = true;
                            }
                        }
                    }
                    for (int m = 0; m < jishu+1; m++ ){
                        if ( facetmp == adjacent(m) ){
                            isfacetmpCounted = true;
                        }
                    }
                    if ( isfacetmpCounted == false && facetmp != -1){
                        jishu = jishu + 1;
                        adjacent(jishu) = facetmp;
                    }
                }
            }
        }
    }
    int jishunew = jishu;
    
    // the second round
    for (int i = 0; i < jishu+1; i++){
        int faceindex = adjacent(i);
        if ( faceindex == -1 ) continue;
        for (int k = 0; k < 3; k++){
                int nodeindex = faceout(faceindex,k);
                for ( int l = 0; l < faces_with_nodei.n_cols; l++ ){
                    int facetmp = faces_with_nodei(nodeindex,l);
                    if ( facetmp == -1 ) continue;
                    bool isfacetmpCounted = false;
                    for (int ii = 0; ii < insertionpatch.n_rows; ii++){
                        for (int jj = 0; jj < insertionpatch.n_cols; jj++){
                            if ( facetmp == insertionpatch(ii,jj)){
                                isfacetmpCounted = true;
                            }
                        }
                    }
                    for (int m = 0; m < jishunew+1; m++ ){
                        if ( facetmp == adjacent(m) ){
                            isfacetmpCounted = true;
                        }
                    }
                    if ( isfacetmpCounted == false && facetmp != -1){
                        jishunew = jishunew + 1;
                        adjacent(jishunew) = facetmp;
                    }
                }
        }
    }

    Row<int> out(jishunew+1);
    for(int i = 0; i < jishunew+1; i++){
        out(i) = adjacent(i);
    }

    return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for flip-flop ratio
// R is the radius of the sphere. 
// h is the monolayer thickness, c0 is the monolayer spontaneous curvature, kc is the monolayer bending modulus. us is the monolayer area modulus.
// 1 is the inner layer, and 2 is the outer layer
double calculate_flipflopRatio(bool isFlatMembrane, double S0mid, double h1, double c01, double kc1, double us1, double h2, double c02, double kc2, double us2){
    if (isFlatMembrane == true){
        return 0.0;
    }
    
    double R = sqrt(S0mid/4.0/M_PI);
    double gama1 = 0.0;
    double gama2 = 0.99;
    double gamatmp; // = 2.0 * R * h /(pow(R,2.0) + pow(h,2.0));
    double rightHand = kc1 * pow(2.0/(R-h1)-c01 ,2.0) + kc2 * pow(2.0/(R+h2)-c02 ,2.0); 
    double leftHand1 = us1 * pow((R-h1)/R,4.0)/pow(1.0-gama1,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gama1,2.0) - (us1-us2); 
    double leftHand2 = us1 * pow((R-h1)/R,4.0)/pow(1.0-gama2,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gama2,2.0) - (us1-us2); 
    if ( (leftHand1-rightHand)*(leftHand2-rightHand) < 0.0 ){
        gama1 = gama1; 
        gama2 = gama2;
    }else{
        cout<<"Warning: multiple values of flip-flop ratio exist in [0,1]."<<endl;
        double gamatmp = 0.5 * (gama1 + gama2);
        double leftHandtmp = us1 * pow((R-h1)/R,4.0)/pow(1.0-gamatmp,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gamatmp,2.0) - (us1-us2); 
        while ( (leftHand1-rightHand)*(leftHandtmp-rightHand) > 0.0 ){
            gamatmp = gama1 + 0.9 * (gamatmp - gama1);
            leftHandtmp = us1 * pow((R-h1)/R,4.0)/pow(1.0-gamatmp,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gamatmp,2.0) - (us1-us2); 
            if ( abs(gamatmp - gama1) < 1.0e-2 ){
                cout<<"No acurate flip-flop ratio is found! But to use the approximate one!"<<endl;
                double h = (h1 + h2) / 2.0;
                return 2.0 * R * h /(pow(R,2.0) + pow(h,2.0));
            }
        }
        gama1 = gama1;
        gama2 = gamatmp;
    }
    
    double criterion = 1.0e-2;
    while ( abs(gama2 - gama1) > criterion ){
        leftHand1 = us1 * pow((R-h1)/R,4.0)/pow(1.0-gama1,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gama1,2.0) - (us1-us2); 
        leftHand2 = us1 * pow((R-h1)/R,4.0)/pow(1.0-gama2,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gama2,2.0) - (us1-us2); 
        double gamatmp = 0.5 * (gama1 + gama2);
        double leftHandtmp = us1 * pow((R-h1)/R,4.0)/pow(1.0-gamatmp,2.0) - us2 * pow((R+h2)/R,4.0)/pow(1.0+gamatmp,2.0) - (us1-us2); 
        if ( (leftHand1-rightHand)*(leftHand2-rightHand) > 0.0 ){
            cout<<"Wrong to find the flip-flop ratio!"<<endl;
            exit(0);
        }else{
            if ( (leftHand1-rightHand)*(leftHandtmp-rightHand) > 0.0 ){
                gama1 = gamatmp; 
                gama2 = gama2;
            }else if ( (leftHand2-rightHand)*(leftHandtmp-rightHand) > 0.0 ){
                gama1 = gama1; 
                gama2 = gamatmp;
            }
        }
     }
     double gama = 0.5 * (gama1 + gama2);
     return gama;
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
// h01: thickness_in; h02, thickness_out
void determine_Target_Height_for_Each_Monolayer(double h01, double h02, mat vertex, double& H0C1, double& H0C2){
    // out-layer
    double Cout = 0.0; // mean curvature of the out-layer, Flat membrane
    double C = Cout;
    double h0 = h02;
    H0C2 = h0 * ( 1.0 + 1.0*C*h0 ) + 2.0/3.0 * pow(h0,3.0) * pow(C,2.0); 

    // in-layer
    double Cin = 0.0; // mean curvature of the in-layer, Flat membrane
    C = -Cin;  // note the sign of the curvature of the inlayer is opposite to the one of the outlayer
    h0 = h01;
    H0C1 = h0 * ( 1.0 + 1.0*C*h0 ) + 2.0/3.0 * pow(h0,3.0) * pow(C,2.0); 
}

