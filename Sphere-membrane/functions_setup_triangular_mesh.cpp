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

struct Param{
    // mesh parameters:
    double meanL;                                   // triangular side length, nm
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
    bool   usingNCG = true;
    bool   isNCGstucked = false;
    bool   usingRpi = true;
    double k_regularization;                         // coefficient of the regulerization constraint, 
    double gama_shape = 0.2;                         // factor of shape deformation
    double gama_area = 0.2;                          // factor of size deformation
    int    subDivideTimes = 5;                       // subdivision times for the irregular patches
    int    GaussQuadratureN = 2;                     // Gaussian quadrature integral 

    // membrane parameters: 
    bool   isBilayerModel = true;
    bool   isGlobalAreaConstraint = true;            // whether to use Global constraints for the area elasticity
    // out-layer (top layer)
    double kc_out  = 20.0*4.17/2.0;                  // pN.nm. bending modulus, out-monolayer 
    double us_out  = 250.0/2.0;                      // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
    double Ktilt_out  = 90.0;                        // pN/nm = mN/m. tilt modulus
    double Kthick_out = Ktilt_out + 2.0*us_out;      // pN/nm, coefficient of the membrane thickness. penalty term
    double thickness_out = 4.0/2.0;                  // nm, out-monolayer thickness. 
    double c0out = 0.0;                              // spontaneous curvature of membrane, out-layer. Convex is positive
    double S0out;                                    //target area
    double Sout;                                     // area, total area 
    double V0out;                                    // target volume of the sphere
    double Vout;                                     // volume of the sphere      
    
    // in-layer (bottom layer)
    double kc_in  = kc_out;                          // pN.nm. bending modulus, in-monolayer 
    double us_in  = us_out;                          // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
    double Ktilt_in  = Ktilt_out;                    // pN/nm = mN/m. tilt modulus
    double Kthick_in = Ktilt_in + 2.0*us_in;
    double thickness_in = thickness_out;             // nm, in-monolayer thickness. 
    double c0in  = c0out;                            // spontaneous curvature of membrane, in-layer. Concave is positive
    double S0in;                                     // target area
    double Sin;                                      // area, total area 
    double V0in;                                     // target volume of the sphere
    double Vin;                                      // volume of the sphere      
    double uv;                                       // coefficeint of the volume constraint, 0.5*uv*(dv)^2/v0;

    // insertion parameters:
    Mat<int> insertionpatch;
    double c0out_ins = 0.3;                          // spontaneous curvature of insertion, outer layer
    double c0in_ins = 0.3;
    double s_insert  = 2.0;                          // insertion area
    double insert_dH0 = 0.3;                         // equilibrium value of thickness decrease induced by the insertion, nm
    double K_insertShape   = 10.0*us_out;            // spring constant for insertion zones, to constraint the insertion shape
    rowvec insertionShapeEdgeLength; 
    Row<int> IsinertionpatchAdjacent;
    double K_adjacentPatch = 0.0;                    // constant for the shape constraint on the patches adjacent around insertion 

    // system setup
    bool   duringStepsToIncreaseInsertDepth = false;
    vector<bool> isLocallyFinerFace;
};  

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for neighor vertex or faces

// find the faces around one vertex
// each vertex have 6 or 5 or 7 neighbor faces that have this vertex. 
// For the regular vertex, it has 6 neighbors; for the irregular one, it has 5  or 7 neighbors;
Mat<int> faces_with_vertexi(mat vertex, Mat<int> face){
    int vertexnum = vertex.n_rows;
    Mat<int> face_with_nodei(vertexnum,7); face_with_nodei.fill(-1);

    #pragma omp parallel for 
     for (int nodeIndex = 0; nodeIndex < vertexnum; nodeIndex++){
        int facenumber = -1;
        for (int faceIndex = 0; faceIndex < face.n_rows; faceIndex++){
            for (int k = 0; k < 3; k++){
                if ( nodeIndex == face(faceIndex,k) ){
                    facenumber = facenumber + 1;
                    face_with_nodei(nodeIndex,facenumber) = faceIndex;
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

/*

Row<int> find_two_faces(int node1, int node2, Mat<int> vertexi_face){
    Row<int> twofaces(2); twofaces.fill(-1);
    int number = 0;
    for (int i = 0; i < 6; i ++){
        if ( vertexi_face(node1,i) == -1 ) continue;
        for (int j = 0; j < 6; j++){
            if ( vertexi_face(node2,j) == -1 ) continue;
            if ( vertexi_face(node1,i) == vertexi_face(node2,j) ){
                twofaces(number) = vertexi_face(node1,i);
                number++;
            }
        }
    }
    if (twofaces(0) == -1 || twofaces(1) == -1){
        cout<<"Wrong: in find_two_faces, unsuccessful! "<<endl;
        exit(0);
    }
    return twofaces;
}
*/


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

Mat<int> one_ring_vertices(Mat<int> face,mat vertex, Mat<int> closest_nodes){   
    int facenum = face.n_rows;
    Mat<int> ring_vertices(facenum,13); ring_vertices.fill(-1);
    // three types of patch: 1. regular patch with 12 one-ring vertices, each vertex has 6 closest nodes. valence vector = [6,6,6];
    //                       2. irregular patch with 11 one-ring vertices, one vertex has 5 closest-nodes. valence vector = [5,6,6];
    //                       3. complex patch with 13 one-ring vertices, one vertex has 7 closest-nodes. valence vector = [6,6,7];
    //                       4. sudo-regular patch with 12 one-ring vertice,but one vertex has 7 closest-nodes, and one vertex has 5 closest-nodes. valence vector = [5,7,6];
    //                       5. sudo-regular patch. chiral symetry to case 4. valence vector = [5,6,7];
    // the one-ring-vertex last element shows out the case index as -1, -2, >0, -4, -5. !!!!!!
    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        //int d1, d2, d3, d5, d6, d9, d10, d11, d12, d13;
        Mat<int> neighbornum = determine_valence_vector(face, closest_nodes, i); 

        // case 1: regular patch
        if ( neighbornum(0,1) == 6 && neighbornum(1,1) == 6 && neighbornum(2,1) == 6 ){
            // note the order of the vertex
            int d4 = face(i,0); 
            int d7 = face(i,1); 
            int d8 = face(i,2);

            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center = 1.0/3.0 * (node4 + node7 + node8);
            mat shu = cross(node7-node4,node8-node4)*strans(center);
            if (shu(0,0) < 0){
                d7 = face(i,2);
                d8 = face(i,1);
            }

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

            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center = 1.0/3.0 * (node4 + node7 + node8);
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

            // make sure that node4-node7-node8 is counterclockwise! this is super important because it determines the surface normal direction is pointing to outside the sphere 
            rowvec node4 = vertex.row(d4); rowvec node7 = vertex.row(d7); rowvec node8 = vertex.row(d8);
            rowvec center = 1.0/3.0 * (node4 + node7 + node8);
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
/*
Mat<int> one_ring_vertices_in(Mat<int> facein,mat vertexin, mat vertexout, Mat<int> closest_nodes_in, Mat<int> ring_vertices_out){   
    int facenum = facein.n_rows;
    Mat<int> ring_vertices_in(facenum,13); ring_vertices_in.fill(-1);
    for ( int i = 0; i < facenum; i++){
        int d4out; int d7out; int d8out;
        if (ring_vertices_out(i,12) == -1){
            d4out = ring_vertices_out(i,3);
            d7out = ring_vertices_out(i,6);
            d8out = ring_vertices_out(i,7);         
        }else if (ring_vertices_out(i,12) == -2){
            d4out = ring_vertices_out(i,2);
            d7out = ring_vertices_out(i,5);
            d8out = ring_vertices_out(i,6);
        }else if (ring_vertices_out(i,12) >= 0){
            d4out = ring_vertices_out(i,3);
            d7out = ring_vertices_out(i,6);
            d8out = ring_vertices_out(i,7); 
        }else if (ring_vertices_out(i,12) == -3){
            d4out = ring_vertices_out(i,2);
            d7out = ring_vertices_out(i,5);
            d8out = ring_vertices_out(i,6);
        }
        int d4, d7, d8;
        rowvec center = 1.0/3.0*( vertexin.row(facein(i,0)) + vertexin.row(facein(i,1)) + vertexin.row(facein(i,2)) );
        if ( d4out < vertexin.n_rows ){
            d4 = d4out;
            mat shu;
            if (facein(i,0) == d4){ 
                d7 = facein(i,1);
                d8 = facein(i,2);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,1) == d4){
                d7 = facein(i,2);
                d8 = facein(i,0);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,2) == d4){
                d7 = facein(i,0);
                d8 = facein(i,1);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }
            if ( shu(0,0) < 0.0 ){
                int tmp = d7;
                d7 = d8;
                d8 = tmp;
            }
        }else if (d7out < vertexin.n_rows){
            d7 = d7out;
            mat shu;
            if (facein(i,0) == d7){ 
                d8 = facein(i,1);
                d4 = facein(i,2);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,1) == d7){
                d8 = facein(i,2);
                d4 = facein(i,0);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,2) == d7){
                d8 = facein(i,0);
                d4 = facein(i,1);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }
            if ( shu(0,0) < 0.0 ){
                int tmp = d4;
                d4 = d8;
                d8 = tmp;
            }
        }else if (d8out < vertexin.n_rows){
            d8 = d8out;
            mat shu;
            if (facein(i,0) == d8){ 
                d4 = facein(i,1);
                d7 = facein(i,2);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,1) == d8){
                d4 = facein(i,2);
                d7 = facein(i,0);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }else if (facein(i,2) == d8){
                d4 = facein(i,0);
                d7 = facein(i,1);
                rowvec vec1 = vertexin.row(d7)-vertexin.row(d4);
                rowvec vec2 = vertexin.row(d8)-vertexin.row(d4);
                shu = cross(vec1,vec2)*strans(center);
            }
            if ( shu(0,0) < 0.0 ){
                int tmp = d4;
                d4 = d7;
                d7 = tmp;
            }
        }
        int d4neighbornum = validatednum(closest_nodes_in.row(d4));
        int d7neighbornum = validatednum(closest_nodes_in.row(d7));
        int d8neighbornum = validatednum(closest_nodes_in.row(d8)); 
        if ( d4neighbornum <= d7neighbornum && d7neighbornum <= d8neighbornum ){
            int d3 = find_nodeindex(d4, d7, d8, closest_nodes_in);
            int d11 = find_nodeindex(d7, d8, d4, closest_nodes_in);
            int d5 = find_nodeindex(d4, d8, d7, closest_nodes_in);
            int d1 = find_nodeindex(d3, d4, d7, closest_nodes_in);
            int d2 = find_nodeindex(d4, d5, d8, closest_nodes_in);
            int d6 = find_nodeindex(d3, d7, d4, closest_nodes_in);
            int d9 = find_nodeindex(d8, d5, d4, closest_nodes_in);
            int d10 = find_nodeindex(d7, d11, d8, closest_nodes_in);
            int d12 = find_nodeindex(d8, d11, d7, closest_nodes_in);
            Row<int> v(13);
            if ( d4neighbornum == 6 ){
                int d13 = -1;
                v << d1 << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << endr;
                ring_vertices_in.row(i) = v;
            }else if ( d4neighbornum == 5 ){
                int d13 = -2;
                v << d2 << d3 << d4 << d5 << d6 << d7 << d8 << d9 << d10 << d11 << d12 << d13 << d13 << endr;
                ring_vertices_in.row(i) = v;
            }
        }else{
            cout<<"Wrong! The order of one-ring-vertices of in-layer is nor in correct!"<<endl;
            exit(0);
        }
    }
    return ring_vertices_in;
}
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for loop's subdivison scheme to build a spherical mesh from an icosahedron

void seticosahedron(mat& vertex, Mat<int>& face, double r){
    // build the vertex
    rowvec t(5);
    for (int i=0; i<5; i++){ t(i) = i * (2.0*M_PI/5.0); }
    mat vertex1(5,3);  // vertex of up pentagon
    for (int i=0; i<5; i++){ 
        rowvec v;
        v << cos(t(i)) << sin(t(i)) << 0.0;
        vertex1.row(i) = v;
    }
    t = t + M_PI/5.0;
    double a = 2.0*sin(M_PI/5.0);   // side length
    mat vertex2(5,3);    // vertex of down pentagon
    for (int i=0; i<5; i++){ 
        rowvec v;
        v << cos(t(i)) << sin(t(i)) << -a*sqrt(3.0)/2.0;
        vertex2.row(i) = v;
    }
    double h = sqrt(a*a-1.0);          // distance between upest point and pentagon
    rowvec vertex3; vertex3<<0.0<<0.0<<h; // upest vertex
    rowvec vertex4; vertex4<<0.0<<0.0<<-a*sqrt(3.0)/2.0-h; // downest vertex
    //mat vertex(12,3);
    vertex.row(0) = vertex3;
    vertex.rows(1,5) = vertex1;
    vertex.rows(6,10) = vertex2;
    vertex.row(11) = vertex4;
    vertex.col(2) += a*sqrt(3.0)/4.0; // move the center to the origin
    for (int i=0; i<12; i++) {
        vertex.row(i) = vertex.row(i)/norm(vertex.row(i),2)*r;  // move the vertex to the surface of the sphere 
    }
    // build the face
    face<< 0 << 1 << 2 << endr
        << 0 << 2 << 3 << endr
        << 0 << 3 << 4 << endr
        << 0 << 4 << 5 << endr
        << 0 << 5 << 1 << endr
        << 1 << 6 << 2 << endr
        << 2 << 7 << 3 << endr
        << 3 << 8 << 4 << endr
        << 4 << 9 << 5 << endr
        << 5 << 10 << 1 << endr
        << 2 << 6 << 7 << endr
        << 3 << 7 << 8 << endr
        << 4 << 8 << 9 << endr
        << 5 << 9 << 10 << endr
        << 1 << 10 << 6 << endr
        << 11 << 7 << 6 << endr
        << 11 << 8 << 7 << endr
        << 11 << 9 << 8 << endr
        << 11 << 10 << 9 << endr
        << 11 << 6 << 10 << endr;
} 

// find the two mirror dots, symetry line node1-node2
Row<int> find_two_mirror_nodes(int node1, int node2, Mat<int> faces_with_nodei, Mat<int> face){
    Row<int> aface = faces_with_nodei.row(node1);
    Row<int> bface = faces_with_nodei.row(node2);
    
    // the two mirror faces, symetry of line node1-node2
    Row<int> A(2); 
    int shu = -1;
    for (int i = 0; i < aface.n_cols; i++){
        for (int j = 0; j < bface.n_cols; j++){
            if ( aface(i) == bface(j) && aface(i) != -1 ){
                shu = shu + 1;
                A(shu) = aface(i);
            }
        }
    }

    // then, the two mirror nodes, symetry of line node1-node2
    Row<int> dot(2);
    for (int j = 0; j < 2; j++){
        for (int k = 0; k < 3; k++) {
            if ( face(A(j),k) != node1 && face(A(j),k) != node2 ){ 
                dot(j) = face(A(j),k);
            }
        }
    }

    return dot; 
}

void setsphere_Loop_scheme(mat& vertex, Mat<int>& face, double& meanl, double sphereRadius){
    double r = sphereRadius * 1.42857; // nm
    double l = 2.0; // nm. these two parameters will make a sphere of R = 28 nm. 

    double a = r*2.0*sin(M_PI/5.0)/(sqrt(4.0*pow(sin(M_PI/5.0),2.0)-1.0)+0.5*cos(M_PI/5.0)); // a is the side length of icosahedron
    int n = round(log(a/l)/log(2.0));                                // division times to make the side-length as l.
    seticosahedron(vertex, face, r);                               // set an icosahedron for division into a sphere   
    for (int j = 0; j < n; j++){
        mat oldvertex; oldvertex = vertex;
        Mat<int> oldface; oldface = face;
        //Mat<int> vertexi_face(vertex.n_rows,6);  vertexi_face.fill(-1); // all faces that has this vertex_i
        //vertexi_face_with_it(oldvertex, oldface, vertexi_face); // if vertexi_face(x,5)=-1, means this vertex has only 5 nearby faces
        Mat<int> vertexi_face = faces_with_vertexi(oldvertex, oldface);
        //Mat<int> vertexi_nearby(vertex.n_rows, 6); vertexi_nearby.fill(-1); // all nearby vertices around this vertex_i, 5 or 6.
        //neighbor_vertices(vertexi_face, oldface, vertexi_nearby);  
        Mat<int> vertexi_nearby = vertex_valence(vertexi_face, oldface);
        int facenumber = oldface.n_rows;
        Mat<int> newface(facenumber*3, 3);
        for (int i=0; i < facenumber; i++){
            int vertexnumber = vertex.n_rows;
            ///////////////////////////////////////////////////////
            //  new vertices
            Row<int> dot1 = find_two_mirror_nodes(oldface(i,0),oldface(i,1),vertexi_face,oldface);
            rowvec newvertex1 = 3.0/8.0*(vertex.row(oldface(i,0))+vertex.row(oldface(i,1)))+1.0/8.0*(vertex.row(dot1(0))+vertex.row(dot1(1)));
            
            Row<int> dot2 = find_two_mirror_nodes(oldface(i,1),oldface(i,2),vertexi_face,oldface);
            rowvec newvertex2 = 3.0/8.0*(vertex.row(oldface(i,1))+vertex.row(oldface(i,2)))+1.0/8.0*(vertex.row(dot2(0))+vertex.row(dot2(1)));
            
            Row<int> dot3 = find_two_mirror_nodes(oldface(i,2),oldface(i,0),vertexi_face,oldface);
            rowvec newvertex3 = 3.0/8.0*(vertex.row(oldface(i,0))+vertex.row(oldface(i,2)))+1.0/8.0*(vertex.row(dot3(0))+vertex.row(dot3(1)));
            
            // check whether the new vertexs are actually those existed ones
            int n1 = 1; int n2 = 1; int n3 = 1;
            int new1, new2, new3;
            for (int k = 0; k < vertexnumber; k++) { 
                if ( norm(newvertex1-vertex.row(k),2) < 1.0e-9 ){ // alread existed 
                    n1 = 0;
                    new1 = k;
                }
                if ( norm(newvertex2-vertex.row(k),2) < 1.0e-9 ){
                    n2 = 0;
                    new2 = k;
                }
                if ( norm(newvertex3-vertex.row(k),2) < 1.0e-9 ){
                    n3 = 0;
                    new3 = k;
                }
            }
            if (n1 == 1){                     // if the new vertex is totally new,not same as existed one
                new1 = n1 + vertexnumber - 1;       // the new vertex's number 
                int k = vertex.n_rows;
                vertex.insert_rows(k, newvertex1); // add the new vertex to matrix
            }
            if (n2 == 1){
                new2 = n1 + n2 + vertexnumber - 1;
                int k = vertex.n_rows;
                vertex.insert_rows(k, newvertex2);
            }
            if (n3 == 1){
                new3 = n1 + n2 + n3+vertexnumber - 1;
                int k = vertex.n_rows;
                vertex.insert_rows(k, newvertex3);
            }
            /////////////////////////////////////////////////
            // new face
            Row<int> temp(3); 
            temp << oldface(i,0) << new1 << new3;
            newface.row(3*i) = temp ;
            temp << oldface(i,1) << new2 << new1;
            newface.row(3*i+1) = temp;
            temp << oldface(i,2) << new3 << new2;
            newface.row(3*i+2) = temp;
            temp << new1 << new2 << new3;
            face.row(i) = temp;
        }
        // update the positions of oldvertices
        for (int i = 0; i < oldvertex.n_rows; i++){
            int N = 6;
            if ( vertexi_nearby(i,5) < 0 ){
                N = 5;
            }
            // w=1/N*(5/8-(3/8+1/4*cos(2*pi/N))^2);
            double w = 3.0/8.0/N; // w=1/(N+3/8/w);
            oldvertex.row(i)=(1.0 - N*w) * oldvertex.row(i);
            for (int k = 0; k < N; k++){
                oldvertex.row(i) = oldvertex.row(i) + w * oldvertex.row(vertexi_nearby(i,k));
            }
        }
        // update the matrix for face and vertex.
        oldface = face;
        face.set_size(oldface.n_rows + newface.n_rows, oldface.n_cols);
        face.rows(0,oldface.n_rows-1) = oldface;
        face.rows(oldface.n_rows, oldface.n_rows + newface.n_rows-1) = newface;
        vertex.rows(0, oldvertex.n_rows-1) = oldvertex;
    }

    //////////////////////////////////////////////////////////
    // adjust the vertex to make the sphere radius as target
    //for (int i = 0; i < vertex.n_rows; i++){
    //    double radiusTmp = norm(vertex.row(i), 2);
    //    vertex.row(i) = vertex.row(i) * sphereRadius/radiusTmp;
    //}

    //////////////////////////////////////////////////////////
    // calculate the mean side-length 
    vec sidelength(3*face.n_rows); 
    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        sidelength(3*i) = norm(vertex.row(face(i,0))-vertex.row(face(i,1)),2);
        sidelength(3*i+1) = norm(vertex.row(face(i,0))-vertex.row(face(i,2)),2);
        sidelength(3*i+2) = norm(vertex.row(face(i,2))-vertex.row(face(i,1)),2);
    }
    meanl = mean(sidelength);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for *********

double get_magnitude(rowvec targvec){
    double out;
    for (int i = 0; i < targvec.n_cols; i++){
        out = out + targvec(i)*targvec(i);
    }
    return sqrt(out);
}

// the following functions are for nick insertions.

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
        rowvec radiusdirect = 0.5*( vertex.row(vertex1) + vertex.row(vertex2) );
        radiusdirect = radiusdirect / get_magnitude(radiusdirect);
        rowvec leftmove = cross(radiusdirect,insertdirect); 
        //leftmove = leftmove * sqrt(3.0)/4.0*meanL/5.0;
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
                mat shu = cross(center,radiusdirect)*strans(insertdirect);
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for box spline

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for subdivision matrix for different patches

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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for *********

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

vec force_scale(mat forcein, mat forcemid, mat forceout){
    int numin = forcein.n_rows;
    int nummid = forcemid.n_rows;
    int numout = forceout.n_rows;
    vec out(numin+nummid+numout);
    #pragma omp parallel for
    for (int i = 0; i < numin; i++){
        out(i) = norm(forcein.row(i),2);
    }
    #pragma omp parallel for
    for (int i = numin; i < numin+nummid; i++){
        out(i) = norm(forcemid.row(i-numin),2);
    }
    #pragma omp parallel for
    for (int i = numin+nummid; i < numin+nummid+numout; i++){
        out(i) = norm(forceout.row(i-numin-nummid),2);
    }
    return out;
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



/*
rowvec determine_spontaneous_curvature(bool isInnerLayer, Param param, Mat<int> face, mat vertex, Mat<int> insertionpatch){
    double C0 = param.Cout;
    double c0 = param.c0out;
    double R  = param.Rout;
    if ( isInnerLayer == true ){
        C0 = param.Cin;
        c0 = param.c0in;
        R  = param.Rin;
    }
    double sigma = param.sigma; // 2*sigma is the radial region of non-zero spontaneous curvature
    bool isAdditiveScheme = param.isAdditiveScheme;
    rowvec spontcurv(face.n_rows); spontcurv.fill(c0);
    
    for (int i = 0; i < insertionpatch.n_rows; i++){
        for (int j = 0; j < insertionpatch.n_cols; j++){
            int facenumber = insertionpatch(i,j);
            spontcurv(facenumber) = C0;
        }
    }
    if ( sigma < 1e-9 ) 
        return spontcurv;
    
    if (insertionpatch.n_rows < 2){
        for (int i = 0; i < face.n_rows; i++ ){
            if (face(i,0) < 0) continue;
            int j = 0;
            rowvec centeri = 1.0/3.0 * ( vertex.row(face(i,0)) + vertex.row(face(i,1)) + vertex.row(face(i,2)) );
            double dismin = 2.0*R;
            for (int k = 0; k < insertionpatch.n_cols; k++){
                int facenumber = insertionpatch(j,k);
                rowvec center0 = 1.0/3.0 * ( vertex.row(face(facenumber,0)) + vertex.row(face(facenumber,1)) + vertex.row(face(facenumber,2)) );
                rowvec disvec = centeri - center0;
                double dis = norm(disvec,2.0);
                double distance = 2.0*R * asin(dis/2.0/R);
                if (distance < dismin){
                    dismin = distance;
                }
            }
            spontcurv(i) = - abs(C0) * exp( - pow(dismin/sigma,2.0)/2.0 ); // here the spont_curvature is negative sign
            if ( abs(spontcurv(i)) < 1.0e-15 )
                spontcurv(i) = 0.0;
        }
        return spontcurv;
    }
    
    for (int i = 0; i < face.n_rows; i++ ){
        rowvec centeri = 1.0/3.0 * ( vertex.row(face(i,0)) + vertex.row(face(i,1)) + vertex.row(face(i,2)) );
        rowvec spontcurvs(insertionpatch.n_rows); spontcurvs.fill(0.0);
        for (int j = 0; j < insertionpatch.n_rows; j++){
            double dismin = 2.0*R;
            for (int k = 0; k < insertionpatch.n_cols; k++){
                int facenumber = insertionpatch(j,k);
                rowvec center0 = 1.0/3.0 * ( vertex.row(face(facenumber,0)) + vertex.row(face(facenumber,1)) + vertex.row(face(facenumber,2)) );
                rowvec disvec = centeri - center0;
                double dis = norm(disvec,2.0);
                double distance = 2.0*R * asin(dis/2.0/R);
                if (distance < dismin){
                    dismin = distance;
                }
            }
            spontcurvs(j) = abs(C0) * exp( - pow(dismin/sigma,2.0)/2.0 );
        }
        if (isAdditiveScheme == true){
            double shu = 0.0;
            double H = 1.0/R;
            for (int j = 0; j < insertionpatch.n_rows; j++){
                shu = shu + pow(2.0*H-spontcurvs(j),2.0);
            }
            shu = shu - pow(2.0*H,2.0);
            if (shu < 0.0){ 
                cout<<"Note: when decide the enhanced spontaneous curvature, nonlinear effect happens. Then no additive scheme is utilized! "<<endl;
                spontcurv(i) = - max(spontcurvs);
            }else{
                if (max(spontcurvs) >= 2.0*H){
                    spontcurv(i) = -(2.0*H + sqrt(shu)); // here the spont_curvature is negative sign
                }else{
                    spontcurv(i) = -(2.0*H - sqrt(shu)); 
                }
                if ( spontcurv(i) > 0.0 ){
                    cout<<"Wrong: not efficient spontaneous curvature is calculated! Exit!"<<endl;
                    exit(0);
                }
            }
        }else{
            spontcurv(i) = - max(spontcurvs); // here the spont_curvature is negative sign
        }
        
        if ( abs(spontcurv(i)) < 1.0e-15 )
            spontcurv(i) = 0.0;
    }
    return spontcurv;
}
*/

mat setup_inner_mesh(mat vertex, double h){
    int vertexnum = vertex.n_rows;
    mat vertexin(vertexnum,3);
    for (int i = 0; i < vertexnum; i++){
        double r = norm(vertex.row(i),2.0);
        vertexin.row(i) = (r-h)/r * vertex.row(i);
    }
    return vertexin;
}

double getsum(rowvec target){
    double out = 0.0;
    #pragma omp parallel for reduction(+:out)
    for (int i = 0; i < target.n_cols; i++){
        out = out + target(i);
    }
    return out;
}

Mat<int> select_insertionVertex(Mat<int> face, mat vertex, Mat<int> faces_with_nodei, double l, double insertLength, int insertionPatchNum, double distance){
    int selectVertexNum = round(insertLength/l) + 1; // each insertionVertex has this number of selected vertices
    Mat<int> insertVertex;
    if ( insertionPatchNum == 0 ){
        return insertVertex;
    }else{
        insertVertex.set_size(insertionPatchNum,selectVertexNum);
    }

    // to find the first vertex 
    int node0 = 0;
    rowvec direction(3); direction << -1.0 << -1.0 << 0.0 << endr; 
    direction = direction/norm(direction,2);
    double shu = 0.0;
    /*
    for (int i = 0; i < vertex.n_rows; i++){
        rowvec vertextmp = vertex.row(i);
        mat timetmp = direction * strans(vertextmp) / norm(vertextmp,2);
        if ( timetmp(0,0) > shu ){
            shu = timetmp(0,0);
            node0 = i;
        }
    }
    */
    node0 = face(0,0); 
    // to find the second vertex 
    int node1;
    direction = vertex.row(0) - vertex.row(node0); direction = direction / norm(direction,2);
    shu = 0.0;
    for ( int i = 0; i < faces_with_nodei.n_cols; i++ ){
        int facenum = faces_with_nodei(node0,i);
        if ( facenum == -1 ) continue;
        for ( int j = 0; j < 3; j++ ){
            int nodetmp = face(facenum,j);
            if ( nodetmp == node0 ) continue;
            rowvec directtmp = vertex.row(nodetmp) - vertex.row(node0); 
            mat timetmp = direction * strans(directtmp) / norm(directtmp,2);
            if ( timetmp(0,0) > shu ){
                shu = timetmp(0,0);
                node1 = nodetmp;
            }
        }
    }
    //
    insertVertex(0,0) = node0; insertVertex(0,1) = node1; 
    // to find the following insertVertex
    direction = vertex.row(node1) - vertex.row(node0); direction = direction / norm(direction,2);
    for ( int i = 2; i < selectVertexNum; i++ ){
        int nodetarget;
        int nodec = insertVertex(0,i-1);
        shu = 0.0;
        for ( int j = 0; j < faces_with_nodei.n_cols; j++ ){
            int facenum = faces_with_nodei(nodec,j);
            if ( facenum == -1 ) continue;
            for ( int k = 0; k < 3; k++ ){
                int nodetmp = face(facenum,k);
                if ( nodetmp == nodec ) continue;
                rowvec directtmp = vertex.row(nodetmp) - vertex.row(nodec); 
                mat timetmp = direction * strans(directtmp) / norm(directtmp,2);
                if ( timetmp(0,0) > shu ){
                    shu = timetmp(0,0);
                    nodetarget = nodetmp;
                }
            }
        }
        insertVertex(0,i) = nodetarget;
    }
    
    // for multiple insertions
    rowvec radiuss(vertex.n_rows); radiuss.fill(0);
    for (int i = 0; i < vertex.n_rows; i++){
        radiuss(i) = norm(vertex.row(i),2.0);
    }
    double radius = mean(radiuss);
    double distancetarget = 2.0*radius * sin(distance/2.0/radius); 
    node0 = insertVertex(0,0); int noden = insertVertex(0,selectVertexNum-1);
    direction = vertex.row(noden) - vertex.row(node0); direction = direction / norm(direction,2.0);
    if ( insertionPatchNum == 2 ){
        rowvec searchdirect = cross(direction,vertex.row(node0)); // it could be opsite direction
        searchdirect = searchdirect / norm(searchdirect,2.0);
        double lamda = radius * tan(distance/radius); // note, only works for distance < radius
        rowvec directiontmp = vertex.row(node0) + lamda*searchdirect; directiontmp = directiontmp /norm(directiontmp,2.0);
        int node10;
        shu = 0.0;
        for (int i = 0; i < vertex.n_rows; i++){
            rowvec vertextmp = vertex.row(i);
            mat timetmp = directiontmp * strans(vertextmp) / norm(vertextmp,2);
            if ( timetmp(0,0) > shu ){
                shu = timetmp(0,0);
                node10 = i;
            }
        }
        int node11;
        shu = 0.0;
        for ( int i = 0; i < faces_with_nodei.n_cols; i++ ){
            int facenum = faces_with_nodei(node10,i);
            if ( facenum == -1 ) continue;
            for ( int j = 0; j < 3; j++ ){
                int nodetmp = face(facenum,j);
                if ( nodetmp == node10 ) continue;
                rowvec directtmp = vertex.row(nodetmp) - vertex.row(node10); 
                mat timetmp = direction * strans(directtmp) / norm(directtmp,2);
                if ( timetmp(0,0) > shu ){
                    shu = timetmp(0,0);
                    node11 = nodetmp;
                }
            }
        }
        insertVertex(1,0) = node10; insertVertex(1,1) = node11; 
        // to find the following nodes
        direction = vertex.row(node11) - vertex.row(node10); direction = direction / norm(direction,2);
        for ( int i = 2; i < selectVertexNum; i++ ){
            int nodetarget;
            int nodec = insertVertex(1,i-1);
            shu = 0.0;
            for ( int j = 0; j < faces_with_nodei.n_cols; j++ ){
                int facenum = faces_with_nodei(nodec,j);
                if ( facenum == -1 ) continue;
                for ( int k = 0; k < 3; k++ ){
                    int nodetmp = face(facenum,k);
                    if ( nodetmp == nodec ) continue;
                    rowvec directtmp = vertex.row(nodetmp) - vertex.row(nodec); 
                    mat timetmp = direction * strans(directtmp) / norm(directtmp,2);
                    if ( timetmp(0,0) > shu ){
                        shu = timetmp(0,0);
                        nodetarget = nodetmp;
                    }
                }
            }
            insertVertex(1,i) = nodetarget;
        }
    }
    
    return insertVertex;
}
/*
void setup_insertion_local_area(double insertLength, double insertWidth, double l, Mat<int> faceout, mat vertexout, Mat<int> insertionVertex, Mat<int> insertionpatch, rowvec& elementS0out){
    // determine the target vertex positions of the insertion patch
    mat vertexouttmp = vertexout;
    if ( insertionVertex.n_cols > 2 && insertionVertex.n_rows > 0 )
    {   // each insertion patch has ellipse shape with long and short radius: a = 1.25 nm, b = 0.5093 nm, area=pi*a*b= 2.0 nm2
        double a = insertLength/2.0; double b = insertWidth/2.0;
        // adjust all insertionVertex, to make a target shape
        for ( int i = 0; i < insertionVertex.n_rows; i++) {
        	// two end nodes
            int node0 = insertionVertex(i,0);
            int noden = insertionVertex(i,insertionVertex.n_cols-1);
            rowvec direction = vertexout.row(noden) - vertexout.row(node0);  double side = norm(direction,2.0);
            direction = direction / side;
            double sidetarget = 2.0*a; 
            vertexouttmp.row(node0) += 0.0 * (-direction);
            vertexouttmp.row(noden) += (sidetarget-side) * direction;
            double ratio = sidetarget / side;
            // mid nodes
            for (int j = 1; j < insertionVertex.n_cols-1; j++){
                //double dl = j*l; // distance to node0;
                //double dL = a/1.25 * dl;

                int nodei = insertionVertex(i,j);
                int nodeii = (vertexout.n_rows-(insertionVertex.n_cols-2)*insertionVertex.n_rows) + (insertionVertex.n_cols-2)*i + j-1;
                
                double dl = norm(vertexout.row(nodei)-vertexout.row(node0),2.0);
                double dL = ratio * dl;

                // nodei's partner, new added node due to the insertion
                rowvec directioni = vertexout.row(nodei) - vertexout.row(nodeii); double sidei = norm(directioni,2.0);
                directioni = directioni / sidei;
                double sideitarget = 2.0*b/a*sqrt(a*a-pow(a-dL,2.0));
                vertexouttmp.row(nodei) += (sideitarget-sidei)/2.0 * directioni;
                vertexouttmp.row(nodeii) += (sideitarget-sidei)/2.0 * (-directioni);

                // to the end nodes of the insertion
                // to node0
                vertexouttmp.row(nodei) += (dL-dl) * (direction);
                vertexouttmp.row(nodeii) += (dL-dl) * (direction);
            }
        }
        // calculate the element area
        for ( int i = 0; i < insertionpatch.n_rows; i++ ){
            for ( int j = 0; j < insertionpatch.n_cols; j++ ){
                int facenum = insertionpatch(i,j);
                int node0 = faceout(facenum,0);
                int node1 = faceout(facenum,1);
                int node2 = faceout(facenum,2);
                double side0 = norm( vertexouttmp.row(node0)-vertexouttmp.row(node1), 2.0 );
                double side1 = norm( vertexouttmp.row(node1)-vertexouttmp.row(node2), 2.0 );
                double side2 = norm( vertexouttmp.row(node2)-vertexouttmp.row(node0), 2.0 );
                double sidem = (side0 + side1 + side2)/2.0;
                elementS0out(facenum) = sqrt(sidem*(sidem-side0)*(sidem-side1)*(sidem-side2));
            }
        }
    } 
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
    Row<int> outBoundaryRingFaceIndex;
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

void finer_local_mesh(int centerVertexIndex, int localRounds, mat& vertex, Mat<int>& face, Mat<int>& faces_with_nodei, LocalFinerMesh& localFinerMesh){
    // find a center vertex, then select several rounds of faces around it. Note, all the faces must be regular patches. 

    int vertexnum = vertex.n_rows;
    int facenum = face.n_rows;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find the center vertex. 
    // int centerVertexIndex = 0;
    // for sphere membrane, the centerVertex is one of the face_index = 6, which is consistent with select_insertionPatches. 
    // bool isSphere = true;
    // if ( isSphere == true ){
    //     centerVertexIndex = 2580; // face(6,0); 
    // }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find how many round of faces need to be subdivided around the centerVertex.
    // int localRounds = round(localRadius / edgeLength); 
    if ( localRounds < 2 ){
        cout<< "Wrong: In finer_local_mesh, too small lcoal area is selected for subdivision!" << endl;
        exit(0);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find each round of faces around the centerVertex.
    Mat<int> selectLocalFaces = select_local_face_for_finer(face, vertex, faces_with_nodei, centerVertexIndex, localRounds);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // subdivide each select face except the last round! 
    // For faces of the last round, some of them need to be bisected, but some do not!
    vector<bool> needBisect(selectLocalFaces.n_cols, false);
    Mat<int> twoVertexIndex(selectLocalFaces.n_cols, 2); twoVertexIndex.fill(-1);
    for ( int i = 0; i < selectLocalFaces.n_cols; i++ ){
        int facetmp = selectLocalFaces(selectLocalFaces.n_rows-1, i);
        if ( facetmp == -1 ) continue;
        // check whether this face needs to be bisect! The criterion is: this face has two common vertices with one of the inner ring face 
        Row<int> threeNodes1 = face.row(facetmp);
        for ( int j = 0; j < selectLocalFaces.n_cols; j++ ){
            int facetmpInner = selectLocalFaces(selectLocalFaces.n_rows-2, j); // the second most boundary ring
            if ( facetmpInner == -1 ) continue;
            Row<int> threeNodes2 = face.row(facetmpInner);
            Row<int> commonVertexIndex = intersect(threeNodes1,threeNodes2);
            if ( commonVertexIndex.n_cols == 2 ){
                needBisect[i] = true;
                twoVertexIndex.row(i) = commonVertexIndex;               
                break;
            }
        }
    }
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // For faces of the last round, some of them need to be bisected, but some do not!
    for ( int i = 0; i < selectLocalFaces.n_cols; i++ ){
        int facetmp = selectLocalFaces(selectLocalFaces.n_rows-1, i);
        if ( facetmp == -1 ) continue;
        // bisect the face
        if ( needBisect[i] == true ){
            // order the 3 vertice, make the first two on the bisection line, and the order is counter-clockwise
            int node0 = twoVertexIndex(i,0);
            int node1 = twoVertexIndex(i,1);
            int node2;
            for (int j = 0; j < 3; j++){
                int nodetmp = face(facetmp, j);
                if ( nodetmp != node0 && nodetmp != node1 ){
                    node2 = nodetmp;
                }
            }
            rowvec vec1 = vertex.row(node1) - vertex.row(node0);
            rowvec vec2 = vertex.row(node2) - vertex.row(node0);
            mat shutmp = cross(vec1,vec2) * strans( vertex.row(node0) );
            if ( shutmp(0,0) < 0.0 ){
                int nodetmp = node0;
                node0 = node1;
                node1 = nodetmp;
                node2 = node2;
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

void build_local_finer_mesh(mat& vertex, Mat<int>& face, double& finestL, double radiusForFinest, Mat<int>& faces_with_nodei, LocalFinerMesh& localFinerMesh){
    
    if ( radiusForFinest < finestL ){
        cout<<"   No need for local-finer, because the selected area is too small!" <<endl;
        return;
    }
    
    // Find the vertex, around which the mesh will be finer on the sphere. 
    int centerVertexIndex; // = face(6,0); 
    double critmp = 1.0e-9;
    rowvec targetVec = vertex.row(1) + vertex.row(2); targetVec = targetVec / norm(targetVec, 2.0);
    for ( int i = 0; i < vertex.n_rows; i++ ){
        mat mattmp = vertex.row(i) * strans(targetVec);
        if ( mattmp(0,0) > critmp ){
            critmp = mattmp(0,0);
            centerVertexIndex = i;
        }
    }
    
    // the smallest edge of the mesh, about half the lipid edge. The target edge length around the insertion is 0.5 nm. 
    double targetEdge = 0.5; 

    // determine how many times of local-finer are needed.
    int times = round(log2(finestL/targetEdge)); 
    
    if ( times < 1 ){
        cout<<"   No need for local-finer, because the mesh is fine enough!" <<endl;
        return;
    }

    double radiusForFinestAdjusted = ceil( (radiusForFinest + 2.0/2.0) / finestL) * finestL;
    double radiusForLocalFiner = radiusForFinestAdjusted / pow(0.8, times-1);
    int localRounds = round( radiusForLocalFiner/finestL );
    
    if ( times >= 1){
        for ( int i = 0; i < times; i++ ){
            finer_local_mesh(centerVertexIndex, localRounds, vertex, face, faces_with_nodei, localFinerMesh);
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

        /////////////////////////////////////////////////////////////////////////////////////
        // rearragne the face index, because the new-added faces are all listed at the bottom, which has more irregular patches and will be assigned to the same node in openmpi, time-consuming.
        vector<int> faceIndex(face.n_rows); iota(faceIndex.begin(), faceIndex.end(), 0); 
        srand(500);
        std::random_shuffle(faceIndex.begin(), faceIndex.end());
        Mat<int> faceRearranged(face.n_rows,3); 
        for ( int i = 0; i < face.n_rows; i++ ){
            int faceindex = faceIndex[i];
            faceRearranged.row(i) =  face.row(faceindex);
        }
        face = faceRearranged;
        // update faces_with_nodei
        faces_with_nodei = faces_with_vertexi(vertex, face); 
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


rowvec determine_finestEdgeLength(Mat<int> face, mat vertexin, mat vertexmid, mat vertexout, LocalFinerMesh localFinerMesh){
    rowvec edgeLength(3); edgeLength.fill(0.0);
    // the 1st element is on the inner layer mesh
    // the 2nd element is on the middle layer mesh  
    // the 3rd element is on the outer layer mesh
    
    if ( localFinerMesh.buildLocalFinerMesh == true ){
        // calculate the mean edge length of each local-finer-face.  
        Row<int> finerFaces = localFinerMesh.faceIndex;
    
        rowvec edgeLengthEachFace(finerFaces.n_cols);
        for ( int j = 0; j < finerFaces.n_cols; j++ ){
            int i = finerFaces(j);
            double sidelength1 = norm(vertexin.row(face(i,0))-vertexin.row(face(i,1)),2);
            double sidelength2 = norm(vertexin.row(face(i,0))-vertexin.row(face(i,2)),2);
            double sidelength3 = norm(vertexin.row(face(i,2))-vertexin.row(face(i,1)),2);
            edgeLengthEachFace(j) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );
        }
        edgeLength(0) = min(edgeLengthEachFace);

        edgeLengthEachFace.fill(0.0);
        for ( int j = 0; j < finerFaces.n_cols; j++ ){
            int i = finerFaces(j);
            double sidelength1 = norm(vertexmid.row(face(i,0))-vertexmid.row(face(i,1)),2);
            double sidelength2 = norm(vertexmid.row(face(i,0))-vertexmid.row(face(i,2)),2);
            double sidelength3 = norm(vertexmid.row(face(i,2))-vertexmid.row(face(i,1)),2);
            edgeLengthEachFace(j) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );
        }
        edgeLength(1) = min(edgeLengthEachFace);

        edgeLengthEachFace.fill(0.0);
        for ( int j = 0; j < finerFaces.n_cols; j++ ){
            int i = finerFaces(j);
            double sidelength1 = norm(vertexout.row(face(i,0))-vertexout.row(face(i,1)),2);
            double sidelength2 = norm(vertexout.row(face(i,0))-vertexout.row(face(i,2)),2);
            double sidelength3 = norm(vertexout.row(face(i,2))-vertexout.row(face(i,1)),2);
            edgeLengthEachFace(j) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );
        }
        edgeLength(2) = min(edgeLengthEachFace);
    }else{
        int i = 6; // select faceindex 6 as the sample

        double sidelength1 = norm(vertexin.row(face(i,0))-vertexin.row(face(i,1)),2);
        double sidelength2 = norm(vertexin.row(face(i,0))-vertexin.row(face(i,2)),2);
        double sidelength3 = norm(vertexin.row(face(i,2))-vertexin.row(face(i,1)),2);
        edgeLength(0) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );

        sidelength1 = norm(vertexmid.row(face(i,0))-vertexmid.row(face(i,1)),2);
        sidelength2 = norm(vertexmid.row(face(i,0))-vertexmid.row(face(i,2)),2);
        sidelength3 = norm(vertexmid.row(face(i,2))-vertexmid.row(face(i,1)),2);
        edgeLength(1) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );

        sidelength1 = norm(vertexout.row(face(i,0))-vertexout.row(face(i,1)),2);
        sidelength2 = norm(vertexout.row(face(i,0))-vertexout.row(face(i,2)),2);
        sidelength3 = norm(vertexout.row(face(i,2))-vertexout.row(face(i,1)),2);
        edgeLength(2) = 1.0/3.0 * ( sidelength1 + sidelength2 + sidelength3 );
    }

    return edgeLength;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the following functions are for selection_insertionPatch

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

Row<int> determine_parallel_insertionPatch(Mat<int> face, mat vertex, Mat<int> faces_with_nodei, double meanL, double insertLength, double insertWidth, double radius, rowvec InsertDirection, rowvec SearchDirect, rowvec Center0, double distBetwn){
        double triangleS = sqrt(3.0)/4.0 * meanL * meanL;
        int n2 = round(insertWidth/(sqrt(3.0)/2.0*meanL));
        int n1 = round( insertLength * insertWidth / triangleS / n2 );

        Row<int> insertionpatch(n1*n2); 
        // for two insertions
        // find the first face along directn2
        rowvec searchdirect = SearchDirect; // unit vector
        rowvec center0 = Center0; 

        double angle = distBetwn/radius;
        if ( angle > M_PI ){
            cout<<"Wrong: The distance between two insertions is larger than half-circumference! Exit!"<<endl;
            exit(0);
        }
        double lamda = radius * sin(angle);
        double peta  = radius - radius * cos(angle);
        rowvec directiontmp = center0 + lamda*searchdirect - peta/norm(center0,2)*center0; 
        directiontmp = directiontmp /norm(directiontmp,2.0);

        double shu = 0.0;
        int face10;
        for (int i = 0; i < face.n_rows; i++){
            rowvec centertmp = 1.0/3.0*( vertex.row(face(i,0)) + vertex.row(face(i,1)) + vertex.row(face(i,2)) );
            mat timetmp = directiontmp * strans(centertmp) / norm(centertmp,2);
            if ( timetmp(0,0) > shu ){
                shu = timetmp(0,0);
                face10 = i;
            }
        }
        insertionpatch(0) = face10;
        center0 = 1.0/3.0*( vertex.row(face(face10,0)) + vertex.row(face(face10,1)) + vertex.row(face(face10,2)) );
        // adjust the insert_direction according to face10
        rowvec direction = InsertDirection;
        {
            double shu = 0.0;
            int node0, node1;
            for ( int i = 0; i < 3; i++ ){
                int nodei = face(face10,i);
                int nodei1;
                if ( i+1 > 2){ 
                    nodei1 = face(face10,0);
                }else{
                    nodei1 = face(face10,i+1);
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
            direction = vertex.row(node1) - vertex.row(node0); direction = direction / norm(direction,2); // adjust this direction along vertex
        }
        // row0 patches along n1
        for ( int i = 1; i < n1; i++ ){
            int facepre = insertionpatch(i-1); // prior face index
            rowvec centerpre = 1.0/3.0*( vertex.row(face(facepre,0)) + vertex.row(face(facepre,1)) + vertex.row(face(facepre,2)) );
            int facetarget = -1;
            shu = 0.0;
            for ( int j = 0; j < 3; j++ ){
                int nodei = face(facepre,j);
                int nodei1;
                if ( j+1 > 2){ 
                    nodei1 = face(facepre,0);
                }else{
                    nodei1 = face(facepre,j+1);
                }
                int facetmp = find_face(nodei,nodei1,facepre,faces_with_nodei);
                rowvec centertmp = 1.0/3.0*( vertex.row(face(facetmp,0)) + vertex.row(face(facetmp,1)) + vertex.row(face(facetmp,2)) );
                rowvec directtmp = centertmp - centerpre; directtmp = directtmp / norm(directtmp,2);
                mat shutmp = directtmp * strans(direction);
                if (  shutmp(0,0) > shu ){
                    shu = shutmp(0,0);
                    facetarget = facetmp;
                }
            }
            insertionpatch(i) = facetarget;
        }
        // row0 patches along n2
        rowvec directn2 = cross(direction,center0); directn2 = directn2 / norm(directn2,2);
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
                        mat shutmp = directtmp * strans(directn2);
                        if (  shutmp(0,0) > shu ){
                            shu = shutmp(0,0);
                            facetarget = facetmp;
                        }
                    }
                }
                insertionpatch(indexnow) = facetarget;
            }
        } 
    return insertionpatch;    
}

Mat<int> select_insertionPatch(Mat<int> face, mat vertex, Mat<int> faces_with_nodei, LocalFinerMesh localFinerMesh, double meanL, double insertLength, double insertWidth, int insertionPatchNum, double distBetwn){
    // int n1 = round(insertLength/meanL);
    double triangleS = sqrt(3.0)/4.0 * meanL * meanL;
    int n2 = round(insertWidth/(sqrt(3.0)/2.0*meanL));
    int n1 = round( insertLength * insertWidth / triangleS / n2 );

    // for each insertionpatch, we need 2*n1 *n2 triangles
    Mat<int> insertionpatch;
    if ( insertionPatchNum == 0 ){
        return insertionpatch;
    }else{
        insertionpatch.set_size(insertionPatchNum, n1 * n2);
        insertionpatch.fill(-1);
    }

    // to find the first triangle, set it as the 0
    int face0 = 6; 
    rowvec center0 = 1.0/3.0*( vertex.row(face(face0,0)) + vertex.row(face(face0,1)) + vertex.row(face(face0,2)) );
    rowvec direction = vertex.row(0) - center0; direction = direction / norm(direction,2); // point to north pole
    if ( localFinerMesh.buildLocalFinerMesh == false ){
      double shu = 0.0;
      int node0, node1;
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
      direction = vertex.row(node1) - vertex.row(node0); direction = direction / norm(direction,2); // adjust this direction along vertex
    }else{
        int centerVertex = localFinerMesh.centerVertexIndex;
        int oneVertex = localFinerMesh.sixVerticeHexagon(0);
        direction = vertex.row(oneVertex) - vertex.row(centerVertex); direction = direction / norm(direction,2);

        // find the suitable face as the first face, considering the distance between insertions and the insertion size
        rowvec vec1 = vertex.row(localFinerMesh.sixVerticeHexagon(3)) - vertex.row(centerVertex); vec1 = vec1 / norm(vec1,2); 
        vec1 = vec1 * (insertLength/2.0);
        rowvec vec2 = cross(vertex.row(centerVertex), direction); vec2 = vec2 / norm(vec2,2);
        vec2 = vec2 * (insertWidth/2.0 + distBetwn*(insertionPatchNum-1)/2.0);
        vec2 = vec2 * 0.0;
        rowvec vectarget = vec1 + vec2 + vertex.row(centerVertex); // vectarget = vectarget / norm(vectarget,2);
        double criterion = 1.0e9;
        for ( int i = 0; i < localFinerMesh.faceIndex.n_cols; i++ ){ 
            int facetmp = localFinerMesh.faceIndex(i);
            for ( int k = 0; k < 3; k++ ){
                double dist = norm(vertex.row(face(facetmp,k))-vectarget, 2 );
                if ( dist < criterion ){
                    criterion = dist;
                    face0 = facetmp;
                    center0 = 1.0/3.0*( vertex.row(face(facetmp,0)) + vertex.row(face(facetmp,1)) + vertex.row(face(facetmp,2)) );
                }
            }
        }
    }
    insertionpatch(0,0) = face0;

    // row0 patches along n1
    for ( int i = 1; i < n1; i++ ){
        int facepre = insertionpatch(0,i-1); // prior face index
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
            int facetmp = find_face(nodei,nodei1,facepre,faces_with_nodei);
            rowvec centertmp = 1.0/3.0*( vertex.row(face(facetmp,0)) + vertex.row(face(facetmp,1)) + vertex.row(face(facetmp,2)) );
            rowvec directtmp = centertmp - centerpre; directtmp = directtmp / norm(directtmp,2);
            mat shutmp = directtmp * strans(direction);
            if (  shutmp(0,0) > shu ){
                shu = shutmp(0,0);
                facetarget = facetmp;
            }
        }
        insertionpatch(0,i) = facetarget;
    }
    
    // row0 patches along n2
    rowvec directn2 = cross(direction,center0); directn2 = directn2 / norm(directn2,2);
    // adjust directn2, if it pass one vertex of insertionpatch(0,0).
    for ( int i = 0; i < 3; i++ ){
        rowvec centerref = 1.0/3.0*( vertex.row(face(face0,0)) + vertex.row(face(face0,1)) + vertex.row(face(face0,2)) );
        rowvec directtmp = vertex.row(face(face0,i)) - centerref; directtmp = directtmp / norm(directtmp,2);
        mat shutmp = directn2 * strans( directtmp );
        if ( acos(shutmp(0,0)) < 40.0/180.0 * M_PI  ){
            directn2 = - directn2;
        }     
    }
    
    for ( int i = 1; i < n2; i++ ){
        for ( int j = 0; j < n1; j++ ){
            int indexnow = n1*i + j;
            int indexpre = n1*(i-1) + j;
            int faceref = insertionpatch(0,indexpre);
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
                    mat shutmp = directtmp * strans(directn2);
                    if (  shutmp(0,0) > shu ){
                        shu = shutmp(0,0);
                        facetarget = facetmp;
                    }
                }
            }
            insertionpatch(0,indexnow) = facetarget;
        }
    } 
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    // for multiple insertions, parallel to each other!!!
    rowvec radiuss(vertex.n_rows); radiuss.fill(0);
    for (int i = 0; i < vertex.n_rows; i++){
        radiuss(i) = norm(vertex.row(i),2.0);
    }
    double radius = mean(radiuss);
    
    for (int i = 1; i < insertionPatchNum; i++){
        double distancetmp = distBetwn * i; 
        Row<int> insertPatchtmp = determine_parallel_insertionPatch(face, vertex, faces_with_nodei, meanL, insertLength, insertWidth, radius, direction, directn2, center0, distancetmp);
        insertionpatch.row(i) = insertPatchtmp;
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
    /*
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
    */
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
double calculate_flipflopRatio(double R, double h1, double c01, double kc1, double us1, double h2, double c02, double kc2, double us2){
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
                exit(0);
                //return 2.0 * R * h /(pow(R,2.0) + pow(h,2.0));
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
