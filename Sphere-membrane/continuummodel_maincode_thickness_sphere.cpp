#include <math.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <armadillo>
#include <sstream>
#include <vector>
#include <string>
#include <omp.h>
#include "functions_main.cpp"

using namespace std;
using namespace arma;

// mesh parameters:
bool   isFlatMembrane = false;
double sphereRadius   = 90.0;                     // sphere radius, nm
double radiusForLocalFiner = 9.0;                // the size of local finer mesh
bool   isBoundaryFixed = false;                  // boundaries
bool   isBoundaryPeriodic  = false;
bool   isBoundaryFree = false;
double k_regularization   = (1.0e1)*10.0*4.17;   // coefficient of the regulerization constraint, 
bool   usingRpi_viscousRegul = false;
double gama_shape = 0.2;                         // factor of shape deformation
double gama_area = 0.2;                          // factor of size deformation
int    subDivideTimes = 5;                       // subdivision times for the irregular patches
int    GaussQuadratureN = 2;                     // Gaussian quadrature integral 

// membrane parameters: 
bool   isBilayerModel = true;
bool   isGlobalAreaConstraint = true;            // whether to use Global constraints for the area elasticity
bool   includeDivTilt  = true;
// out-layer (top layer)
double kc_out  = 19.4*4.17/2.0;                  // pN.nm. bending modulus, out-monolayer 
double kst_out = 0.0;//17.2*4.17;                    // pN.nm . splay-tilt modulus, out-monolayer. refer to Markus Deserno, J. Chem. Phys. 151, 164108 (2019)
double us_out  = 265.0/2.0;                      // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
double Ktilt_out  = 89.0;                        // pN/nm = mN/m. tilt modulus
double Kthick_out = 3.0 * Ktilt_out;             // pN/nm, coefficient of the membrane thickness. penalty term
double thickness_out = 2.71/2.0;                 // nm, out-monolayer thickness. 
double c0out = -0.04;                            // spontaneous curvature of membrane, out-layer. Convex is positive
// in-layer (bottom layer)
double kc_in  = kc_out;                          // pN.nm. bending modulus, in-monolayer 
double kst_in = kst_out;                          // pN.nm . Guassian Curvature modulus, in-monolayer
double us_in  = us_out;                          // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
double Ktilt_in  = Ktilt_out;                    // pN/nm = mN/m. tilt modulus
double Kthick_in = 3.0 * Ktilt_in;
double thickness_in = thickness_out;             // nm, in-monolayer thickness. 
double c0in  = c0out;                            // spontaneous curvature of membrane, in-layer. Concave is positive
double uv = 1.0e3;                               // coefficeint of the volume constraint, 0.5*uv*(dv)^2/v0;

double Kthick_constraint = 1.0e3 * 2.5;// 1GPa * thickness_out; 1GPa = 1e3 pN/nm2
int    stepsToIncreaseKthickConstraint = 100; 

// external forces: tension
double tension = 0.0;                            // pN/nm. 0.01~10
double ci  = 1.0; //us_out/(tension + us_out);          // area constraint target
double miu = 1.0;                                // volume constraint target

// insertion parameters:
int    insertionPatchNum = 1;                    // number of insertions
bool   isInsertionsParallel = true;
double distBetwn = 0.0;                          // distance between the insertions
double c0out_ins  = 0.3;                         // spontaneous curvature of insertion, outer layer
double s_insert  = 2.0;                          // insertion area
double insertLength = 2.0;                       // insertion size
double insertWidth = 1.0; 
double insert_dH0 = 0.17;                         // equilibrium value of thickness decrease induced by the insertion, nm
double K_insertShape   = 10.0*us_out;            // spring constant for insertion zones, to constraint the insertion shape

// parameters for simulation setup
int    N   = 1e5;                                // total step of iteration
double criterion_force = 1.0e-2;                 // critera for the equilibrium of the simulation
double criterion_E = 2.0e-5; 
double criterion_S = 1e-5;
double criterion_V = 1e-5;
double criterion_Er = 1e-5;

/////////////////////////////////////////////////////////////////////////////////
// main code
int main() {
    srand((unsigned)time(NULL)); 
    //////////////////////////////////////////////////////////
    // gauss_quadrature and shape functions
    cout<<"1. To build the shape functions for the differential geometry on different types of patches."<<endl;
    mat VWU = setVMU(GaussQuadratureN);
    cube shape_functions(12,7,VWU.n_rows);
    for (int i = 0; i < VWU.n_rows; i++) {
        rowvec vwu = VWU.row(i);
        mat sf = shapefunctions(vwu);          // 12 shape functions
        shape_functions.slice(i) = sf;
    }
    rowvec gqcoeff = setVMUcoefficient(GaussQuadratureN);
    SubMatrix subMatrix;
    subdivision_matrix_regular(subMatrix.regM, subMatrix.regM1, subMatrix.regM2, subMatrix.regM3, subMatrix.regM4);
    subdivision_matrix_irregular(subMatrix.irregM, subMatrix.irregM1, subMatrix.irregM2, subMatrix.irregM3, subMatrix.irregM4);
    subdivision_matrix_complex(subMatrix.comregM, subMatrix.comregM1, subMatrix.comregM2, subMatrix.comregM3, subMatrix.comregM4);
    subdivision_matrix_sudoregular1(subMatrix.sudoreg1M, subMatrix.sudoreg1M1, subMatrix.sudoreg1M2, subMatrix.sudoreg1M3, subMatrix.sudoreg1M4);
    subdivision_matrix_sudoregular2(subMatrix.sudoreg2M, subMatrix.sudoreg2M1, subMatrix.sudoreg2M2, subMatrix.sudoreg2M3, subMatrix.sudoreg2M4);
    
    // build the sphere (triangular mesh), and calculate some parameters of it
    cout<<"2. To build a spherical mesh with Loop's subdivision scheme."<<endl;
    double meanL = 0.0;     // the mean value of the triangular side length.
    Mat<double> vertex(12,3);  // vertex position
    Mat<int> face(20,3);    // face and its surrounding vertex
    setsphere_Loop_scheme(vertex, face, meanL, sphereRadius); // build a sample sphere mesh, radius is about 28 nm. 

    cout<<"   Sphere Radius (outLayer) = "<< norm(vertex.row(0), 2) <<" nm"<<endl;

    // boundary vertex or face is located on the most edge
    Row<int> isBoundaryNode(vertex.n_rows); isBoundaryNode.fill(-1);// = determine_BoundaryVertex(sideX, sideY, l); // element 1 means this vertex is on boundary
    Row<int> isBoundaryFace(face.n_rows); isBoundaryFace.fill(-1); // = determine_Boundaryface(face,isBoundaryNode); // element 1 means this face is on boundary
    Row<int> isGhostNode(vertex.n_rows); isGhostNode.fill(-1); // = determine_GhostVertex(sideX, sideY, l, isBoundaryFixed, isBoundaryPeriodic, isBoundaryFree); // element 1 means this vertex is ghost
    Row<int> isGhostFace(face.n_rows); isGhostFace.fill(-1); // = determine_GhostFace(sideX, sideY, l, isBoundaryFixed, isBoundaryPeriodic, isBoundaryFree); // element 1 means this face is ghost   
    
    Mat<int> faces_with_nodei = faces_with_vertexi(vertex, face);
    
    vector<int> ghostPartner(vertex.n_rows, -1); // = determine_GhostPartner(sideX, sideY, l);
    vector<vector<int>> freePartners(vertex.n_rows, vector<int>( 3, -1) ); // = determine_freePartners(sideX, sideY, l);

    ///////////////////////////////////////////////////////
    // Finer the mesh aroud the center of the membrane if the membrane is flat.
    // Meanwhile, the vertex, face need to be updated!
    cout<<"3. To make the local mesh finer."<<endl;
    LocalFinerMesh localFinerMesh;
    double finestL = meanL; 
    build_local_finer_mesh(vertex, face, finestL, radiusForLocalFiner, faces_with_nodei, localFinerMesh, isBoundaryNode, isBoundaryFace, isGhostNode, isGhostFace, ghostPartner, freePartners);
    if ( localFinerMesh.buildLocalFinerMesh == true ){
        Row<int> finerFaces = localFinerMesh.faceIndex + 1; finerFaces.print("   finer_Faces: ");
    }
    cout<< "   The number of faces:    "<< face.n_rows << endl;
    cout<< "   The number of vertices: "<< vertex.n_rows << endl;
    
    //////////////////////////////////////////////////////
    // set the target height of the in-layer and out-layer
    cout<<"4. To determine the target height or curvature-modified height of each monolayer."<<endl;
    double H0C_out = thickness_out;
    double H0C_in = thickness_in;
    if ( isBilayerModel == true ){
        determine_Target_Height_for_Each_Monolayer(thickness_in, thickness_out, vertex, H0C_in, H0C_out);
        cout<<"   The target height of the out-layer is "<< H0C_out <<" nm; the one of the in-layer is "<< H0C_in << " nm." <<endl;
    }else{
        cout<<"   Single-layer model is used, so no need to determine the target height."<<endl;
    }
    
    //////////////////////////////////////////////////////
    // set the inner layer and out layer
    if ( isBilayerModel == true ){
    cout<<"5. To build the multi-layer meshes, as the model of bilayer membrane."<<endl;
    }else{
    cout<<"5. To build the single-layer mesh, as the model of bilayer membrane."<<endl;   
    }
    Vertex3Layers vertex3;
    vertex3.outlayer = vertex;
    //double height_out = thickness_out * (1.0 - 2.0*thickness_out/sphereRadius); // curvature-modified height
    if ( isBilayerModel == true ){
        vertex3.midlayer = setup_inner_mesh(vertex, H0C_out); 
        vertex3.inlayer = setup_inner_mesh(vertex, H0C_out + H0C_in); 
    }

    cout<<"6. To output the initial structure: vertex_begin_* and face*."<<endl;
    // output the vertex and face matrix
    printout_initial_structures(face, vertex3, isBilayerModel);
    
    //////////////////////////////////////////////////////
    cout<<"7. To output the boundaryFace for the mesh."<<endl;
    // printout_isBoundaryFace(isBoundaryFace);

    ///////////////////////////////////////////////////////
    // select the insertion patches
    cout<<"8. To select insertionPatch, several triangles on the outer layer mesh as the helix insertion zone."<<endl;
    Mat<int> insertionpatch = select_insertionPatch(face, vertex3.outlayer, faces_with_nodei, localFinerMesh, finestL, insertLength, insertWidth, insertionPatchNum, distBetwn);   
    
    Mat<int> InsertionPatchPrint = insertionpatch + 1;
    InsertionPatchPrint.print("   InsertionPatch: ");
    
    IsInsertionpatch3Layers isInsertionPatch;
    isInsertionPatch.outlayer.resize(face.n_rows, false); 
    for (int i = 0; i < insertionpatch.n_rows; i++){
        for (int j = 0; j < insertionpatch.n_cols; j++){
            int facenum = insertionpatch(i,j); 
            isInsertionPatch.outlayer[facenum] = true; 
        }
    }
    if ( isBilayerModel == true ) isInsertionPatch.inlayer.resize(face.n_rows, false); 

    int zonenumber = insertionpatch.n_rows; // the number of insertion zones.

    ///////////////////////////////////////////////////////
    cout<<"9. To select the zone around the insertionPatch. This neighbor zone will intermittently have shape constraints."<<endl;
    Row<int> insertionpatchAdjacent = select_insertionPatchAdjacent(face, vertex3.outlayer, faces_with_nodei, insertionpatch);
    Row<int> insertionpatchAdjacentPrint = insertionpatchAdjacent + 1;
    insertionpatchAdjacentPrint.print("   InsertionPatchAdjacent: ");
    Row<int> IsinertionpatchAdjacent(face.n_rows); IsinertionpatchAdjacent.fill(-1);
    for (int i = 0; i < insertionpatchAdjacent.n_cols; i++){
        int facenum = insertionpatchAdjacent(i);
        IsinertionpatchAdjacent(facenum) = 1;
    }

    //////////////////////////////////////////////////////
    // update the vertex nearby and one-ring-vertex
    cout<<"10. To find the one-ring-vertex for each triangle."<<endl;
    // find out what faces that have vertex_i,6 or 5 or 7, by checking how many -1 it has.
    // faces_with_nodei; 
    // find out the closest nodes around vertex_i, should be 6 or 5 or 7, by checking how many -1 it has.
    Mat<int> closest_nodes = vertex_valence(faces_with_nodei, face); 
    
    // find out the ring_vertices around each face, should be 12 or 11. if all are -1, this face is deleted for insertion-setup.
    Mat<int> one_ring_nodes = one_ring_vertices(face, vertex3.outlayer, closest_nodes);

    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    // setup the target area and volume 
    cout<<"11. To determine the target area and volume."<<endl;
    double S0out, S0mid, S0in;
    double V0out, V0mid, V0in;
    ElementS03Layers elementS03;
    ElementV03Layers elementV03;
    if ( isBilayerModel == false ){
        elementS03.outlayer.resize(1, face.n_rows); elementS03.outlayer.fill(0.0);
        elementV03.outlayer.resize(1, face.n_rows); elementV03.outlayer.fill(0.0);
        cell_area_volume(face, vertex3.outlayer, one_ring_nodes, GaussQuadratureN, elementS03.outlayer, elementV03.outlayer, gqcoeff,shape_functions,subMatrix,subDivideTimes); // calculate the elemental area and volume
        S0out = sum(elementS03.outlayer);
        double Radiusout = sqrt(S0out/4.0/M_PI);;   
        V0out  = 4.0/3.0*M_PI*pow(Radiusout,3.0) * miu;
        S0out = S0out * ci + s_insert * insertionPatchNum;
    }else{
        elementS03.midlayer.resize(1, face.n_rows); elementS03.midlayer.fill(0); 
        elementV03.midlayer.resize(1, face.n_rows); elementV03.midlayer.fill(0);   
        cell_area_volume(face, vertex3.midlayer, one_ring_nodes, GaussQuadratureN, elementS03.midlayer, elementV03.midlayer, gqcoeff,shape_functions,subMatrix,subDivideTimes); // calculate the elemental area and volume
        S0mid = sum(elementS03.midlayer);
        double Radiusmid = sqrt(S0mid/4.0/M_PI);
        // set the equilibrium area for out-layer and inner-layer, considering the flip-flop
        double thickness = thickness_in + thickness_out;
        double flipflopRatio = calculate_flipflopRatio(isFlatMembrane, S0mid, thickness_in, c0in, kc_in, us_in, thickness_out, c0out, kc_out, us_out);
        cout<<"    The flip-flop ratio is: "<<flipflopRatio<<endl;
        elementS03.inlayer = elementS03.midlayer * (1.0 - flipflopRatio);
        elementS03.outlayer = elementS03.midlayer * (1.0 + flipflopRatio);
        S0in = sum(elementS03.inlayer);
        S0out = sum(elementS03.outlayer);
        S0out = S0out + s_insert * insertionPatchNum;

        double Radiusin = Radiusmid - thickness_in;   
        V0in  = 4.0/3.0 * M_PI * pow(Radiusin,3.0) * miu;
        double Radiusout = Radiusmid + thickness_out;   
        V0out  = 4.0/3.0 * M_PI * pow(Radiusout,3.0) * miu;
   }
    ////////////////////////////////////////////////////////
    // generate the parameters
    cout<<"12. To set up the structure Parameter."<<endl;
    Param param;  
        param.isFlatMembrane = isFlatMembrane;
        // mesh parameters:
        param.meanL = meanL;
        param.numface = face.n_rows;                          // face number on each mesh layer
        param.numvertex = vertex3.outlayer.n_rows;                    // vertex number on each mesh layer
        param.isBoundaryFixed = isBoundaryFixed;           // boundary conditions
        param.isBoundaryPeriodic = isBoundaryPeriodic;
        param.isBoundaryFree = isBoundaryFree;
        param.isBoundaryVertex = isBoundaryNode;           // labels of boundaries
        param.isBoundaryFace = isBoundaryFace;
        param.isGhostVertex = isGhostNode;                 // lables of ghost boundaries. The ghosts must be boundaries. 
        param.isGhostFace = isGhostFace;
        param.ghostPartner = ghostPartner;                 
        param.freePartners = freePartners;
        // mesh regularization
        param.usingNCG = true;
        param.isNCGstucked = false;
        param.usingRpi = usingRpi_viscousRegul;
        param.k_regularization = k_regularization;            // coefficient of the regulerization constraint, 
        param.gama_shape = gama_shape;                        // factor of shape deformation
        param.gama_area = gama_area;                          // factor of size deformation
        param.subDivideTimes = subDivideTimes;                // subdivision times for the irregular patches
        param.GaussQuadratureN = GaussQuadratureN;            // Gaussian quadrature integral 

        // membrane parameters: 
        param.isBilayerModel = isBilayerModel;
        param.isGlobalAreaConstraint = isGlobalAreaConstraint;// whether to use Global constraints for the area elasticity
        param.includeDivTilt = includeDivTilt;
        // out-layer (top layer)
        param.kc_out = kc_out;                                // pN.nm. bending modulus, out-monolayer 
        param.kst_out = kst_out;                              // pN.nm . splay-tilt modulus, out-monolayer
        param.us_out = us_out;                                // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
        param.Ktilt_out = Ktilt_out;                          // pN/nm = mN/m. tilt modulus
        param.Kthick_out = Kthick_out;                        // pN/nm, membrane thickness modulus 
        param.thickness_out = thickness_out;                  // nm, out-monolayer thickness. 
        param.H0C_out = H0C_out;                              // nm, out-monolayer target height or curvature-modified height. 
        param.c0out = c0out;                                  // spontaneous curvature of membrane, out-layer. Convex is positive
        param.S0out = S0out;                                  //target area 
        param.V0out = V0out;
        // in-layer (bottom layer)
        param.kc_in = kc_in;                                  // pN.nm. bending modulus, in-monolayer 
        param.kst_in = kst_in;                                // pN.nm . splay-tilt modulus, in-monolayer
        param.us_in = us_in;                                  // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
        param.Ktilt_in = Ktilt_in;                            // pN/nm = mN/m. tilt modulus
        param.Kthick_in = Kthick_in;                          // pN/nm, membrane thickness modulus 
        param.thickness_in = thickness_in;                    // nm, in-monolayer thickness. 
        param.H0C_in = H0C_in;                                // nm, in-monolayer target height or curvature-modified height. 
        param.c0in = c0in;                                    // spontaneous curvature of membrane, in-layer. Concave is positive
        param.S0in = S0in;                                    //target area
        param.V0in = V0in;
        param.uv = uv;
        
        param.Kthick_constraint = Kthick_out;
        
        // insertion parameters:
        param.insertionpatch = insertionpatch;
        param.isInsertionPatch = isInsertionPatch;            // labels of insertion patches. 
        param.c0out_ins = c0out_ins;                          // spontaneous curvature of insertion, outer layer
        param.s_insert  = s_insert;                           // insertion area
        param.insert_dH0 = insert_dH0;                        // equilibrium value of thickness decrease induced by the insertion, nm
        
        param.K_insertShape = K_insertShape;                  // spring constant for insertion zones, to constraint the insertion shape
        param.insertionShapeEdgeLength = determine_finestEdgeLength(face, vertex3, isBilayerModel, localFinerMesh);  
        param.IsinertionpatchAdjacent = IsinertionpatchAdjacent;
        //param.K_adjacentPatch = 0.0;                          // constant for the shape constraint on the patches adjacent around insertion 
        //param.Kthick_insertion = Kthick_insertion_initial;

        // system setup
        param.isLocallyFinerFace = determine_isLocallyFinerFace(face, localFinerMesh);

    
    //////////////////////////////////////////////////////
    // read a structure file
    cout<<"    To read the initial structures."<<endl;
    char name1[32] = "vertexRead.csv";
    read_struture_vertex(vertex3.outlayer, name1);
    /*
    char name1[32] = "vertexin_R15.csv";
    read_struture_vertex(vertexin, name1);
    char name2[32] = "vertexmid_R15.csv";
    read_struture_vertex(vertexmid, name2);
    char name3[32] = "vertexout_R15.csv";
    read_struture_vertex(vertexout, name3);
    */
    
    ///////////////////////////////////////////////////////////
    int numvertex = vertex3.outlayer.n_rows; 
    int numvertextotal = 3 * numvertex;
    int numface = face.n_rows;
    int numfacetotal = 3 * numface;
    Row<double> energy(7); energy.fill(0.0); // E_bending, E_constraint, E_memthick, E_regularization, E_insert, E_tot
    Force3Layers force3;
    force3.outlayer.resize(numvertex,3); force3.outlayer.fill(0);
    if ( isBilayerModel == true ){
        force3.midlayer.resize(numvertex,3); force3.midlayer.fill(0);
        force3.inlayer.resize(numvertex,3); force3.inlayer.fill(0);
    }
    Vertex3Layers vertex3ref;
    vertex3ref.outlayer = vertex3.outlayer;
    if ( isBilayerModel == true ){
        vertex3ref.midlayer = vertex3.midlayer;
        vertex3ref.inlayer = vertex3.inlayer;
    }
    rowvec deformnumbers(3);

    cout<<"13. To set the spontaneous curvature value on each triangle."<<endl;
    SpontCurv3Layers spontCurv3;
    int whichLayer = 2; // 0 means inner layer, 1 means middle layer, 2 means outer layer
    spontCurv3.outlayer = determine_spontaneous_curvature(whichLayer, param, face, vertex3.outlayer);
    printout_spontaneouscurvature(whichLayer, spontCurv3.outlayer);  
    if ( isBilayerModel == true ){
        whichLayer = 0; // 0 means inner layer, 1 means middle layer, 2 means outer layer
        spontCurv3.inlayer = determine_spontaneous_curvature(whichLayer, param, face, vertex3.inlayer);
        printout_spontaneouscurvature(whichLayer, spontCurv3.inlayer);
    }
    
    /*
    check_nodal_force(face, vertex3, vertex3ref, one_ring_nodes, 
                      param, elementS03, spontCurv3, deformnumbers, gqcoeff, shape_functions, subMatrix);
    exit(0);
    */
    
    cout<<"14. The first run of 'Energy_and_Force', to generate the initial nodal force."<<endl;
    Energy_and_Force(face, vertex3, vertex3ref, one_ring_nodes,
                     param, elementS03, spontCurv3, energy, 
                     force3, deformnumbers, gqcoeff, shape_functions, subMatrix);

    mat Energy(N,7); Energy.fill(0); Energy.row(0) = energy;
    vec forcescale = force3_scale(force3, isBilayerModel);
    vec MeanForce(N); MeanForce.fill(0); MeanForce(0) = mean(forcescale);
    vec MaxForce(N); MaxForce.fill(0); MaxForce(0) = max(forcescale);
    vec totalarea(N); totalarea(0) = param.Sout;
    vec totalvolume(N); totalvolume(0) = param.Vout;

    Force3Layers force30 = force3; 
    Force3Layers force31 = force3; 
    Vertex3Layers vertex30 = vertex3;
    Vertex3Layers vertex31 = vertex3; 
    
    Force3Layers s0 = force30;
    Force3Layers s1 = s0;

    rowvec energy0 = energy;
    rowvec energy1(7);
    double a0;
    double a1;
 
    bool isMinimized = false;
    bool isCriteriaSatisfied = false; 
    bool updateReference = true;
    int i = 0;
    Row<int> timesOffNCG(N); timesOffNCG.fill(0);

    cout<<"15. Begin the while-loop to minimize the system energy..."<<endl;
    cout<<"........................................................."<<endl;
    cout<<"........................................................."<<endl;

    while ( isCriteriaSatisfied == false && i < N-1){
        // updates
        param.currentStep = i; 
        if ( i == 0 || i % 50 == 0 ){
           a0 = finestL/max(force3_scale(s0, isBilayerModel)); 
           if ( max(force3_scale(s0, isBilayerModel)) < 1.0e-9 ) a0 = 0.1;
           //if ( a0 > 1.0) a0 = 1.0; 
        }else{
           a0 = a0 * 1.0e+1;
        } 
        {
            a1 = line_search_a_to_minimize_energy(a0, s0, face, vertex30, vertex3ref, one_ring_nodes, 
                                                 param, elementS03, spontCurv3, energy0, 
                                                 force30, gqcoeff, shape_functions, subMatrix);
            vertex31.outlayer = update_vertex(vertex30.outlayer, face, a1, s0.outlayer, param); 
            vertex31.midlayer = update_vertex(vertex30.midlayer, face, a1, s0.midlayer, param);
            vertex31.inlayer  = update_vertex(vertex30.inlayer, face, a1, s0.inlayer, param);
            // adjust vertex positions, constraint by the tube
            // constraintVertexWithinBoundaries(face, vertex31, param);
           
            if ( a1 == -1 ){
               cout<<"step: "<<i<<". Note: no efficent step size a is found. Stop now! "<<endl;
               printoutREF(vertex3ref.outlayer);
               printoutstuck(vertex30.outlayer);

               cout<<"Check whether the nodal force is correctly calculated: "<<endl;
               check_nodal_force(face, vertex30, vertex3ref, one_ring_nodes, param, elementS03, spontCurv3, deformnumbers, gqcoeff, shape_functions, subMatrix);

               break;
            }

            {// gradually increase the coefficient of height-constraint
                if ( i <= stepsToIncreaseKthickConstraint ){
                    if ( i % 10 == 0 && i > 0){
                        param.Kthick_constraint = Kthick_out + (Kthick_constraint - Kthick_out)/stepsToIncreaseKthickConstraint * i;
                    }
                } 
            }

           // calculate the new force and energy
            Energy_and_Force(face, vertex31, vertex3ref, one_ring_nodes, 
                            param, elementS03, spontCurv3, energy1, 
                            force31, deformnumbers, gqcoeff, shape_functions, subMatrix);
            // calculate the direction s
            Force3Layers ftemp1 = force31;
            //if ( param.isNCGstucked == false )
            {
                Force3Layers ftemp0 = force30;
                mat shu0 =  changeForce3ToVector(ftemp0, isBilayerModel) * strans(changeForce3ToVector(ftemp0, isBilayerModel));
                mat shu1 =  changeForce3ToVector(ftemp1, isBilayerModel) * strans(changeForce3ToVector(ftemp1, isBilayerModel));
                mat shu10 =  changeForce3ToVector(ftemp1, isBilayerModel) * strans(changeForce3ToVector(ftemp0, isBilayerModel)); 
                double peta1 = shu1(0,0) / shu0(0,0); 
                //if ( param.duringStepsToIncreaseInsertDepth == true ) peta1 = 0.0;
                s1.outlayer = ftemp1.outlayer + peta1 * s0.outlayer;
                if ( isBilayerModel == true ){
                    s1.inlayer = ftemp1.inlayer + peta1 * s0.inlayer;
                    s1.midlayer = ftemp1.midlayer + peta1 * s0.midlayer;
                }
            }

           // update the parameters
           vertex30 = vertex31;
           force30 = force31;
           s0 = s1;
           a0 = a1;
           energy0 = energy1;
           
           i = i + 1;
           // store the energy and nodal force
           Energy.row(i) = energy1;
           vec forcescale = force3_scale(force31, isBilayerModel);
           MeanForce(i) = mean(forcescale);
           MaxForce(i) = max(forcescale); 
           totalarea(i) = param.Sout;
           totalvolume(i) = param.Vout;
           if (updateReference == true){
               //if ( abs(Energy(i,6)-Energy(i-1,6)) < 1e-3 || abs(MeanForce(i)-MeanForce(i-1)) < 1e-3 ){ 
               if ( abs(Energy(i,6)-Energy(i-1,6)) < 1e-3 || abs(MaxForce(i)-MaxForce(i-1)) < 1e-3 ){
                   cout<<"update the reference structure! "<<endl;
                   vertex3ref = vertex31;
               }
           }
       }
       
       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       // output parameters     
       cout<<"step: "<< i <<". Sratio= "<<totalarea(i)/S0out<<", Vratio= "<<totalvolume(i)/V0out<<". Energy= "<<Energy(i,6)<<". meanF= "<<MeanForce(i)<<". maxF= "<<MaxForce(i)<<". a= "<<a0<<endl;
       //cout<<"step: "<< i <<". Deform number: Area = "<<deformnumbers(0)<<", Shape = "<<deformnumbers(1)<<", nodeform = "<<deformnumbers(2)<<endl;
       //cout<<"step: "<< i <<". Kthick_constraint = "<< param.Kthick_constraint << endl;

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        // check whether to stop. if the total energy is flat, then stop
        // check whether to stop. if the total energy is flat for 100 simulation steps, then stop
        if ( insertionPatchNum < 1 ){
            //if ( MaxForce(i) < criterion_force ){
            if ( MeanForce(i) < criterion_force ){
                cout<<"The energy is minimized. End of the while loop!"<<endl;
                isCriteriaSatisfied = true;
                break;
            }
        }else{
            if ( i > stepsToIncreaseKthickConstraint ){
                //if ( MaxForce(i) < criterion_force ){
                if ( MeanForce(i) < criterion_force ){
                    cout<<"The energy is minimized. End of the while loop!"<<endl;
                    isCriteriaSatisfied = true;
                    break;
                }
            }
        } // end of check whether to stop

       double Ere_vs_Etot = (Energy(i,5)+0.0)/(Energy(i,6)+0.0);
       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       // output the newvertex
        if ( i%100 == 0 ){    
            int kk = i/100;
            printout_temporary_structures(kk, vertex31, isBilayerModel);
        }

        // output the energy and nodal force 
        ofstream outfile2("energy_force.csv"); 
        outfile2 << "E_bending" << ',' << "E_insert" << ',' << "E_thick" << ',' << "E_splayTilt" << ',' << "E_constraint" << ',' << "E_regularization" << ',' << "E_total" << ',' << "E_re/E_tot" << ',' << "S/S0" << ',' << "V/V0" << ',' << "meanF" << ',' << "maxF" << '\n';
        for (int j = 0; j <= i; j++) {
            Ere_vs_Etot = (Energy(j,5)+0.0)/(Energy(j,6)+0.0);
            outfile2 << Energy(j,0)+0.0 << ',' << Energy(j,1)+0.0 << ',' << Energy(j,2)+0.0 << ',' << Energy(j,3)+0.0 << ',' << Energy(j,4)+0.0 << ',' << Energy(j,5)+0.0 << ',' << Energy(j,6)+0.0 << ',' << Ere_vs_Etot << ',' << totalarea(j)/S0in << ',' << totalvolume(j)/V0in << ',' << MeanForce(j) << ',' << MaxForce(j) << '\n';
        }
        outfile2.close();

    } // end of while loop: energy minimization
    cout<<"........................................................."<<endl;
    cout<<"........................................................."<<endl;

    // output the final structure
    cout<<"16. To output the final structure: vertex_final_*."<<endl;
    printout_final_structures(vertex30, isBilayerModel);

    // output the final thickness
    cout<<"17. To output the final height of each leaflet: height_*.csv"<<endl;
    // first column is the target height; second column is the observed height
    if ( isBilayerModel == true ){
        whichLayer = 0;
        mat heightin = calculate_thickness(whichLayer, face, vertex30.inlayer, vertex30.midlayer, one_ring_nodes, param, shape_functions, subMatrix);
        whichLayer = 2;
        mat heightout = calculate_thickness(whichLayer, face, vertex30.midlayer, vertex30.outlayer, one_ring_nodes, param, shape_functions, subMatrix);
        printout_thickness(heightin, heightout);
    }else{
        cout<<"    No need to output the heights, because single-layer mesh model is used. "<<endl;
    }
    // output the final shape; triangle centers on the limit surface
    cout<<"18. To output the final positions of each element center: triangleCenters*.csv"<<endl;
    mat centersout = calculate_triangleCenter(face, vertex30.outlayer, one_ring_nodes, param, shape_functions, subMatrix);
    mat centersin, centersmid;
    if ( isBilayerModel == true ){
        centersin = calculate_triangleCenter(face, vertex30.inlayer, one_ring_nodes, param, shape_functions, subMatrix);
        centersmid = calculate_triangleCenter(face, vertex30.midlayer, one_ring_nodes, param, shape_functions, subMatrix);
    }
    printout_elementcCenter_onLimitSurface(centersin, centersmid, centersout, isBilayerModel);
}
