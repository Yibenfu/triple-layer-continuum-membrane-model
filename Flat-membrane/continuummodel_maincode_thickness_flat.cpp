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
double l   = 2.07457 / 2;                        // triangular side length, nm
double sideX  = 40.0;                            // rectangle sidelength x, nm
double sideY  = sideX;                           // rectangle sidelength y, nm
double radiusForLocalFiner = 6.0;                // the size of local finer mesh
bool   isBoundaryFixed = false;                  // boundaries
bool   isBoundaryPeriodic  = true;
bool   isBoundaryFree = false;
double k_regularization   = (1.0e1)*10.0*4.17;   // coefficient of the regulerization constraint, 
double gama_shape = 0.2;                         // factor of shape deformation
double gama_area = 0.2;                          // factor of size deformation
int    subDivideTimes = 5;                       // subdivision times for the irregular patches
int    GaussQuadratureN = 2;                     // Gaussian quadrature integral 

// membrane parameters: 
bool   isBilayerModel = true;
bool   isGlobalAreaConstraint = true;            // whether to use Global constraints for the area elasticity
// out-layer (top layer)
double kc_out  = 19.4*4.17/2.0;                  // pN.nm. bending modulus, out-monolayer 
double us_out  = 265.0/2.0;                      // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
double Ktilt_out  = 89.0;                        // pN/nm = mN/m. tilt modulus
double Kthick_out = Ktilt_out + 2.0*us_out;      // pN/nm, coefficient of the membrane thickness. penalty term
double thickness_out = 2.71/2.0;                 // nm, out-monolayer thickness. 
double c0out = -0.04;                            // spontaneous curvature of membrane, out-layer. Convex is positive
// in-layer (bottom layer)
double kc_in  = kc_out;                          // pN.nm. bending modulus, in-monolayer 
double us_in  = us_out;                          // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
double Ktilt_in  = Ktilt_out;                    // pN/nm = mN/m. tilt modulus
double Kthick_in = Ktilt_in + 2.0*us_in;
double thickness_in = thickness_out;             // nm, in-monolayer thickness. 
double c0in  = c0out;                            // spontaneous curvature of membrane, in-layer. Concave is positive
// external forces: tension
double tension = 0.0;                            // pN/nm. 0.01~10
double ci  = us_out/(tension + us_out);          // area constraint target
double miu = 1.0;                                // volume constraint target

// insertion parameters:
int    insertionPatchNum = 1;                    // number of insertions
double distBetwn = 0.0;                          // distance between the insertions
double c0out_ins  = 0.3;                         // spontaneous curvature of insertion, outer layer
double s_insert  = 2.0;                      // insertion area
double insertLength = 2.0;                       // insertion size
double insertWidth = 1.0; 
double insert_dH0 = 0.3;                         // equilibrium value of thickness decrease induced by the insertion, nm
double insert_dH0_initial = 0.1;                 // insertion depth, nm
int    stepsToIncreaseInsertDepth = 100;         // gradually increase the thickness decrease value. after this step, coefficient will be full value
double K_insertShape   = 10.0*us_out;            // spring constant for insertion zones, to constraint the insertion shape
int    neighReguleSteps = 100;                   // every n steps to impose regularizations on the adjacent triangules around the insertion

// parameters for simulation setup
int    N   = 1e5;                                // total step of iteration
double criterion_force = 0.009; //1.0e-2;                 // critera for the equilibrium of the simulation
double criterion_E = 2.0e-5; 
double criterion_S = 1e-5;
double criterion_V = 1e-5;
double criterion_Er = 1e-5;

/////////////////////////////////////////////////////////////////////////////////
// main code
int main() {
    srand((unsigned)time(NULL)); 
    // build the triangular mesh plane. NOTE: ghost vertices and faces are included. All the boundary faces are ghost faces, which should be eliminated when output faces
    cout<<"1. To build a spherical mesh with Loop's subdivision scheme."<<endl;
    Mat<double> vertex = setvertex_Loop_scheme(sideX, sideY, l); // vertex position
    Mat<int> face = setface_Loop_scheme(sideX, sideY, l); // face and its surrounding vertex
    
    cout<<"   Flat membrane, Xside = "<< max(vertex.col(0)) - min(vertex.col(0)) <<" nm, Yside = "<< max(vertex.col(1)) - min(vertex.col(1)) <<endl;
    
    // boundary vertex or face is located on the most edge
    Row<int> isBoundaryNode = determine_BoundaryVertex(sideX, sideY, l); // element 1 means this vertex is on boundary
    Row<int> isBoundaryFace = determine_Boundaryface(face,isBoundaryNode); // element 1 means this face is on boundary
    Row<int> isGhostNode = determine_GhostVertex(sideX, sideY, l, isBoundaryFixed, isBoundaryPeriodic, isBoundaryFree); // element 1 means this vertex is ghost
    Row<int> isGhostFace = determine_GhostFace(sideX, sideY, l, isBoundaryFixed, isBoundaryPeriodic, isBoundaryFree); // element 1 means this face is ghost   
    
    Mat<int> faces_with_nodei = faces_with_vertexi(vertex, face);
    
    vector<int> ghostPartner = determine_GhostPartner(sideX, sideY, l);
    vector<vector<int>> freePartners = determine_freePartners(sideX, sideY, l);

    ///////////////////////////////////////////////////////
    // Finer the mesh aroud the center of the membrane if the membrane is flat.
    // Meanwhile, the vertex, face need to be updated!
    cout<<"2. To make the local mesh finer."<<endl;
    LocalFinerMesh localFinerMesh;
    double finestL = l;
    build_local_finer_mesh(vertex, face, finestL, radiusForLocalFiner, faces_with_nodei, localFinerMesh, isBoundaryNode, isBoundaryFace, isGhostNode, isGhostFace, ghostPartner, freePartners);
    if ( localFinerMesh.buildLocalFinerMesh == true ){
        Row<int> finerFaces = localFinerMesh.faceIndex + 1; finerFaces.print("   finer_Faces: ");
    }

    //////////////////////////////////////////////////////
    // set the inner layer and out layer
    cout<<"3. To build the multi-layer meshes, as the model of bilayer membrane."<<endl;
    mat vertexout(vertex.n_rows, 3); vertexout = vertex;
    // thickness = thickness * ci; // tension will change the membrane thickness, according to fixed volume 
    mat vertexin = setup_inner_mesh(vertexout,thickness_out+thickness_in); 
    mat vertexmid = setup_inner_mesh(vertexout,thickness_out); 
    
    /*
    //////////////////////////////////////////////////////
    // read a structure file
    cout<<"   Read in the vertex files"<<endl;
    char name1[32] = "vertex_final_in.csv";
    read_struture_vertex(vertexin, name1);
    char name2[32] = "vertex_final_mid.csv";
    read_struture_vertex(vertexmid, name2);
    char name3[32] = "vertex_final_out.csv";
    read_struture_vertex(vertexout, name3);
    */

    cout<<"4. To output the intial structure: vertex_begin_* and face*."<<endl;
    // output the vertex and face matrix
    ofstream outfile("face.csv");
    for (int i = 0; i < face.n_rows; i++) {
        outfile << face(i,0)+1 << ',' << face(i,1)+1 << ',' << face(i,2)+1 << '\n';
    }
    outfile.close();
    ofstream outfile1("vertex_begin_in.csv");
    for (int i = 0; i < vertexin.n_rows; i++) {
        outfile1 << setprecision(16) << vertexin(i,0)+0.0 << ',' << vertexin(i,1)+0.0 << ',' << vertexin(i,2)+0.0 << '\n';
    }
    outfile1.close();
     ofstream outfile11("vertex_begin_mid.csv");
    for (int i = 0; i < vertexmid.n_rows; i++) {
        outfile11 << setprecision(16) << vertexmid(i,0)+0.0 << ',' << vertexmid(i,1)+0.0 << ',' << vertexmid(i,2)+0.0 << '\n';
    }
    outfile11.close();
    ofstream outfile2("vertex_begin_out.csv");
    for (int i = 0; i < vertexout.n_rows; i++) {
        outfile2 << setprecision(16) << vertexout(i,0)+0.0 << ',' << vertexout(i,1)+0.0 << ',' << vertexout(i,2)+0.0 << '\n';
    }
    outfile2.close();

    //////////////////////////////////////////////////////
    cout<<"5. To select insertionPatch, several triangles on the outer layer mesh as the helix insertion zone."<<endl;
    // split the insertionVertex line, and insert several triangles as insertionPatch
    // each insertion patch has ellipse shape with long and short radius: a = 1.25 nm, b = 0.5093 nm, area=pi*a*b= 2.0 nm2
    // Mat<int> insertionpatch = adjust_vertex_dueTo_insertion(vertexout, faceout, faces_with_nodei, insertionVertex, l);
    
    // find insertions
    Mat<int> insertionpatch = select_insertionPatch(face, vertexout, faces_with_nodei, localFinerMesh, finestL, insertLength, insertWidth, insertionPatchNum, distBetwn);
    
    Mat<int> InsertionPatchPrint = insertionpatch + 1;
    InsertionPatchPrint.print("InsertionPatch: ");
    
    Row<int> Isinsertionpatchin(face.n_rows); Isinsertionpatchin.fill(-1);
    Row<int> Isinsertionpatchout(face.n_rows); Isinsertionpatchout.fill(-1);
    for ( int i = 0; i <  insertionpatch.n_rows; i++){
        for ( int j = 0; j < insertionpatch.n_cols; j++ ){
            Isinsertionpatchout( insertionpatch(i,j) ) = 1;
        }
    }
    int zonenumber = insertionpatch.n_rows; // the number of insertion zones.
    
    cout<<"6. To select the zone around the insertionPatch. This neighbor zone will intermittently have shape constraints."<<endl;
    Row<int> insertionpatchAdjacent = select_insertionPatchAdjacent(face, vertexout, faces_with_nodei, insertionpatch);
    Row<int> insertionpatchAdjacentPrint = insertionpatchAdjacent + 1;
    insertionpatchAdjacentPrint.print("InsertionPatchAdjacent: ");
    Row<int> IsinertionpatchAdjacent(face.n_rows); IsinertionpatchAdjacent.fill(-1);
    for (int i = 0; i < insertionpatchAdjacent.n_cols; i++){
        int facenum = insertionpatchAdjacent(i);
        IsinertionpatchAdjacent(facenum) = 1;
    }
    
    //////////////////////////////////////////////////////
    // update the vertex nearby and one-ring-vertex
    cout<<"7. To find the one-ring-vertex for each triangle."<<endl;
    // find out what faces that have vertex_i,6 or 5 or 7, by checking how many -1 it has.
    // faces_with_nodei; // faces_with_vertexi(vertexin, facein);

    // find out the closest nodes around vertex_i, should be 6 or 5 or 7, by checking how many -1 it has.
    Mat<int> closest_nodes = vertex_valence(faces_with_nodei, face);

    // find out the ring_vertices around each face, should be 12 or 11. if all are -1, this face is deleted for insertion-setup.
    Mat<int> one_ring_nodes = one_ring_vertices(face, vertexout, closest_nodes, isBoundaryFace);

    //////////////////////////////////////////////////////////
    cout<<"8. To build the shape functions for the differential geometry on different types of patches."<<endl;
    // gauss_quadrature and shape functions
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

    //////////////////////////////////////////////////////////
    // setup the target area 
    rowvec elementS0in(face.n_rows); elementS0in.fill(0.0);
    plane_area(vertexin,face,isBoundaryFace,one_ring_nodes,Isinsertionpatchin,GaussQuadratureN,elementS0in,gqcoeff,shape_functions,subMatrix,subDivideTimes); // calculate the elemental area and volume
    elementS0in = elementS0in * ci; 
    double S0in = sum(elementS0in); // total area
    rowvec elementS0out(face.n_rows); elementS0out.fill(0.0);
    plane_area(vertexout,face,isBoundaryFace,one_ring_nodes,Isinsertionpatchout,GaussQuadratureN,elementS0out,gqcoeff,shape_functions,subMatrix,subDivideTimes);
    elementS0out = elementS0out * ci;  

    rowvec insertionpatchArea = get_insertionpatchArea(insertionpatch,elementS0out);  
    double S0out = sum(elementS0out) + sum(insertionpatchArea); //sum(elementS0out); // total area


    ////////////////////////////////////////////////////////
    // generate the parameters
    cout<<"9. To set up the structure Parameter."<<endl;
    Param param;  
        // mesh parameters:
        param.l = l;                                       // triangular side length, nm
        param.sideX = sideX;                                   // rectangle sidelength x, nm
        param.sideY = sideY;                                   // rectangle sidelength y, nm
        param.numface = face.n_rows; 
        param.numvertex = vertexin.n_rows;
        param.isBoundaryFixed = isBoundaryFixed;                 // boundaries
        param.isBoundaryPeriodic = isBoundaryPeriodic;
        param.isBoundaryFree = isBoundaryFree;
        param.isBoundaryVertex = isBoundaryNode;
        param.isBoundaryFace = isBoundaryFace;
        param.isGhostVertex = isGhostNode;
        param.isGhostFace = isGhostFace;
        param.ghostPartner = ghostPartner;
        param.freePartners = freePartners;
        // mesh regularization
        param.isNCGstucked = false;
        param.usingNCG = true;
        param.usingRpi = true;
        param.k_regularization = k_regularization;                         // coefficient of the regulerization constraint, 
        param.gama_shape = gama_shape;                         // factor of shape deformation
        param.gama_area = gama_area;                          // factor of size deformation
        param.subDivideTimes = subDivideTimes;                        // subdivision times for the irregular patches
        param.GaussQuadratureN = GaussQuadratureN;                     // Gaussian quadrature integral 

        // membrane parameters: 
        param.isBilayerModel = isBilayerModel;
        param.isGlobalAreaConstraint = isGlobalAreaConstraint;            // whether to use Global constraints for the area elasticity
        // out-layer (top layer)
        param.kc_out = kc_out;                  // pN.nm. bending modulus, out-monolayer 
        param.us_out = us_out;                       // pN/nm, area stretching modulus, out-monolayer; 0.5*us*(ds)^2/s0;
        param.Ktilt_out = Ktilt_out;                         // pN/nm = mN/m. tilt modulus
        param.Kthick_out = Kthick_out;       // pN/nm, coefficient of the membrane thickness. penalty term
        param.thickness_out = thickness_out;                   // nm, out-monolayer thickness. 
        param.c0out = c0out;                              // spontaneous curvature of membrane, out-layer. Convex is positive
        param.S0out = S0out;                                     //target area 
        // in-layer (bottom layer)
        param.kc_in = kc_in;                           // pN.nm. bending modulus, in-monolayer 
        param.us_in = us_in;                          // pN/nm, area stretching modulus, in-monolayer; 0.5*us*(ds)^2/s0;
        param.Ktilt_in = Ktilt_in;                    // pN/nm = mN/m. tilt modulus
        param.Kthick_in = Kthick_in; 
        param.thickness_in = thickness_in;             // nm, in-monolayer thickness. 
        param.c0in = c0in;                            // spontaneous curvature of membrane, in-layer. Concave is positive
        param.S0in = S0in;                                      //target area

        // insertion parameters:
        param.insertionpatch = insertionpatch;
        param.c0out_ins = c0out_ins;                          // spontaneous curvature of insertion, outer layer
        param.s_insert  = s_insert;                          // insertion area
        param.insert_dH0 = insert_dH0_initial;                         // equilibrium value of thickness decrease induced by the insertion, nm
        param.K_insertShape = K_insertShape;            // spring constant for insertion zones, to constraint the insertion shape
        param.insertionShapeEdgeLength = finestL;  
        param.IsinertionpatchAdjacent = IsinertionpatchAdjacent;
        param.K_adjacentPatch = 0.0;                         // constant for the shape constraint on the patches adjacent around insertion 

    ///////////////////////////////////////////////////////////
    int numvertex = vertexout.n_rows;
    int numvertextotal = 3 * numvertex;
    Row<double> energy(7); energy.fill(0.0); // E_bending, E_constraint, E_memthick, E_regularization, E_insert, E_tot
    mat forcein(numvertex,3); forcein.fill(0);
    mat forceout(numvertex,3); forceout.fill(0);
    mat forcemid(numvertex,3); forcemid.fill(0);
    mat vertexinref = vertexin; 
    mat vertexoutref = vertexout; 
    mat vertexmidref = vertexmid; 
    rowvec deformnumbers(3);
    
    cout<<"10. To set the spontaneous curvature value on each triangle."<<endl;
    bool isInnerLayer = false;
    rowvec spontcurv_out = determine_spontaneous_curvature(isInnerLayer, param, face, vertexout);
    printout_spontaneouscurvature(isInnerLayer, spontcurv_out);  
    isInnerLayer = true;
    rowvec spontcurv_in = determine_spontaneous_curvature(isInnerLayer, param, face, vertexin);
    printout_spontaneouscurvature(isInnerLayer, spontcurv_in);  
    /*
    check_nodal_force(vertexin, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref, facein, facemid, faceout, one_ring_nodes_in, one_ring_nodes_mid, one_ring_nodes_out, 
                      param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, Isinsertionpatchin, Isinsertionpatchmid, Isinsertionpatchout, 
                      energy, forcein, forcemid, forceout, deformnumbers, gqcoeff, shape_functions, subMatrix);
    exit(0);
    */

    cout<<"11. The first run of 'Energy_and_Force', to generate the initial nodal force."<<endl;
    Energy_and_Force(face, vertexin, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref,
                     one_ring_nodes, param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, Isinsertionpatchin, Isinsertionpatchout, energy, 
                     forcein, forcemid, forceout,deformnumbers, gqcoeff, shape_functions, subMatrix);

    mat Energy(N,7); Energy.fill(0); Energy.row(0) = energy;
    vec forcescale = force_scale(forcein, forcemid, forceout);
    vec MeanForce(N); MeanForce.fill(0); MeanForce(0) = mean(forcescale);
    vec totalarea(N); totalarea(0) = param.Sin;

    mat forcein0 = forcein; mat forcein1(numvertex,3);
    mat forcemid0 = forcemid; mat forcemid1(numvertex,3);
    mat forceout0 = forceout; mat forceout1(numvertex,3);
    mat vertexin0 = vertexin; mat vertexin1(numvertex,3);
    mat vertexmid0 = vertexmid; mat vertexmid1(numvertex,3);
    mat vertexout0 = vertexout; mat vertexout1(numvertex,3);
    mat s0(numvertextotal,3); 
    s0.submat(span(0,numvertex-1),span(0,2)) = forcein0;
    s0.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid0; 
    s0.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout0;
    mat s1(numvertextotal,3);
    rowvec energy0 = energy;
    rowvec energy1(7);
    double a0;
    double a1;
 
    bool isMinimized = false;
    bool isCriteriaSatisfied = false; 
    bool updateReference = true;
    int i = 0;
    Row<int> timesOffNCG(N); timesOffNCG.fill(0);

    cout<<"12. Begin the while-loop to minimize the system energy..."<<endl;
    while ( isCriteriaSatisfied == false && i < N-1){
       // updates 
       if ( i == 0 || i%50 == 0 ){
           a0 = l/max(toscale(s0));
           //if ( a0 > 1.0) a0 = 1.0; 
       }else{
           a0 = a0 * 2e1;
       } 
       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       {
           a1 = line_search_a_to_minimize_energy(a0, s0, face, vertexin0, vertexinref, vertexmid0, vertexmidref, vertexout0, vertexoutref,
                                                 one_ring_nodes, param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, Isinsertionpatchin, Isinsertionpatchout, 
                                                 energy0, forcein0, forcemid0, forceout0, gqcoeff, shape_functions, subMatrix);
           mat s0in = s0.submat(span(0,numvertex-1),span(0,2));
           mat s0mid = s0.submat(span(numvertex,2*numvertex-1),span(0,2));
           mat s0out = s0.submat(span(2*numvertex,numvertextotal-1),span(0,2));
           vertexin1 = update_vertex(vertexin0, face, a1, s0in, param);
           vertexmid1 = update_vertex(vertexmid0, face, a1, s0mid, param);
           vertexout1 = update_vertex(vertexout0, face, a1, s0out, param);
           if ( a1 == -1 ){
               cout<<"step: "<<i<<". Note: no efficent step size a is found. Stop now! "<<endl;
               printoutREF(vertexinref);
               printoutstuck(vertexout0);
               
               cout<<"Check whether the nodal force is correctly calculated: "<<endl;

               check_nodal_force(face, vertexin0, vertexinref, vertexmid0, vertexmidref, vertexout0, vertexoutref, one_ring_nodes,
                      param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, Isinsertionpatchin, Isinsertionpatchout, 
                      energy0, forcein0, forcemid0, forceout0, deformnumbers, gqcoeff, shape_functions, subMatrix);

               break;
           }
           { // gradually increase the coefficient of insertion-shape-constraint
             // gradually increase the thickness decrease value
               // int stepstarget = 500; // after this step, coefficient will be full value
                if (insertionPatchNum > 0 && i <= stepsToIncreaseInsertDepth){
                    if ( i%10 == 0 && i > 0){
                        param.insert_dH0 = insert_dH0_initial + (insert_dH0 - insert_dH0_initial)/stepsToIncreaseInsertDepth * i;
                    }
                    param.duringStepsToIncreaseInsertDepth = true;
                }else{
                    param.duringStepsToIncreaseInsertDepth = false;
                } 
           }
           { // sometimes, add the shape constraint on the adjacent patches around the insertions
                if ( insertionPatchNum > 0 && i > stepsToIncreaseInsertDepth && pow(-1.0,floor((i-stepsToIncreaseInsertDepth)/neighReguleSteps)) > 0 ){
                    param.K_adjacentPatch = param.K_insertShape;
                }else{
                    param.K_adjacentPatch = 0.0;
                }
           }
           // calculate the new force and energy
           Energy_and_Force(face, vertexin1, vertexinref, vertexmid1, vertexmidref, vertexout1, vertexoutref,
                            one_ring_nodes, param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, Isinsertionpatchin, Isinsertionpatchout, 
                            energy1, forcein1, forcemid1, forceout1, deformnumbers, gqcoeff, shape_functions, subMatrix);
           // calculate the direction s
           mat ftemp1(numvertextotal,3); 
           ftemp1.submat(span(0,numvertex-1),span(0,2)) = forcein1; 
           ftemp1.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid1; 
           ftemp1.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout1;
           if ( param.isNCGstucked == false ){
               mat ftemp0(numvertextotal,3); 
               ftemp0.submat(span(0,numvertex-1),span(0,2)) = forcein0;
               ftemp0.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid0;
               ftemp0.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout0;     
               rowvec df0 = tovector(-ftemp0);
               rowvec df1 = tovector(-ftemp1);
               mat shu0 = df0 * strans(df0);
               mat shu1 = df1 * strans(df1); 
               mat shu10 = df1 * strans(df0);
               double peta1= shu1(0,0) / shu0(0,0); 
               s1 = ftemp1 + peta1 * s0;
           }else{
               s1 = ftemp1;
           }
           // update the parameters
           vertexin0 = vertexin1;
           vertexmid0 = vertexmid1;
           vertexout0 = vertexout1;
           forcein0 = forcein1;
           forcemid0 = forcemid1;
           forceout0 = forceout1;
           s0 = s1;
           a0 = a1;
           energy0 = energy1;
           
           i=i+1;
           // store the energy and nodal force
           Energy.row(i) = energy1;
           vec forcescale = force_scale(forcein1, forcemid1, forceout1);
           MeanForce(i) = mean(forcescale); 
           totalarea(i) = param.Sin;
           if (updateReference == true){
               if ( abs(Energy(i,6)-Energy(i-1,6)) < 1e-3 || abs(MeanForce(i)-MeanForce(i-1)) < 1e-3 ){
                   cout<<"update the reference structure! "<<endl;
                   vertexinref = vertexin1;
                   vertexmidref = vertexmid1;
                   vertexoutref = vertexout1;
               }
           }
           
           /*
           // sometimes,the NCG is too strong and slow down the simulations. so turn it off!
           if ( param.isNCGstucked == true ){
               timesOffNCG(i) = 1;
           }
           if ( i > 10 ){
                Mat<int> temp = sum(timesOffNCG.submat(0,i-9,0,i), 1);
                if ( temp(0,0) > 8 ){
                    param.usingNCG = false;
                }
           }
           */
       }

       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       // output parameters     
       cout<<"step: "<< i <<". Sratio= "<<totalarea(i)/S0in<<". Energy= "<<Energy(i,6)<<". meanF= "<<MeanForce(i)<<". a= "<<a0<<endl;
       cout<<"step: "<< i <<". Deform number: Area = "<<deformnumbers(0)<<", Shape = "<<deformnumbers(1)<<", nodeform = "<<deformnumbers(2)<<endl;


       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       // check whether to stop. if the total energy is flat, then stop
       if ( insertionPatchNum < 1 ){
           int checkSteps = 10;
           //if ( MeanForce(i) < criterion_force || (i > checkSteps && abs((Energy(i,6)-Energy(i-checkSteps,6))/checkSteps) < criterion_E) ){
           if ( MeanForce(i) < criterion_force ){
                cout<<"The energy is minimized. Stop now!"<<endl;
                isCriteriaSatisfied = true;
           }
       }else{
           // perform this check during the period when adjacent-regularization is off longer than steps of checkSteps.
           int checkSteps = floor(neighReguleSteps/2);
           bool adjacentRegularizeOff = false;
           if ( param.K_adjacentPatch < 1e-9 && pow(-1.0,floor((i-checkSteps-stepsToIncreaseInsertDepth)/neighReguleSteps)) < 0 ){
                adjacentRegularizeOff = true;
           }
           if ( i > stepsToIncreaseInsertDepth && adjacentRegularizeOff == true ){
                // two ways to judge whether minimized
                // #1: the energy curve is flat enough
                //if ( abs((Energy(i,6)-Energy(i-checkSteps,6))/checkSteps) < criterion_E || MeanForce(i) < criterion_force ){
                if ( MeanForce(i) < criterion_force ){
                    cout<<"The energy is minimized. Stop now!"<<endl;
                    isCriteriaSatisfied = true;
                }
                
                /*
                // #2, pseudo-minimized: if two adjacent-regularization-off zones give similar energy, then the simulation also reaches the E-minimization.
                bool adjacentRegularizeTwiceAtLeast = false;
                if ( i > stepsToIncreaseInsertDepth + neighReguleSteps*3 ){
                    adjacentRegularizeTwiceAtLeast = true;
                }
                if ( adjacentRegularizeTwiceAtLeast == true ){
                    if ( abs((Energy(i,6)-Energy(i-neighReguleSteps*2,6))/(neighReguleSteps*2)) < criterion_E || MeanForce(i) < criterion_force ){
                        cout<<"The energy is pseudo-minimized. Stop now!"<<endl;
                        isCriteriaSatisfied = true;
                        break;
                    }
                }
                */ 
           }
       }
       double Ere_vs_Etot = (Energy(i,5)+0.0)/(Energy(i,6)+0.0);
       ////////////////////////////////////////////////////////////////////////////
       ////////////////////////////////////////////////////////////////////////////
       // output the newvertex
        if ( i%100 == 0 ){    
            int kk = i/100;
            char filename[20] = "vertex%din.csv";
            sprintf(filename,"vertex%din.csv",kk);
            ofstream outfile(filename);
            for (int j = 0; j < vertexin1.n_rows; j++) {
                outfile << vertexin1(j,0)+0.0 << ',' << vertexin1(j,1)+0.0 << ',' << vertexin1(j,2)+0.0 << '\n';
            }
            outfile.close();

            char filename1[20] = "vertex%dout.csv";
            sprintf(filename1,"vertex%dout.csv",kk);
            ofstream outfile1(filename1);
            for (int j = 0; j < vertexout1.n_rows; j++) {
                outfile1 << vertexout1(j,0)+0.0 << ',' << vertexout1(j,1)+0.0 << ',' << vertexout1(j,2)+0.0 << '\n';
            }
            outfile1.close();

            char filename2[20] = "vertex%dmid.csv";
            sprintf(filename2,"vertex%dmid.csv",kk);
            ofstream outfile22(filename2);
            for (int j = 0; j < vertexmid1.n_rows; j++) {
                outfile22 << vertexmid1(j,0)+0.0 << ',' << vertexmid1(j,1)+0.0 << ',' << vertexmid1(j,2)+0.0 << '\n';
            }
            outfile22.close();
        }
        ofstream outfile2("energy_force.csv"); // E_bending, Einsert, Ethick, Etilt, E_const, E_regul, E_tot, Ere/Etot, s/s0, meanForce
        for (int j = 0; j <= i; j++) {
            Ere_vs_Etot = (Energy(j,5)+0.0)/(Energy(j,6)+0.0);
            outfile2 << Energy(j,0)+0.0 << ',' << Energy(j,1)+0.0 << ',' << Energy(j,2)+0.0 << ',' << Energy(j,3)+0.0 << ',' << Energy(j,4)+0.0 << ',' << Energy(j,5)+0.0 << ',' << Energy(j,6)+0.0 << ',' << Ere_vs_Etot << ',' << totalarea(j)/S0in << ',' << MeanForce(j) << '\n';
        }
        outfile2.close();
    }
    // output the final structure
    ofstream outfile3("vertex_final_in.csv");
    for (int j = 0; j < vertexin0.n_rows; j++) {
        outfile3 << setprecision(16) << vertexin0(j,0)+0.0 << ',' << vertexin0(j,1)+0.0 << ',' << vertexin0(j,2)+0.0 << '\n';
    }
    outfile3.close();
    ofstream outfile5("vertex_final_mid.csv");
    for (int j = 0; j < vertexmid0.n_rows; j++) {
        outfile5 << setprecision(16) << vertexmid0(j,0)+0.0 << ',' << vertexmid0(j,1)+0.0 << ',' << vertexmid0(j,2)+0.0 << '\n';
    }
    outfile5.close();
    ofstream outfile4("vertex_final_out.csv");
    for (int j = 0; j < vertexout0.n_rows; j++) {
        outfile4 << setprecision(16) << vertexout0(j,0)+0.0 << ',' << vertexout0(j,1)+0.0 << ',' << vertexout0(j,2)+0.0 << '\n';
    }
    outfile4.close();
    
    // output the final thickness
    isInnerLayer = true;
    rowvec thicknessin = calculate_thickness(isInnerLayer, face, vertexin0, vertexmid0, one_ring_nodes, param, shape_functions, subMatrix);
    isInnerLayer = false;
    rowvec thicknessout = calculate_thickness(isInnerLayer, face, vertexmid0, vertexout0, one_ring_nodes, param, shape_functions, subMatrix);
    ofstream outfile6("thicknessin.csv");
    for (int j = 0; j < face.n_rows; j++) {
        outfile6 << setprecision(16) << thicknessin(j)+0.0 << '\n';
    }
    outfile6.close();  
    ofstream outfile7("thicknessout.csv");
    for (int j = 0; j < face.n_rows; j++) {
        outfile7 << setprecision(16) << thicknessout(j)+0.0 << '\n';
    }
    outfile7.close();  
    
    // output the final shape; triangle centers on the limit surface
    mat centersin = calculate_triangleCenter(face, vertexin0, one_ring_nodes, param, shape_functions, subMatrix);
    mat centersmid = calculate_triangleCenter(face, vertexmid0, one_ring_nodes, param, shape_functions, subMatrix);
    mat centersout = calculate_triangleCenter(face, vertexout0, one_ring_nodes, param, shape_functions, subMatrix);
    ofstream outfile8("triangleCentersIn.csv");
    for (int j = 0; j < face.n_rows; j++) {
        outfile8 << setprecision(16) << centersin(j,0)+0.0 << ',' << centersin(j,1)+0.0 << ',' << centersin(j,2)+0.0 << '\n';
    }
    outfile8.close();  
    ofstream outfile9("triangleCentersMid.csv");
    for (int j = 0; j < face.n_rows; j++) {
        outfile9 << setprecision(16) << centersmid(j,0)+0.0 << ',' << centersmid(j,1)+0.0 << ',' << centersmid(j,2)+0.0 << '\n';
    }
    outfile9.close(); 
    ofstream outfile10("triangleCentersOut.csv");
    for (int j = 0; j < face.n_rows; j++) {
        outfile10 << setprecision(16) << centersout(j,0)+0.0 << ',' << centersout(j,1)+0.0 << ',' << centersout(j,2)+0.0 << '\n';
    }
    outfile10.close(); 
}
