#include <math.h>
#include <ctime>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include "functions_membrane_energetics.cpp"


// A: one_ring_vertices; A(i,1)==A(i,2), means face(i) is irregular, has 11 one-ring vertices
void Energy_and_Force(Mat<int> face, Vertex3Layers vertex3, Vertex3Layers vertex3ref, Mat<int> one_ring_nodes, 
                      Param& param, ElementS03Layers elementS03, SpontCurv3Layers spontCurv3, Row<double>& E, 
                      Force3Layers& force3, rowvec& deformnumbers, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // NOTE: nodal force should be the negative of the energy derivative to the nodal position. Here, the negative sign will be added at the last.
    double E_bending = 0.0;        // bending energy
    double E_thick = 0.0;          // height/thickness energy
    double E_constraint = 0.0;     // area and volume constraint energy
    double E_regularization = 0.0; // mesh regularization energy
    double E_splayTilt = 0.0;           // tilt energy (none)
    double E_insert = 0.0;         // insertion shape constriant energy
    double E_total = 0.0;
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    // single-layer mesh model
    if ( param.isBilayerModel == false ){
        // initialize the output parameters;
        int numvertex = vertex3.outlayer.n_rows;
        int numface = face.n_rows;
        E.fill(0.0);
        // update the total area and volume;
        rowvec elementSout(numface); rowvec elementVout(numface); 
        cell_area_volume(face, vertex3.outlayer, one_ring_nodes, param.GaussQuadratureN, elementSout, elementVout, gqcoeff, shape_functions, subMatrix, param.subDivideTimes);
        double Sout = sum(elementSout); double Vout = sum(elementVout);
        param.Sout = Sout; param.Vout = Vout;

        // area elasticity, and volume constaint 
        double Eareaout = 0.5 * param.us_out/param.S0out*pow(param.Sout-param.S0out,2.0);
        if ( param.isGlobalAreaConstraint == false ){
            int whichLayer = 2; 
            Eareaout = LocalAreaConstraintEnergy(whichLayer, vertex3.outlayer, face, one_ring_nodes, param, param.isInsertionPatch.outlayer, gqcoeff, shape_functions, subMatrix, 2.0*elementS03.outlayer);
        }
        // volume constraint is only on the out-layer mesh
        double uv = param.uv;
        double V0out = param.V0out; 
        double Evolume = 0.5*uv/V0out * pow(Vout-V0out,2.0); 
        
        // bending force and energy, constraint force and energy
        // for out-layer
        mat fbeout(numvertex,3); fbeout.fill(0);
        rowvec Ebendingout(numface); Ebendingout.fill(0);
        // bending and area/volume terms
        #pragma omp parallel for reduction(+:fbeout) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( param.isInsertionPatch.outlayer[i] == true ){
                isInsertionPatch = true;
            }
            double Cout = spontCurv3.outlayer(i); // spontaneous curvature of each patch, 
            int whichLayer = 2; // outer layer 
            double ebeout = 0.0; 
            element_energy_force(whichLayer, isInsertionPatch, one_ring_nodes.row(i), param, Cout, ebeout, fbeout, vertex3.outlayer, gqcoeff, shape_functions, subMatrix, 2.0*elementS03.outlayer(i));
            Ebendingout(i) = ebeout; 
        }

        // insertion shape constraint energy, and the adjacent patches 
        // for out-layer
        mat finsertout(numvertex,3); finsertout.fill(0);
        double Einsertout = 0.0;
        int whichLayer = 2; // out-layer 
        //insertion_shape_constraint(whichLayer, param, face, vertex3.outlayer, Einsertout, finsertout); 
        
        // regularization force and energy
        mat foutre(numvertex,3); 
        double Eout_regularization = 0.0; 
        rowvec deformnumbersout(deformnumbers.n_cols); // deformnumbersout.fill(0);
        energy_force_regularization(vertex3.outlayer, vertex3ref.outlayer, face, param, Eout_regularization, foutre, deformnumbersout);    
        deformnumbers =  deformnumbersout;
        // record whether the regularization insults deformed shape or area. 
        // If so, the energy will jump in this step. Then, there may not be suitable stepSize to decrease the energy when calling 'line_search_a_to_minimize_energy'.
        int deformAreaOrShape = deformnumbers(0) + deformnumbers(1);
        if ( deformAreaOrShape > 0 ){
            param.deformAreaOrShape = true;
        }else{
            param.deformAreaOrShape = false;
        }

        // total energy and force
        E_bending = sum(Ebendingout);      // bending energy
        E_constraint = Eareaout + Evolume; // area elasticity energy and volume constraint.
        E_insert = Einsertout;             // insertion shape-constraint energy 
        E_regularization =  Eout_regularization; // regularizaiotn energy
        E_total = E_bending + E_constraint + E_insert + E_regularization;
        // the total force is the sum of bending, constraint and regularization force.
        force3.outlayer = -fbeout - finsertout + foutre;
        force3.outlayer  = manage_ghost_force(force3.outlayer, face, param);
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    // three-layer mesh model
    }else{
        // initialize the output parameters;
        int numvertex = vertex3.outlayer.n_rows;
        int numface = face.n_rows;
        int numvertextotal = 3 * numvertex;
        int numfacetotal = 3 * numface;
        E.fill(0.0);
        //bool isInsertionAreaConstraint = param.isInsertionAreaConstraint;
        // update the total area and volume;
        rowvec elementSout(numface); rowvec elementVout(numface); 
        cell_area_volume(face, vertex3.outlayer, one_ring_nodes, param.GaussQuadratureN, elementSout, elementVout, gqcoeff, shape_functions, subMatrix, param.subDivideTimes);
        double Sout = sum(elementSout); double Vout = sum(elementVout);
        param.Sout = Sout; param.Vout = Vout;
        rowvec elementSin(numface); rowvec elementVin(numface);   
        cell_area_volume(face, vertex3.inlayer, one_ring_nodes, param.GaussQuadratureN, elementSin, elementVin, gqcoeff, shape_functions, subMatrix, param.subDivideTimes);
        double Sin = sum(elementSin); double Vin = sum(elementVin); 
        param.Sin = Sin; param.Vin = Vin;
        ///////////////////////////////////////////////////////////////////
        // area elasticity, and volume constaint 
        double Eareaout = 0.5 * param.us_out/param.S0out*pow(param.Sout-param.S0out,2.0);
        double Eareain = 0.5 * param.us_in/param.S0in*pow(param.Sin-param.S0in,2.0);
        if ( param.isGlobalAreaConstraint == false ){
            int whichLayer = 2; 
            Eareaout = LocalAreaConstraintEnergy(whichLayer, vertex3.outlayer, face, one_ring_nodes, param, param.isInsertionPatch.outlayer, gqcoeff, shape_functions, subMatrix, 2.0*elementS03.outlayer);
            whichLayer = 0; 
            Eareain  = LocalAreaConstraintEnergy(whichLayer, vertex3.inlayer,  face,  one_ring_nodes,  param, param.isInsertionPatch.inlayer,  gqcoeff, shape_functions, subMatrix, 2.0*elementS03.inlayer );
        }
        // volume constraint is only on the out-layer mesh
        double uv = param.uv;
        double V0out = param.V0out; 
        double Evolume = 0.5*uv/V0out * pow(Vout-V0out,2.0); 
        /////////////////////////////////////////////////////////////////////////
        // for for bending, height, splay-tilt, area and volume elasticity
        // for out-layer
        mat fthickout(numvertextotal,3); fthickout.fill(0.0);
        mat Ethickout(numface,3); Ethickout.fill(0.0); // the 1 column is bending Energey; 2nd is Heihgt or Tilt; 3rd is splay-tilt 
        int whichLayer = 2; // out-layer 
        #pragma omp parallel for reduction(+:fthickout) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( param.isInsertionPatch.outlayer[i] == true ){
                isInsertionPatch = true;
            }
            double C0out = spontCurv3.outlayer(i); // spontaneous curvature of each patch,          
            rowvec ethick(3); ethick.fill(0.0);
            element_energy_force_thickness(isInsertionPatch, whichLayer, one_ring_nodes.row(i), param, ethick, fthickout, vertex3.midlayer, vertex3.outlayer, C0out, 2.0*elementS03.outlayer(i), gqcoeff, shape_functions, subMatrix);            
            Ethickout.row(i) = ethick;
        }
        // for in-layer
        mat fthickin(numvertextotal,3); fthickin.fill(0.0);
        mat Ethickin(numface,3); Ethickin.fill(0.0);  
        whichLayer = 0; // in-layer 
        #pragma omp parallel for reduction(+:fthickin) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( param.isInsertionPatch.inlayer[i] == true ){
                isInsertionPatch = true;
            }
            double C0in = spontCurv3.inlayer(i); // spontaneous curvature of each patch,           
            rowvec ethick(3); ethick.fill(0.0);
            element_energy_force_thickness(isInsertionPatch, whichLayer, one_ring_nodes.row(i), param, ethick, fthickin, vertex3.midlayer, vertex3.inlayer, C0in, 2.0*elementS03.inlayer(i), gqcoeff, shape_functions, subMatrix);            
            Ethickin.row(i) = ethick;
        }
        //////////////////////////////////////////////////////////////////////////////
        // insertion shape constraint energy, and the adjacent patches 
        // for out-layer
        mat finsertout(numvertex,3); finsertout.fill(0);
        double Einsertout = 0.0;
        whichLayer = 2; // out-layer 
        //insertion_shape_constraint(whichLayer, param, face, vertex3.outlayer, Einsertout, finsertout);
        // for patch of the mid-layer, and the adjacent patches 
        mat finsertmid(numvertex,3); finsertmid.fill(0);
        double Einsertmid = 0.0;
        //insertion_shape_constraint(param, face, vertex3.midlayer, Einsertmid, finsertmid);  
        // for in-layer
        mat finsertin(numvertex,3); finsertin.fill(0);
        double Einsertin = 0.0;
        whichLayer = 0; // in-layer 
        //insertion_shape_constraint(whichLayer, param, face, vertex3.inlayer, Einsertin, finsertin); 
        ////////////////////////////////////////////////////////////////////////////////////
        // regularization force and energy
        mat foutre(numvertex,3); 
        double Eout_regularization = 0.0; 
        rowvec deformnumbersout(deformnumbers.n_cols); // deformnumbersout.fill(0);
        energy_force_regularization(vertex3.outlayer, vertex3ref.outlayer, face, param, Eout_regularization, foutre, deformnumbersout);    
        mat fmidre(numvertex,3); 
        double Emid_regularization = 0.0; 
        rowvec deformnumbersmid(deformnumbers.n_cols); // deformnumbersmid.fill(0);
        energy_force_regularization(vertex3.midlayer, vertex3ref.midlayer, face, param, Emid_regularization, fmidre, deformnumbersmid);    
        mat finre(numvertex,3); finre.fill(0.0);
        double Ein_regularization = 0.0; 
        rowvec deformnumbersin(deformnumbers.n_cols); // deformnumbersin.fill(0);
        energy_force_regularization(vertex3.inlayer, vertex3ref.inlayer, face, param, Ein_regularization, finre, deformnumbersin);
        
        deformnumbers = deformnumbersin + deformnumbersmid + deformnumbersout;
        // record whether the regularization insults deformed shape or area. 
        // If so, the energy will jump in this step. Then, there may not be suitable stepSize to decrease the energy when calling 'line_search_a_to_minimize_energy'.
        int deformAreaOrShape = deformnumbers(0) + deformnumbers(1);
        if ( deformAreaOrShape > 0 ){
            param.deformAreaOrShape = true;
        }else{
            param.deformAreaOrShape = false;
        }
        ///////////////////////////////////////////////////////////////////////////
        // total energy and force
        E_bending = sum(Ethickout.col(0)) + sum(Ethickin.col(0));
        E_thick = sum(Ethickin.col(1)) + sum(Ethickout.col(1));  
        E_splayTilt = sum(Ethickin.col(2)) + sum(Ethickout.col(2));  
        E_insert = Einsertin + Einsertmid + Einsertout;  
        E_constraint = Eareain + Eareaout + Evolume;
        E_regularization = Ein_regularization + Emid_regularization + Eout_regularization;
        E_total = E_bending + E_constraint + E_insert + E_thick + E_splayTilt + E_regularization;
        // force
        mat fthick = fthickin + fthickout;
        mat fin  = fthick.submat(span(0,numvertex-1),span(0,2)) + finsertin;
        mat fmid = fthick.submat(span(numvertex, 2*numvertex-1),span(0,2)) + finsertmid;
        mat fout = fthick.submat(span(2*numvertex, numvertextotal-1),span(0,2)) + finsertout;
        // the total force is the sum of bending, constraint and regularization force.
        force3.inlayer  = -fin  + finre;
        force3.midlayer = -fmid + fmidre;
        force3.outlayer = -fout + foutre;
        // adjust the nodal force according to the boundary!
        force3.inlayer   = manage_ghost_force(force3.inlayer, face, param);
        force3.midlayer  = manage_ghost_force(force3.midlayer, face, param);
        force3.outlayer  = manage_ghost_force(force3.outlayer, face, param);
    }
    ////////////////////////////////////////////////////////////////////////////
    // the total energy is the sum of bending, constraint and regularization energy
    E << E_bending << E_insert << E_thick << E_splayTilt << E_constraint << E_regularization << E_total << endr;

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // constraint the movement of the vertex along z or r coordinates
    #pragma omp parallel for
    for ( int i = 0; i < vertex3.outlayer.n_rows; i++ ){
        rowvec direction = vertex3.outlayer.row(i) / norm(vertex3.outlayer.row(i), 2); // sphere surface, move along r direction
        force3.outlayer.row(i) = ( force3.outlayer(i,0)*direction(0) + force3.outlayer(i,1)*direction(1) + force3.outlayer(i,2)*direction(2) ) * direction;
        if ( param.isBilayerModel == true ){ 
            force3.inlayer.row(i) = ( force3.inlayer(i,0)*direction(0) + force3.inlayer(i,1)*direction(1) + force3.inlayer(i,2)*direction(2) ) * direction;
            force3.midlayer.row(i) = ( force3.midlayer(i,0)*direction(0) + force3.midlayer(i,1)*direction(1) + force3.midlayer(i,2)*direction(2) ) * direction;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // adjust the precision, to avoid the openmp fluctuation (race condition: each step, it will produce a difference at the 10th decimal place.)
    double alphashu = 1.0e9;
    #pragma omp parallel for
    for ( int i = 0; i < vertex3.outlayer.n_rows; i++ ){
        for ( int j = 0; j < 3; j++ ){
            force3.outlayer(i,j) = round(force3.outlayer(i,j) * alphashu) / alphashu;
            if ( param.isBilayerModel == true ){
                force3.inlayer(i,j) = round(force3.inlayer(i,j) * alphashu) / alphashu;
                force3.midlayer(i,j) = round(force3.midlayer(i,j) * alphashu) / alphashu;
            }
        }
    }
    E(6) = round( E(6) * alphashu ) / alphashu;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
double line_search_a_to_minimize_energy(double a0, Force3Layers& dx, Mat<int> face, Vertex3Layers vertex3, Vertex3Layers& vertex3ref, Mat<int> one_ring_nodes, 
                Param& param, ElementS03Layers elementS03, SpontCurv3Layers spontCurv3, rowvec energy0, 
                Force3Layers force30, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    double a;
    Force3Layers force31 = force30;
    rowvec energy1(7);
    rowvec deformnumbers(3);
    
    a = a0;
    bool isCriterionSatisfied = false;
    double c1 = 1e-4;
    double c2 = 0.1;
    double E0 = energy0(6);
    double E1;
    bool isEnergyDecreased = false;
   
    bool toUpdateReference = false;

    mat shu0 = -1.0 * changeForce3ToVector(force30, param.isBilayerModel) * strans(changeForce3ToVector(dx, param.isBilayerModel));
    Vertex3Layers vertex3new;
    while ( isCriterionSatisfied == false ) {
        a = a * 0.8;
        vertex3new.outlayer = update_vertex(vertex3.outlayer, face, a, dx.outlayer, param); 
        if ( param.isBilayerModel == true ){
            vertex3new.inlayer  = update_vertex(vertex3.inlayer, face, a, dx.inlayer, param); 
            vertex3new.midlayer = update_vertex(vertex3.midlayer, face, a, dx.midlayer, param); 
        }
        Energy_and_Force(face, vertex3new, vertex3ref, one_ring_nodes, 
                         param, elementS03, spontCurv3, energy1, 
                         force31, deformnumbers, gqcoeff, shape_functions, subMatrix);
        E1 = energy1(6);
        //if ( param.usingNCG == true && param.isNCGstucked == false ){
        if ( toUpdateReference == false ){
            mat shu1 = -1.0 * changeForce3ToVector(force31, param.isBilayerModel) * strans(changeForce3ToVector(dx, param.isBilayerModel));
            if ( E1 <= E0 + c1*a*shu0(0,0) && abs(shu1(0,0)) <= c2*abs(shu0(0,0)) ){ // strong Wolfe conditions
                break;
            }
            if ( a < 1.0e-9 ) {
                if ( param.deformAreaOrShape == true ){
                    cout<<"Deformed area or shape will happen! There is no suitable stepSize to decrease the energy. Then, set stepSize as 1.0e-5 to move on the simulation."<<endl;
                    a = 1.0e-5;
                    break;
                }else{
                    cout<<"First try to find the stepSize failed. Then update the reference, and try again."<<endl;
                    vertex3ref = vertex3;
                    toUpdateReference = true;
                    a = a0;
                    dx = force30;
                }
            }
        }else{
            if ( E1 < E0 ){ 
                break;
            }
            if ( a < 1.0e-9 ) {
                if ( param.deformAreaOrShape == true ){
                    cout<<"Deformed area or shape will happen! There is no suitable stepSize to decrease the energy. Then, set stepSize as 1.0e-5 to move on the simulation."<<endl;
                    a = 1.0e-5;
                    break;
                }else{
                    a = -1.0;
                    break;
                }
            }
        }
    }
    cout<<"value of stepsize: initial a = "<<a0<<", final a = "<<a<<endl;
    return a;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// to check whether the nodal force is calculated correctly
void check_nodal_force(Mat<int> face, Vertex3Layers vertex3, Vertex3Layers vertex3ref, Mat<int> one_ring_nodes, 
                      Param param, ElementS03Layers elementS03, SpontCurv3Layers spontCurv3, 
                      rowvec deformnumbers, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix){
    cout<<"check if the nodal force is correct: "<<endl;
    rowvec energy(7); energy.fill(0);
    Force3Layers force3;
    //cout<<"calulate force and energy, begin."<<endl;
    Energy_and_Force(face, vertex3, vertex3ref, one_ring_nodes, param, elementS03,
                    spontCurv3, energy, force3, deformnumbers, gqcoeff, shape_functions, subMatrix);
    //cout<<"calulate force and energy, end."<<endl;
  
    double E = energy(6); 
    double dx = 1.0e-7;
  
    Force3Layers force3ha;

    // outer layer 
  {
    int numvertexout = vertex3.outlayer.n_rows;
    vec forcescale = toscale(force3.outlayer);
    mat forcereal(numvertexout,3); forcereal.fill(0);
    Vertex3Layers vertex3try = vertex3;
    for (int i =0; i < numvertexout; i++){
        vertex3try.outlayer(i,0) = vertex3.outlayer(i,0) + dx;
        //mat forcetemp(numvertexout,3); forcetemp.fill(0);
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
                      spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fx = - (energy(6)-E)/dx;    
        vertex3try.outlayer.row(i) = vertex3.outlayer.row(i);

        vertex3try.outlayer(i,1) = vertex3.outlayer(i,1) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
            spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fy = - (energy(6)-E)/dx;        
        vertex3try.outlayer.row(i) = vertex3.outlayer.row(i);

        vertex3try.outlayer(i,2) = vertex3.outlayer(i,2)+dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
            spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fz = - (energy(6)-E)/dx;
        vertex3try.outlayer.row(i) = vertex3.outlayer.row(i);

        forcereal(i,0) = fx;
        forcereal(i,1) = fy;
        forcereal(i,2) = fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(force3.outlayer.row(i),2.0);
        double cha = norm(forcereal.row(i)-force3.outlayer.row(i),2.0);
        cout<<"Outlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - force3.outlayer;
    vec dforcescale = toscale(dforce);
    vec dfratio(numvertexout);
    dfratio.fill(0);
    int maxindex = -1; double maxdfratio = -1;
    for(int i=0; i<numvertexout; i++) {
        dfratio(i)=dforcescale(i)/forcescale(i);
        if ( dfratio(i) > maxdfratio){
            maxindex = i;
            maxdfratio = dfratio(i);
        }
    }
    cout<<"Minratio = "<<min(dfratio)<< endl;
    cout<<"Maxratio = "<<maxdfratio<<", index = "<<maxindex<<endl;
  }
  
  // inner layer 
  if ( param.isBilayerModel == true ){
    int numvertexin = vertex3.inlayer.n_rows;
    vec forcescale = toscale(force3.inlayer);
    mat forcereal(numvertexin,3); forcereal.fill(0);
    Vertex3Layers vertex3try = vertex3;
    for (int i = 0; i < numvertexin; i++) {
        vertex3try.inlayer(i,0) = vertex3.inlayer(i,0) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fx = - (energy(6)-E)/(dx);
        vertex3try.inlayer.row(i) = vertex3.inlayer.row(i);
        
        vertex3try.inlayer(i,1) = vertex3.inlayer(i,1) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fy = - (energy(6)-E)/(dx);
        vertex3try.inlayer.row(i) = vertex3.inlayer.row(i);
        
        vertex3try.inlayer(i,2) = vertex3.inlayer(i,2) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes, param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fz = - (energy(6)-E)/(dx);
        vertex3try.inlayer.row(i) = vertex3.inlayer.row(i);
        
        forcereal(i,0) = fx;
        forcereal(i,1) = fy;
        forcereal(i,2) = fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(force3.inlayer.row(i),2.0);
        double cha = norm(forcereal.row(i)-force3.inlayer.row(i),2.0);
        cout<<"Inlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - force3.inlayer;
    vec dforcescale = toscale(dforce);
    vec dfratio(numvertexin);
    dfratio.fill(0);
    int maxindex = -1; double maxdfratio = -1;
    for(int i=0; i<numvertexin; i++) {
        dfratio(i)=dforcescale(i)/forcescale(i);
        if ( dfratio(i) > maxdfratio){
            maxindex = i;
            maxdfratio = dfratio(i);
        }
    }
    //cout<<"Minratio = "<<min(dfratio)<<", Maxratio = "<<max(dfratio)<<endl;
    cout<<"Minratio = "<<min(dfratio)<< endl;
    cout<<"Maxratio = "<<maxdfratio<<", index = "<<maxindex<<endl;
  }

  // mid- layer 
  if ( param.isBilayerModel == true ){
    int numvertexmid = vertex3.midlayer.n_rows;
    vec forcescale = toscale(force3.midlayer);
    mat forcereal(numvertexmid,3); forcereal.fill(0);
    Vertex3Layers vertex3try = vertex3;
    for (int i = 0; i < numvertexmid; i++) {
        vertex3try.midlayer(i,0) = vertex3.midlayer(i,0) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes,  param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);        
        double fx = - (energy(6)-E)/(dx);
        vertex3try.midlayer.row(i) = vertex3.midlayer.row(i);
        
        vertex3try.midlayer(i,1) = vertex3.midlayer(i,1) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes,  param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);        
        double fy = - (energy(6)-E)/(dx);
        vertex3try.midlayer.row(i) = vertex3.midlayer.row(i);
        
        vertex3try.midlayer(i,2) = vertex3.midlayer(i,2) + dx;
        Energy_and_Force(face, vertex3try, vertex3ref, one_ring_nodes,  param, elementS03,
                         spontCurv3, energy, force3ha, deformnumbers, gqcoeff, shape_functions, subMatrix);        
        double fz = - (energy(6)-E)/(dx);
        vertex3try.midlayer.row(i) = vertex3.midlayer.row(i);
        
        forcereal(i,0) = fx;
        forcereal(i,1) = fy;
        forcereal(i,2) = fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(force3.midlayer.row(i),2.0);
        double cha = norm(forcereal.row(i)-force3.midlayer.row(i),2.0);
        cout<<"midlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - force3.midlayer;
    vec dforcescale = toscale(dforce);
    vec dfratio(numvertexmid);
    dfratio.fill(0);
    int maxindex = -1; double maxdfratio = -1;
    for(int i=0; i<numvertexmid; i++) {
        dfratio(i)=dforcescale(i)/forcescale(i);
        if ( dfratio(i) > maxdfratio){
            maxindex = i;
            maxdfratio = dfratio(i);
        }
    }
    //cout<<"Minratio = "<<min(dfratio)<<", Maxratio = "<<max(dfratio)<<endl;
    cout<<"Minratio = "<<min(dfratio)<< endl;
    cout<<"Maxratio = "<<maxdfratio<<", index = "<<maxindex<<endl;
  } 
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// if vesicle is limited inside a tube or has boundaries:
void constraintVertexWithinBoundaries(Mat<int> face, Vertex3Layers& vertex3, Param param){
    //////////////////////
    // tube, along z axis.
    
    {
        double radiusTubeTarget = 16.0; // nm
        double radiusInitial = sqrt( param.S0out / 4.0 / M_PI );
        double radiusTube = radiusInitial;
        // radiusTube decreases 1 nm every 100 steps.
        if ( param.currentStep % 50 == 0 && param.currentStep > 0){
            radiusTube = radiusInitial - floor(param.currentStep/50) * 1.0;
            if ( radiusTube < radiusTubeTarget ){
               radiusTube = radiusTubeTarget;
            }
        }
        // if vertex is outside the tube, then move it back onto the tube
        #pragma omp parallel for
        for ( int i = 0; i < vertex3.outlayer.n_rows; i++ ){
            double x = vertex3.outlayer(i,0);
            double y = vertex3.outlayer(i,1);
            double radiusTmp = sqrt( x*x + y*y );
            if ( radiusTmp > radiusTube ){
                vertex3.outlayer(i,0) = x * radiusTube / radiusTmp;
                vertex3.outlayer(i,1) = y * radiusTube / radiusTmp;
            }
        }
    }
    
    /*
    //////////////////////
    // Dumbbell shaped tube
    {
        double radiusInitial = 16.0;// sqrt( param.S0out / 4.0 / M_PI );
        double radiusTubeFarEnd = radiusInitial; // nm
        double radiusTubeNeckTarget = 10.0;      // nm
        double radiusTubeNeck = radiusTubeFarEnd;
        // radiusTubeNeck decreases 1 nm every 100 steps.
        if ( param.currentStep % 100 == 0 && param.currentStep > 0){
            radiusTubeNeck = radiusTubeFarEnd - floor(param.currentStep/100) * 1.0;
            if ( radiusTubeNeck < radiusTubeNeckTarget ){
                radiusTubeNeck = radiusTubeNeckTarget;
            }
        }
        // if vertex is outside the tube, then move it back onto the tube
        for ( int i = 0; i < vertex3.outlayer.n_rows; i++ ){
            double x = vertex3.outlayer(i,0);
            double y = vertex3.outlayer(i,1);
            double z = vertex3.outlayer(i,2);
            double radiusTmp = sqrt( x*x + y*y );
            double radiusTube = radiusTubeFarEnd;
            if ( abs(z) < radiusInitial ){
                radiusTube = (radiusTubeFarEnd - radiusTubeNeck) / 2.0 * ( - cos(M_PI/radiusInitial * z) + 1.0 ) + radiusTubeNeck;
            }
            if ( radiusTmp > radiusTube ){
                vertex3.outlayer(i,0) = x * radiusTube / radiusTmp;
                vertex3.outlayer(i,1) = y * radiusTube / radiusTmp;
            }
        }    
    }
    */
    /*
    //////////////////////
    // curved tube (normal distribution function)
    {
        double radiusInitial = 16.0; // sqrt( param.S0out / 4.0 / M_PI );
        double radiusTubeTarget = 14.0; // nm
        double radiusTube = radiusInitial;
        double sigma = radiusInitial / 2.0;
        double shiftXpeakTarget = 3.0; // the tube arches along +x by this value finally.
        double shiftXpeak = 0.0;
        // shiftX increases 1 nm every 100 steps.
        if ( param.currentStep % 100 == 0 && param.currentStep > 0){
            // arch the tube more toward +x direction 
            shiftXpeak = floor(param.currentStep/100) * 1.0;
            if ( shiftXpeak > shiftXpeakTarget ){
                shiftXpeak = shiftXpeakTarget;
            }
            // shrink the tube radius
            radiusTube = radiusInitial - floor(param.currentStep/100) * 1.0;
            if ( radiusTube < radiusTubeTarget ){
                radiusTube = radiusTubeTarget;
            }
        }
        // if vertex is outside the tube, then move it back onto the tube
        for ( int i = 0; i < vertex3.outlayer.n_rows; i++ ){
            double x = vertex3.outlayer(i,0);
            double y = vertex3.outlayer(i,1);
            double z = vertex3.outlayer(i,2);
            //double shiftX = shiftXpeak * 1.0/sqrt(2.0*M_PI)/sigma * exp( - z*z /2/sigma/sigma ); // normal distribution
            double shiftX = shiftXpeak * exp( - z*z /2/sigma/sigma ); // normal distribution
            rowvec axisPoint(3); axisPoint << shiftX << 0.0 << z << endr;
            double radiusTmp = norm(vertex3.outlayer.row(i) - axisPoint, 2);
            if ( radiusTmp > radiusTube ){
                rowvec vectmp = vertex3.outlayer.row(i) - axisPoint;
                vertex3.outlayer.row(i) = axisPoint + vectmp/radiusTmp * radiusTube;
            }
        }
    }
    */
}