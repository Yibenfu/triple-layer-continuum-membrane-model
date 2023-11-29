#include <math.h>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include "functions_membrane_energetics.cpp"


// A: one_ring_vertices; A(i,1)==A(i,2), means face(i) is irregular, has 11 one-ring vertices
void Energy_and_Force(Mat<int> face, mat vertexin, mat vertexinref, mat vertexmid, mat vertexmidref, mat vertexout, mat vertexoutref,  
                      Mat<int> one_ring_nodes, Param& param, rowvec elementS0in, rowvec elementS0out,
                      rowvec spontcurv_in, rowvec spontcurv_out, Row<int> isinsertionpatchin, Row<int> isinsertionpatchout, 
                      Row<double>& E, mat& Fin, mat& Fmid, mat& Fout,  rowvec& deformnumbers, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // initialize the output parameters;
    int numvertex = vertexin.n_rows;
    int numface = face.n_rows;
    int numvertextotal = 3 * numvertex;
    int numfacetotal = 3 * numface;
    E.fill(0.0);
    Fin.fill(0.0); Fout.fill(0.0); Fmid.fill(0.0);
    //bool isInsertionAreaConstraint = param.isInsertionAreaConstraint;
    // update the total area and volume;
    rowvec elementSin(numface); rowvec elementVin(numface); 
    cell_area_volume(face, vertexin, one_ring_nodes, isinsertionpatchin, param.GaussQuadratureN, elementSin, elementVin, gqcoeff, shape_functions, subMatrix, param.subDivideTimes);
    double Sin = sum(elementSin); double Vin = sum(elementVin); 
    param.Sin = Sin; param.Vin = Vin;
    rowvec elementSout(numface); rowvec elementVout(numface); 
    cell_area_volume(face, vertexout, one_ring_nodes, isinsertionpatchout, param.GaussQuadratureN, elementSout, elementVout, gqcoeff, shape_functions, subMatrix, param.subDivideTimes);
    double Sout = sum(elementSout); double Vout = sum(elementVout);
    param.Sout = Sout; param.Vout = Vout;

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    // area elasticity, and volume constaint 
    double S0in = param.S0in; 
    double S0out = param.S0out; 
    double Eareain = 0.5 * param.us_in/S0in*pow(Sin-S0in,2.0);
    double Eareaout = 0.5 * param.us_out/S0out*pow(Sout-S0out,2.0);
    if ( param.isGlobalAreaConstraint == false ){
        int whichLayer = 0; 
        Eareain  = LocalAreaConstraintEnergy(whichLayer, vertexin,  face,  one_ring_nodes,  param, isinsertionpatchin,  gqcoeff, shape_functions, subMatrix, 2.0*elementS0in );
        whichLayer = 2; 
        Eareaout = LocalAreaConstraintEnergy(whichLayer, vertexout, face, one_ring_nodes, param, isinsertionpatchout, gqcoeff, shape_functions, subMatrix, 2.0*elementS0out);
    
    }
    // volume constraint is only on the inner mesh
    double uv = param.uv;
    double V0in = param.V0in; 
    double Evolume = 0.5*uv/V0in*pow(Vin-V0in,2.0); 

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    // bending force and energy, constraint force and energy
    // for in-layer
    mat fbein(numvertex,3); fbein.fill(0);
    rowvec Ebendingin(numface); Ebendingin.fill(0);
    if ( param.isBilayerModel == true )
    {
        // bending and area/volume terms
        #pragma omp parallel for reduction(+:fbein) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( isinsertionpatchin(i) == 1 ){ 
                isInsertionPatch = true;
            }
            double Cin = spontcurv_in(i); // spontaneous curvature of each patch, 
            int whichLayer = 0; // inner layer
            double ebein = 0.0; 
            element_energy_force(whichLayer, isInsertionPatch, one_ring_nodes.row(i), param, Cin, ebein, fbein, vertexin, gqcoeff, shape_functions, subMatrix, 2.0*elementS0in(i));
            Ebendingin(i) = ebein; 
        }     
    }

    // for out-layer
    mat fbeout(numvertex,3); fbeout.fill(0);
    rowvec Ebendingout(numface); Ebendingout.fill(0);
    {
        // bending and area/volume terms
        #pragma omp parallel for reduction(+:fbeout) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( isinsertionpatchout(i) == 1 ){
                isInsertionPatch = true;
            }
            double Cout = spontcurv_out(i); // spontaneous curvature of each patch, 
            int whichLayer = 2; // outer layer 
            double ebeout = 0.0; 
            element_energy_force(whichLayer, isInsertionPatch, one_ring_nodes.row(i), param, Cout, ebeout, fbeout, vertexout, gqcoeff, shape_functions, subMatrix, 2.0*elementS0out(i));
            Ebendingout(i) = ebeout; 
        }
    }

    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    // for thickness term
    // thickness term, for in-layer
    mat fthickin(numvertextotal,3); fthickin.fill(0.0);
    rowvec Ethickin(numface); Ethickin.fill(0);
    if ( param.isBilayerModel == true ){  
        #pragma omp parallel for reduction(+:fthickin) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( isinsertionpatchin(i) == 1 ){
                isInsertionPatch = true;
            }
            int whichLayer = 0; // inner layer 
            double ethick = 0.0;
            element_energy_force_thickness(isInsertionPatch, whichLayer, one_ring_nodes.row(i), param, ethick, fthickin, vertexin, vertexmid, gqcoeff, shape_functions, subMatrix);            
            Ethickin(i) = ethick;
        }
    }

    // for out-layer
    mat fthickout(numvertextotal,3); fthickout.fill(0.0);
    rowvec Ethickout(numface); Ethickout.fill(0);
    if ( param.isBilayerModel == true ){  
        #pragma omp parallel for reduction(+:fthickout) 
        for ( int i = 0; i < numface; i++) {
            bool isInsertionPatch = false;
            if ( isinsertionpatchout(i) == 1 ){
                isInsertionPatch = true;
            }
            int whichLayer = 2; // out-layer 
            double ethick = 0.0;
            element_energy_force_thickness(isInsertionPatch, whichLayer, one_ring_nodes.row(i), param, ethick, fthickout, vertexmid, vertexout, gqcoeff, shape_functions, subMatrix);            
            Ethickout(i) = ethick;
        }
    }
    
    mat fthick = fthickin + fthickout;

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    // insertion shape constraint energy, and the adjacent patches 
    // for in-layer
    mat finsertin(numvertex,3); finsertin.fill(0);
    double Einsertin = 0.0;
    int whichLayer = 0; // in-layer 
    insertion_shape_constraint(whichLayer, param, face, vertexin, Einsertin, finsertin); 
    
    // for out-layer
    mat finsertout(numvertex,3); finsertout.fill(0);
    double Einsertout = 0.0;
    whichLayer = 2; // out-layer 
    insertion_shape_constraint(whichLayer, param, face, vertexout, Einsertout, finsertout); 

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    // regularization force and energy
    mat finre(numvertex,3); finre.fill(0.0);
    double Ein_regularization = 0.0; 
    rowvec deformnumbersin(deformnumbers.n_cols); // deformnumbersin.fill(0);
    whichLayer = 0;
    energy_force_regularization(whichLayer, vertexin, vertexinref, face, param, isinsertionpatchin, Ein_regularization, finre, deformnumbersin);
    
    mat fmidre(numvertex,3); 
    double Emid_regularization = 0.0; 
    rowvec deformnumbersmid(deformnumbers.n_cols); // deformnumbersmid.fill(0);
    whichLayer = 1;
    energy_force_regularization(whichLayer, vertexmid, vertexmidref, face, param, isinsertionpatchin, Emid_regularization, fmidre, deformnumbersmid);    
    
    mat foutre(numvertex,3); 
    double Eout_regularization = 0.0; 
    rowvec deformnumbersout(deformnumbers.n_cols); // deformnumbersout.fill(0);
    whichLayer = 2;
    energy_force_regularization(whichLayer, vertexout, vertexoutref, face, param, isinsertionpatchout, Eout_regularization, foutre, deformnumbersout);    
    
    deformnumbers = deformnumbersin + deformnumbersmid + deformnumbersout;
    double E_regularization = Ein_regularization + Emid_regularization + Eout_regularization;

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    // total energy and force
    // bending energy
    double E_bending = sum(Ebendingin) + sum(Ebendingout);
    // thickness energy
    double E_thick = sum(Ethickin) + sum(Ethickout);  
    // insertion shape-constraint energy 
    double E_insert = Einsertin + Einsertout;  
    // area elasticity energy and volume constraint.
    double E_constraint = Eareain + Eareaout + Evolume;

    ///////////////////////////////////////////////////////////////////////////
    // force
    mat fin  = fbein  + fthick.submat(span(0,numvertex-1),span(0,2)) + finsertin;
    mat fmid = fthick.submat(span(numvertex, 2*numvertex-1),span(0,2));
    mat fout = fbeout + fthick.submat(span(2*numvertex, numvertextotal-1),span(0,2)) + finsertout;
    // the total force is the sum of bending, constraint and regularization force.
    Fin  = -fin  + finre;
    Fmid = -fmid + fmidre;
    Fout = -fout + foutre;
    ////////////////////////////////////////////////////////////////////////////
    // the total energy is the sum of bending, constraint and regularization energy
    double E_tilt = 0.0;
    E << E_bending << E_insert << E_thick << E_tilt << E_constraint << E_regularization << E_bending + E_constraint + E_insert + E_thick + E_tilt + E_regularization << endr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
double line_search_a_to_minimize_energy(double a0, Mat<double>& dx, Mat<int> face, Mat<double> vertexin, Mat<double>& vertexinref, Mat<double> vertexmid, Mat<double>& vertexmidref, Mat<double> vertexout, Mat<double>& vertexoutref, 
                Mat<int> one_ring_nodes, Param& param, rowvec elementS0in, rowvec elementS0out, rowvec spontcurv_in, rowvec spontcurv_out, Row<int> Isinsertionpatchin, Row<int> Isinsertionpatchout, 
                rowvec energy0, Mat<double> forcein0, Mat<double> forcemid0, Mat<double> forceout0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    double a;
    int numvertex = vertexin.n_rows;
    int numvertextotal = 3 * numvertex;
    mat forcein1(numvertex,3);
    mat forcemid1(numvertex,3);
    mat forceout1(numvertex,3);
    rowvec energy1(7);
    rowvec deformnumbers(3);
    
    a = a0;
    bool isCriterionSatisfied = false;
    double c1 = 1e-4;
    double c2 = 0.1;
    double E0 = energy0(6);
    double E1;
    bool isEnergyDecreased = false;

    //bool WolfeConditions = true;
    param.isNCGstucked = false;
    if ( param.duringStepsToIncreaseInsertDepth == true ) param.isNCGstucked = true;
    bool usingRpi = true;
    param.usingRpi = true;
    
    mat force0(numvertextotal,3); 
    force0.submat(span(0,numvertex-1),span(0,2)) = forcein0; 
    force0.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid0; 
    force0.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout0; 
    mat shu0 = tovector(-force0) * strans(tovector(dx));
    
    while ( isCriterionSatisfied == false ) {
        a = a * 0.8;
        mat vertexinnew  = vertexin + a*dx.submat(span(0,numvertex-1),span(0,2));
        mat vertexmidnew = vertexmid + a*dx.submat(span(numvertex,2*numvertex-1),span(0,2));
        mat vertexoutnew = vertexout + a*dx.submat(span(2*numvertex,numvertextotal-1),span(0,2));
        Energy_and_Force(face, vertexinnew, vertexinref, vertexmidnew, vertexmidref, vertexoutnew, vertexoutref,
                         one_ring_nodes, param, elementS0in, elementS0out, spontcurv_in, spontcurv_out, 
                         Isinsertionpatchin, Isinsertionpatchout, energy1, forcein1, forcemid1, forceout1, 
                         deformnumbers, gqcoeff, shape_functions, subMatrix);
        E1 = energy1(6);
        if ( param.usingNCG == true && param.isNCGstucked == false ){
            mat force1(numvertextotal,3); 
            force1.submat(span(0,numvertex-1),span(0,2)) = forcein1;
            force1.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid1; 
            force1.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout1; 
            mat shu1 = tovector(-force1) * strans(tovector(dx));
            //if ( energy1(3) <= energy0(3) + c1*a*shu0(0,0) && shu1(0,0) >= c2*shu0(0,0) ){ // Wolfe conditions
            if ( E1 <= E0 + c1*a*shu0(0,0) && abs(shu1(0,0)) <= c2*abs(shu0(0,0)) ){ // strong Wolfe conditions
                break;
            }
            if ( a < 1.0e-9 ) {
                cout<<"Now change the NCG WolfeConditions to simple line search method!"<<endl;
                //WolfeConditions = false;
                a = a0;
                vertexinref = vertexin;
                vertexmidref = vertexmid;
                vertexoutref = vertexout;
                param.isNCGstucked = true;
                dx.submat(span(0,numvertex-1),span(0,2)) = forcein0;
                dx.submat(span(numvertex,2*numvertex-1),span(0,2)) = forcemid0;
                dx.submat(span(2*numvertex,numvertextotal-1),span(0,2)) = forceout0;
            }
        }else{
            param.isNCGstucked = true;
            if ( E1 < E0 ) {
                break;
            }
            if ( a < 1.0e-9 && usingRpi == true ){
                param.usingRpi = false;
                usingRpi = false;
                a = a0;
            }
            if ( a < 1.0e-20 && usingRpi == false ) {
                 cout<<"Note: cannot find an efficient samll a to minimize the energy even with the simple method!"<<endl;
                 break;
                 //return -1;
            }
        }
    }
    cout<<"value of stepsize: initial a = "<<a0<<", final a = "<<a<<endl;
    return a;
}

void read_struture_vertex(Mat<double>& vertex, char* filename) {
    ifstream fin(filename);
    string line;
    int i = 0;
    while ( getline(fin,line) ) {
        istringstream sin(line);
        vector<string> positions;
        string info;
        while (getline(sin, info, ',')) {
            positions.push_back(info);
        }
        string xstr = positions[0];
        string ystr = positions[1];
        string zstr = positions[2];
        double x, y, z;
        stringstream sx, sy, sz;
        sx << xstr;
        sy << ystr;
        sz << zstr;
        sx >> x;
        sy >> y;
        sz >> z;
        vertex(i,0) = x;
        vertex(i,1) = y;
        vertex(i,2) = z;
        i++;
    }
    if ( i != vertex.n_rows ) {
        cout<< "Warning: vertices number is "<<vertex.n_rows<<" but read number i = "<< i << endl;
    }
}

void printoutGlobalNCG(mat vertex1) {
    int numvertex = vertex1.n_rows;
    ofstream outfile("vertex_Global_NCG.csv");
    for (int j = 0; j < numvertex; j++) {
        outfile << vertex1(j,0)+0.0 << ',' << vertex1(j,1)+0.0 << ',' << vertex1(j,2)+0.0 << '\n';
    }
    outfile.close();
}
void printoutGlobalGD(mat vertex1) {
    int numvertex = vertex1.n_rows;
    ofstream outfile("vertex_Global_GD.csv");
    for (int j = 0; j < numvertex; j++) {
        outfile << vertex1(j,0)+0.0 << ',' << vertex1(j,1)+0.0 << ',' << vertex1(j,2)+0.0 << '\n';
    }
    outfile.close();
}
void printoutstuck(mat vertex1) {
    int numvertex = vertex1.n_rows;
    ofstream outfile("vertex_stuck.csv");
    for (int j = 0; j < numvertex; j++) {
        outfile << vertex1(j,0)+0.0 << ',' << vertex1(j,1)+0.0 << ',' << vertex1(j,2)+0.0 << '\n';
    }
    outfile.close();
}

void printoutREF(mat vertex1){
    int numvertex = vertex1.n_rows;
    ofstream outfile("vertex_reference.csv");
    for (int j = 0; j < numvertex; j++) {
        outfile << vertex1(j,0)+0.0 << ',' << vertex1(j,1)+0.0 << ',' << vertex1(j,2)+0.0 << '\n';
    }
    outfile.close();
}

void printoutforce(mat vertex1){
    int numvertex = vertex1.n_rows;
    ofstream outfile("force.csv");
    for (int j = 0; j < numvertex; j++) {
        outfile << vertex1(j,0)+0.0 << ',' << vertex1(j,1)+0.0 << ',' << vertex1(j,2)+0.0 << '\n';
    }
    outfile.close();
}

void printout_spontaneouscurvature(bool isInnerLayer, rowvec spontcurv){
    if ( isInnerLayer == true ){
        ofstream outfile11("spont_in.csv");
        for (int i = 0; i < spontcurv.n_cols; i++) {
            outfile11 << setprecision(16) << spontcurv(i)+0.0 << '\n';
        }
        outfile11.close(); 
    }else{
        ofstream outfile11("spont_out.csv");
        for (int i = 0; i < spontcurv.n_cols; i++) {
            outfile11 << setprecision(16) << spontcurv(i)+0.0 << '\n';
        }
        outfile11.close(); 
    }
}

void check_nodal_force(Mat<int> face, mat vertexin, mat vertexinref, mat vertexmid, mat vertexmidref, mat vertexout, mat vertexoutref,  
                      Mat<int> one_ring_nodes, Param& param, rowvec elementS0in, rowvec elementS0out,
                      rowvec spontcurv_in, rowvec spontcurv_out, Row<int> isinsertionpatchin, Row<int> isinsertionpatchout, 
                      Row<double>& EE, mat& Fin, mat& Fmid, mat& Fout, rowvec& deformnumbers, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix){
    cout<<"check if the nodal force is correct: "<<endl;
    int numvertexin = vertexin.n_rows;
    int numvertexmid = vertexmid.n_rows;
    int numvertexout = vertexout.n_rows;
    rowvec energy(7); energy.fill(0);
    mat forcein(numvertexin,3); forcein.fill(0);
    mat forcemid(numvertexmid,3); forcemid.fill(0);
    mat forceout(numvertexout,3); forceout.fill(0);
    mat force(numvertexin+numvertexout+numvertexmid,3); force.fill(0);
    //cout<<"calulate force and energy, begin."<<endl;
    Energy_and_Force(face, vertexin, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                    spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forcein, forcemid, forceout, deformnumbers, gqcoeff, shape_functions, subMatrix);
    //cout<<"calulate force and energy, end."<<endl;
  
    double E = energy(6); 
    double dx=1e-7;
  
    mat forceinha(numvertexin,3);
    mat forcemidha(numvertexmid,3);
    mat forceoutha(numvertexout,3);
 
 
  // inner layer 
  {
    vec forcescale = toscale(forcein);
    mat forcereal(numvertexin,3); forcereal.fill(0);
    for (int i =0; i<numvertexin; i++) {
        mat vertextry = vertexin;
        vertextry(i,0)=vertexin(i,0)+dx;
        Energy_and_Force(face, vertextry, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fx = - (energy(6)-E)/(dx);
        
        vertextry = vertexin;
        vertextry(i,1)=vertexin(i,1)+dx;
        Energy_and_Force(face, vertextry, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fy = - (energy(6)-E)/(dx);
        
        vertextry = vertexin;
        vertextry(i,2)=vertexin(i,2)+dx;
        Energy_and_Force(face, vertextry, vertexinref, vertexmid, vertexmidref, vertexout, vertexoutref, one_ring_nodes,  param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fz = - (energy(6)-E)/(dx);
        
        forcereal(i,0)=fx;
        forcereal(i,1)=fy;
        forcereal(i,2)=fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(forcein.row(i),2.0);
        double cha = norm(forcereal.row(i)-forcein.row(i),2.0);
        cout<<"Inlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - forcein;
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
/*
  // mid- layer 
  {
    vec forcescale = toscale(forcemid);
    mat forcereal(numvertexmid,3); forcereal.fill(0);
    for (int i =0; i < numvertexmid; i++) {
        mat vertextry = vertexmid;
        vertextry(i,0)=vertexmid(i,0)+dx;
        Energy_and_Force(vertexin, vertexinref, vertextry, vertexmidref, vertexout, vertexoutref, facein, facemid, faceout, one_ring_nodesin, one_ring_nodesmid, one_ring_nodesout, param, elementS0in, elementS0mid, elementS0out,
                      spontcurv_in, spontcurv_mid, spontcurv_out, isinsertionpatchin, isinsertionpatchmid, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fx = - (energy(6)-E)/(dx);
        
        vertextry = vertexmid;
        vertextry(i,1)=vertexmid(i,1)+dx;
        Energy_and_Force(vertexin, vertexinref, vertextry, vertexmidref, vertexout, vertexoutref, facein, facemid, faceout, one_ring_nodesin, one_ring_nodesmid, one_ring_nodesout, param, elementS0in, elementS0mid, elementS0out,
                      spontcurv_in, spontcurv_mid, spontcurv_out, isinsertionpatchin, isinsertionpatchmid, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fy = - (energy(6)-E)/(dx);
        
        vertextry = vertexmid;
        vertextry(i,2)=vertexmid(i,2)+dx;
        Energy_and_Force(vertexin, vertexinref, vertextry, vertexmidref, vertexout, vertexoutref, facein, facemid, faceout, one_ring_nodesin, one_ring_nodesmid, one_ring_nodesout, param, elementS0in, elementS0mid, elementS0out,
                      spontcurv_in, spontcurv_mid, spontcurv_out, isinsertionpatchin, isinsertionpatchmid, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fz = - (energy(6)-E)/(dx);
        
        forcereal(i,0)=fx;
        forcereal(i,1)=fy;
        forcereal(i,2)=fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(forcemid.row(i),2.0);
        double cha = norm(forcereal.row(i)-forcemid.row(i),2.0);
        cout<<"midlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - forcemid;
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
*/
  // outer layer 
  {
    vec forcescale = toscale(forceout);
    mat forcereal(numvertexout,3); forcereal.fill(0);
    for (int i =0; i<numvertexout; i++){
        mat vertextry = vertexout;
        vertextry(i,0)=vertexout(i,0)+dx;
        //mat forcetemp(numvertexout,3); forcetemp.fill(0);
        Energy_and_Force(face, vertexin, vertexinref, vertexmid, vertexmidref, vertextry, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fx = - (energy(6)-E)/dx;
        
        vertextry = vertexout;
        vertextry(i,1)=vertexout(i,1)+dx;
        Energy_and_Force(face, vertexin, vertexinref, vertexmid, vertexmidref, vertextry, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fy = - (energy(6)-E)/dx;
        
        vertextry = vertexout;
        vertextry(i,2)=vertexout(i,2)+dx;
        Energy_and_Force(face, vertexin, vertexinref, vertexmid, vertexmidref, vertextry, vertexoutref, one_ring_nodes, param, elementS0in, elementS0out,
                      spontcurv_in, spontcurv_out, isinsertionpatchin, isinsertionpatchout, energy, forceinha, forcemidha, forceoutha, deformnumbers, gqcoeff, shape_functions, subMatrix);
        double fz = - (energy(6)-E)/dx;
        forcereal(i,0)=fx;
        forcereal(i,1)=fy;
        forcereal(i,2)=fz;
        double frea = norm(forcereal.row(i),2.0);
        double f = norm(forceout.row(i),2.0);
        double cha = norm(forcereal.row(i)-forceout.row(i),2.0);
        cout<<"Outlayer: "<<i<<setprecision(15)<<". f= "<<f<<", freal= "<<frea<<". cha= "<<cha<<endl;
    }
    mat dforce = forcereal - forceout;
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
}
