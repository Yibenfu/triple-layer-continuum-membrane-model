#include <math.h>
#include <ctime>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include "functions_setup_triangular_mesh.cpp"

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
        cout<< "    Warning: vertices number is "<<vertex.n_rows<<" but read number i = "<< i << endl;
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

void printout_spontaneouscurvature(int whichLayer, rowvec spontcurv){
    if (whichLayer == 0){ // inlayer
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

void  printout_initial_structures(Mat<int> face, Vertex3Layers vertex, bool isBilayerModel){
    ofstream outfile("face.csv");
    for (int i = 0; i < face.n_rows; i++) {
        outfile << face(i,0)+1 << ',' << face(i,1)+1 << ',' << face(i,2)+1 << '\n';
    }
    outfile.close();
    
    if ( isBilayerModel == false ){
        ofstream outfile2("vertex_begin.csv");
        for (int i = 0; i < vertex.outlayer.n_rows; i++) {
            outfile2 << setprecision(16) << vertex.outlayer(i,0)+0.0 << ',' << vertex.outlayer(i,1)+0.0 << ',' << vertex.outlayer(i,2)+0.0 << '\n';
        }
        outfile2.close();  
    }else{
        ofstream outfile1("vertex_begin_in.csv");
        for (int i = 0; i < vertex.inlayer.n_rows; i++) {
            outfile1 << setprecision(16) << vertex.inlayer(i,0)+0.0 << ',' << vertex.inlayer(i,1)+0.0 << ',' << vertex.inlayer(i,2)+0.0 << '\n';
        }
        outfile1.close();
        ofstream outfile11("vertex_begin_mid.csv");
        for (int i = 0; i < vertex.midlayer.n_rows; i++) {
            outfile11 << setprecision(16) << vertex.midlayer(i,0)+0.0 << ',' << vertex.midlayer(i,1)+0.0 << ',' << vertex.midlayer(i,2)+0.0 << '\n';
        }
        outfile11.close();
        ofstream outfile2("vertex_begin_out.csv");
        for (int i = 0; i < vertex.outlayer.n_rows; i++) {
            outfile2 << setprecision(16) << vertex.outlayer(i,0)+0.0 << ',' << vertex.outlayer(i,1)+0.0 << ',' << vertex.outlayer(i,2)+0.0 << '\n';
        }
        outfile2.close();  
    }
}

void  printout_final_structures(Vertex3Layers vertex, bool isBilayerModel){
    if ( isBilayerModel == false ){
        ofstream outfile2("vertex_final.csv");
        for (int i = 0; i < vertex.outlayer.n_rows; i++) {
            outfile2 << setprecision(16) << vertex.outlayer(i,0)+0.0 << ',' << vertex.outlayer(i,1)+0.0 << ',' << vertex.outlayer(i,2)+0.0 << '\n';
        }
        outfile2.close();  
    }else{
        ofstream outfile1("vertex_final_in.csv");
        for (int i = 0; i < vertex.inlayer.n_rows; i++) {
            outfile1 << setprecision(16) << vertex.inlayer(i,0)+0.0 << ',' << vertex.inlayer(i,1)+0.0 << ',' << vertex.inlayer(i,2)+0.0 << '\n';
        }
        outfile1.close();
        ofstream outfile11("vertex_final_mid.csv");
        for (int i = 0; i < vertex.midlayer.n_rows; i++) {
            outfile11 << setprecision(16) << vertex.midlayer(i,0)+0.0 << ',' << vertex.midlayer(i,1)+0.0 << ',' << vertex.midlayer(i,2)+0.0 << '\n';
        }
        outfile11.close();
        ofstream outfile2("vertex_final_out.csv");
        for (int i = 0; i < vertex.outlayer.n_rows; i++) {
            outfile2 << setprecision(16) << vertex.outlayer(i,0)+0.0 << ',' << vertex.outlayer(i,1)+0.0 << ',' << vertex.outlayer(i,2)+0.0 << '\n';
        }
        outfile2.close();  
    }  
}

void printout_temporary_structures(int kk, Vertex3Layers vertex3, bool isBilayerModel){
    if ( isBilayerModel == false ){
        char filename1[20] = "vertex%d.csv";
        sprintf(filename1,"vertex%d.csv",kk);
        ofstream outfile1(filename1);
        for (int j = 0; j < vertex3.outlayer.n_rows; j++) {
            outfile1 << vertex3.outlayer(j,0)+0.0 << ',' << vertex3.outlayer(j,1)+0.0 << ',' << vertex3.outlayer(j,2)+0.0 << '\n';
        }
        outfile1.close();
    }else{
        char filename[20] = "vertex%din.csv";
        sprintf(filename,"vertex%din.csv",kk);
        ofstream outfile(filename);
        for (int j = 0; j < vertex3.inlayer.n_rows; j++) {
            outfile << vertex3.inlayer(j,0)+0.0 << ',' << vertex3.inlayer(j,1)+0.0 << ',' << vertex3.inlayer(j,2)+0.0 << '\n';
        }
        outfile.close();

        char filename1[20] = "vertex%dout.csv";
        sprintf(filename1,"vertex%dout.csv",kk);
        ofstream outfile1(filename1);
        for (int j = 0; j < vertex3.outlayer.n_rows; j++) {
            outfile1 << vertex3.outlayer(j,0)+0.0 << ',' << vertex3.outlayer(j,1)+0.0 << ',' << vertex3.outlayer(j,2)+0.0 << '\n';
        }
        outfile1.close();

        char filename2[20] = "vertex%dmid.csv";
        sprintf(filename2,"vertex%dmid.csv",kk);
        ofstream outfile22(filename2);
        for (int j = 0; j < vertex3.midlayer.n_rows; j++) {
            outfile22 << vertex3.midlayer(j,0)+0.0 << ',' << vertex3.midlayer(j,1)+0.0 << ',' << vertex3.midlayer(j,2)+0.0 << '\n';
        }
        outfile22.close();
    }
}

void printout_isBoundaryFace(Row<int> isBoundaryFace){
    ofstream outfile88("isBoundaryFace.csv");
    for (int j = 0; j < isBoundaryFace.n_cols; j++) {
        outfile88 << isBoundaryFace(j) << '\n';
    }
    outfile88.close();  
}

void printout_thickness(mat thicknessin, mat thicknessout){
    ofstream outfile6("height_in.csv");
    outfile6 << "Equilibrium" << ',' << "Observed" << '\n'; 
    for (int j = 0; j < thicknessin.n_rows; j++) {
        outfile6 << setprecision(16) << thicknessin(j,0)+0.0 << ',' << thicknessin(j,1)+0.0 << '\n';
    }
    outfile6.close();  
    ofstream outfile7("height_out.csv");
    outfile7 << "Equilibrium" << ',' << "Observed" << '\n';
    for (int j = 0; j < thicknessout.n_rows; j++) {
        outfile7 << setprecision(16) << thicknessout(j,0)+0.0 << ',' << thicknessout(j,1)+0.0  << '\n';
    }
    outfile7.close();  
}

void printout_elementcCenter_onLimitSurface(mat centersin, mat centersmid, mat centersout, bool isBilayerModel){
    if ( isBilayerModel == true ){
        ofstream outfile8("triangleCentersIn.csv");
        for (int j = 0; j < centersin.n_rows; j++) {
            outfile8 << setprecision(16) << centersin(j,0)+0.0 << ',' << centersin(j,1)+0.0 << ',' << centersin(j,2)+0.0 << '\n';
        }
        outfile8.close();  
        ofstream outfile9("triangleCentersMid.csv");
        for (int j = 0; j < centersmid.n_rows; j++) {
            outfile9 << setprecision(16) << centersmid(j,0)+0.0 << ',' << centersmid(j,1)+0.0 << ',' << centersmid(j,2)+0.0 << '\n';
        }
        outfile9.close(); 
        ofstream outfile10("triangleCentersOut.csv");
        for (int j = 0; j < centersout.n_rows; j++) {
            outfile10 << setprecision(16) << centersout(j,0)+0.0 << ',' << centersout(j,1)+0.0 << ',' << centersout(j,2)+0.0 << '\n';
        }
        outfile10.close(); 
    }else{
        ofstream outfile10("triangleCenters.csv");
        for (int j = 0; j < centersout.n_rows; j++) {
            outfile10 << setprecision(16) << centersout(j,0)+0.0 << ',' << centersout(j,1)+0.0 << ',' << centersout(j,2)+0.0 << '\n';
        }
        outfile10.close(); 
    }
}
