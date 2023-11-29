#include <math.h>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include "functions_setup_triangular_mesh.cpp"

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// the following are bending energy, volume energy, and local area energy 

// bending energy and forces, area constraint energy and forces, volume constriant energy and forces
void element_energy_force_regular(bool isInnerLayer, bool isInsertionPatch, mat dots, Param param, double c0, double& Ebe, mat& F_be, mat& F_s, rowvec gqcoeff, cube shape_functions, double a0) {
    // F_be is the force related to the curvature; F_s is the force related to the area-constraint; F_v is the force related to the volume-constraint.
    // initialize output parameters
    Ebe = 0.0;
    F_be.fill(0);
    F_s.fill(0);
    //////////////////////////////////////////////////////////////
    double S0 = param.S0in; 
    double S  = param.Sin; 
    double kc = param.kc_in;
    double us = param.us_in/S0;
    if (isInnerLayer == false){
        S0 = param.S0out; 
        S  = param.Sout; 
        kc = param.kc_out;
        us = param.us_out/S0;
    }
    if ( param.isGlobalAreaConstraint == false ){
        us = us/a0;
    }

    int GaussQuadratureN = param.GaussQuadratureN;
    // Gaussian quadrature, second-order or 3 points.
    mat VWU = setVMU(GaussQuadratureN); 
    ///////////////////////////////////////////
    //rowvec gqcoeff = setVMUcoefficient(GaussQuadratureN); 
    for (int i = 0; i < VWU.n_rows; i++) {
        double ebe = 0.0;
        mat f_be(12,3); f_be.fill(0);
        mat f_cons(12,3); f_cons.fill(0);
        // 12 shape functions
        mat sf = shape_functions.slice(i);
        // a_1,2,3 covariant vectors; a1,2 contravariant vectors;
        // a_11: a_1 differential to v; a_12: a_1 differential to w;
        rowvec x(3); trans_time(sf.col(0),dots,x);
        rowvec a_1(3); trans_time(sf.col(1),dots,a_1);
        rowvec a_2(3); trans_time(sf.col(2),dots,a_2);
        rowvec a_11(3); trans_time(sf.col(3),dots,a_11);
        rowvec a_22(3); trans_time(sf.col(4),dots,a_22);
        rowvec a_12(3); trans_time(sf.col(5),dots,a_12);
        rowvec a_21(3); trans_time(sf.col(6),dots,a_21);
        rowvec xa = cross(a_1,a_2);
        double sqa = norm(xa,2);
        rowvec xa_1 = cross(a_11,a_2) + cross(a_1,a_21);
        rowvec xa_2 = cross(a_12,a_2) + cross(a_1,a_22);
        mat oneelement1 = xa*strans(xa_1);
        double sqa_1 = 1.0/sqa * oneelement1(0,0);
        mat oneelement2 = xa*strans(xa_2);
        double sqa_2 = 1.0/sqa * oneelement2(0,0);
        rowvec a_3 = xa/sqa;
        rowvec a_31 = 1.0/sqa/sqa *(xa_1*sqa-xa*sqa_1);
        rowvec a_32 = 1.0/sqa/sqa *(xa_2*sqa-xa*sqa_2);
        rowvec d = a_3;
        rowvec d_1 = a_31;
        rowvec d_2 = a_32;
        if ( isInnerLayer == true ){ // inner layer, the sign of memrbane direction is opposite! 
            d = -d; d_1 = -d_1; d_2 = -d_2;
        }
        rowvec a1 = cross(a_2,a_3)/sqa;
        rowvec a2 = cross(a_3,a_1)/sqa;
        rowvec a11 = 1.0/sqa/sqa *( (cross(a_21,a_3)+cross(a_2,a_31))*sqa - cross(a_2,a_3)*sqa_1 );
        rowvec a12 = 1.0/sqa/sqa *( (cross(a_22,a_3)+cross(a_2,a_32))*sqa - cross(a_2,a_3)*sqa_2 );
        rowvec a21 = 1.0/sqa/sqa *( (cross(a_31,a_1)+cross(a_3,a_11))*sqa - cross(a_3,a_1)*sqa_1 );
        rowvec a22 = 1.0/sqa/sqa *( (cross(a_32,a_1)+cross(a_3,a_12))*sqa - cross(a_3,a_1)*sqa_2 );
        mat shu = a1*strans(d_1) + a2*strans(d_2);
        double H_curv = 0.5*shu(0,0);
        ebe = 0.5*kc*sqa*pow(2.0*H_curv-c0,2.0);    // bending energy
        rowvec n1_be = -kc*(2.0*H_curv-c0)*(a1*strans(a1)*d_1+a1*strans(a2)*d_2) + kc*0.5*pow(2.0*H_curv-c0,2.0)*a1;
        rowvec n2_be = -kc*(2.0*H_curv-c0)*(a2*strans(a1)*d_1+a2*strans(a2)*d_2) + kc*0.5*pow(2.0*H_curv-c0,2.0)*a2;
        rowvec m1_be = kc*(2.0*H_curv-c0)*a1;
        rowvec m2_be = kc*(2.0*H_curv-c0)*a2;
        rowvec n1_cons = us*(S-S0)*a1;
        rowvec n2_cons = us*(S-S0)*a2;
        if ( param.isGlobalAreaConstraint == false ){
            n1_cons = us*(sqa-a0)*a1;
            n2_cons = us*(sqa-a0)*a2;
        }
        for (int j = 0; j < 12; j++) {
            mat da1 = -sf(j,3)*kron(strans(a1),d) - sf(j,1)*kron(strans(a11),d) - sf(j,1)*kron(strans(a1),d_1) - sf(j,6)*kron(strans(a2),d) - sf(j,2)*kron(strans(a21),d) - sf(j,2)*kron(strans(a2),d_1);
            mat da2 = -sf(j,5)*kron(strans(a1),d) - sf(j,1)*kron(strans(a12),d) - sf(j,1)*kron(strans(a1),d_2) - sf(j,4)*kron(strans(a2),d) - sf(j,2)*kron(strans(a22),d) - sf(j,2)*kron(strans(a2),d_2);
            rowvec tempf_be = n1_be*sf(j,1) + m1_be*da1 + n2_be*sf(j,2) + m2_be*da2;
            f_be.row(j) = tempf_be*sqa;
            rowvec tempf_cons = n1_cons*sf(j,1) + n2_cons*sf(j,2);
            f_cons.row(j) = tempf_cons*sqa;
        }
        Ebe = Ebe + 1.0/2.0*gqcoeff(i)*ebe;
        F_be = F_be + 1.0/2.0*gqcoeff(i)*f_be;
        F_s = F_s + 1.0/2.0*gqcoeff(i)*f_cons;
    }
}

void element_energy_force_irregularMinus(bool isInnerLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    // five matrix used for subdivision of the irregular patch
    mat M(17,11), M1(11,17), M2(12,17), M3(12,17), M4(12,17); 
    M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    // bending energy E_bending and cunchu2_constraint force
    mat ori_dots = Dots; // (11,3)
    mat temp = eye(11,11);
    for (int j = 0; j < n; j++) {
        mat newnodes17 = M*ori_dots; // 17 new nodes

        if (j != 0) {
            temp = (M1*M) * temp;
        }
        mat matrix = M*temp;

        mat dots = M4*newnodes17;    // element 4
        mat f1(12,3);  f1.fill(0.0); // f1(12,3)
        mat f2 = f1; 
        double ebe1 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E_bending = E_bending + ebe1;

        dots = M2*newnodes17;    // element 2
        f1.fill(0.0);
        f2.fill(0.0);
        double ebe2 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E_bending = E_bending + ebe2;

        dots = M3*newnodes17;    // element 3
        f1.fill(0.0);
        f2.fill(0.0);
        double ebe3 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E_bending = E_bending + ebe3;

        mat dots1 = M1*newnodes17;   // element 4, still irregular patch
        ori_dots = dots1;
    }
}

void element_energy_force_irregularPlus(bool isInnerLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    // five matrix used for subdivision of the irregular patch
    mat M(19,13), M1(12,19), M2(12,19), M3(12,19), M4(13,19); 
    M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    // bending energy E_bending and cunchu2_constraint force
    mat ori_dots = Dots; // (13,3)
    mat temp = eye(13,13);
    for (int j = 0; j < n; j++) {
        mat newnodes19 = M*ori_dots; // 19 new nodes

        if (j != 0) {
            temp = (M4*M) * temp;
        }
        mat matrix = M*temp;

        mat dots = M1*newnodes19;    // element 1
        mat f1(12,3);  f1.fill(0.0); // f1(12,3)
        mat f2 = f1; 
        double ebe1 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m1m = strans(M1*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E_bending = E_bending + ebe1;

        dots = M2*newnodes19;    // element 2
        f1.fill(0.0);
        f2.fill(0.0);
        double ebe2 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E_bending = E_bending + ebe2;

        dots = M3*newnodes19;    // element 3
        f1.fill(0.0);
        f2.fill(0.0);
        double ebe3 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E_bending = E_bending + ebe3;

        mat dots4 = M4*newnodes19;   // element 4, still irregular patch
        ori_dots = dots4;
    }
}

void element_energy_force_pseudoRegular1(bool isInnerLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    // five matrix used for subdivision of the irregular patch
    mat MM(18,12), MM1(11,18), MM2(12,18), MM3(12,17), MM4(13,18); 
    MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
    int n = param.subDivideTimes;
    mat newnodes18 = MM*Dots; // 18 new nodes
    /////////////////////////////////////////////////////////////////
    // two regular patch
    {
        mat dots = MM2*newnodes18;    // element 2
        mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
        double ebe1 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2,  gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m1m = strans(MM2*MM);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E_bending = E_bending + ebe1;
        
        dots = MM3*newnodes18;    // element 3
        double ebe3 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m3m = strans(MM3*MM);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E_bending = E_bending + ebe3;
    }
    // irregular-minus patch
    {
        mat ori_dots = MM1*newnodes18;    // element 1
        mat M(17,11), M1(11,17), M2(12,17), M3(12,17), M4(12,17); 
        M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
        mat temp = MM1*MM; //eye(11,11);
        for (int j = 0; j < n-1; j++) {
            mat newnodes17 = M*ori_dots; // 19 new nodes
            if (j != 0) {
                temp = (M1*M) * temp;
            }
            mat matrix = M*temp;

            mat dots = M4*newnodes17;    // element 4
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
            double ebe1 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M4*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes17;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes17;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E_bending = E_bending + ebe3;

            mat dots1 = M1*newnodes17;   // element 1, still irregular-minus patch
            ori_dots = dots1;
        }
    }
    // irregular-plus patch
    {
        mat ori_dots = MM4*newnodes18;    // element 4
        mat M(19,13), M1(12,19), M2(12,19), M3(12,19), M4(13,19); 
        M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
        mat temp = MM4*MM;//eye(13,13);
        for (int j = 0; j < n-1; j++) {
            mat newnodes19 = M*ori_dots; // 19 new nodes
            if (j != 0) {
                temp = (M4*M) * temp;
            }
            mat matrix = M*temp;

            mat dots = M1*newnodes19;    // element 1
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
            double ebe1 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M1*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes19;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes19;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E_bending = E_bending + ebe3;

            mat dots4 = M4*newnodes19;   // element 4, still irregular-plus patch
            ori_dots = dots4;
        }
    }
}

void element_energy_force_pseudoRegular2(bool isInnerLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    // five matrix used for subdivision of the irregular patch
    mat MM(18,12), MM1(11,18), MM2(12,18), MM3(12,18), MM4(13,18); 
    MM = subMatrix.sudoreg2M; MM1 = subMatrix.sudoreg2M1; MM2 = subMatrix.sudoreg2M2; MM3 = subMatrix.sudoreg2M3; MM4 = subMatrix.sudoreg2M4;
    int n = param.subDivideTimes;
    mat newnodes18 = MM*Dots; // 18 new nodes
    /////////////////////////////////////////////////////////////////
    // two regular patch
    {
        mat dots = MM2*newnodes18;    // element 2
        mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
        double ebe1 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2,  gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m1m = strans(MM2*MM);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E_bending = E_bending + ebe1;
        
        dots = MM3*newnodes18;    // element 3
        double ebe3 = 0.0;
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m3m = strans(MM3*MM);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E_bending = E_bending + ebe3;
    }
    // irregular patch
    {
        mat ori_dots = MM1*newnodes18;    // element 1
        mat M(17,11), M1(11,17), M2(12,17), M3(12,17), M4(12,17); 
        M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
        mat temp = MM1*MM; //eye(11,11);
        for (int j = 0; j < n-1; j++) {
            mat newnodes17 = M*ori_dots; // 19 new nodes
            if (j != 0) {
                temp = (M1*M) * temp;
            }
            mat matrix = M*temp;

            mat dots = M4*newnodes17;    // element 4
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
            double ebe1 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M4*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes17;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes17;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E_bending = E_bending + ebe3;

            mat dots1 = M1*newnodes17;   // element 1, still irregular patch
            ori_dots = dots1;
        }
    }
    // complex patch
    {
        mat ori_dots = MM4*newnodes18;    // element 4
        mat M(19,13), M1(12,19), M2(12,19), M3(12,19), M4(13,19); 
        M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
        mat temp = MM4*MM;//eye(13,13);
        for (int j = 0; j < n-1; j++) {
            mat newnodes19 = M*ori_dots; // 19 new nodes
            if (j != 0) {
                temp = (M4*M) * temp;
            }
            mat matrix = M*temp;

            mat dots = M1*newnodes19;    // element 1
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; 
            double ebe1 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M1*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes19;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes19;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(isInnerLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E_bending = E_bending + ebe3;

            mat dots4 = M4*newnodes19;   // element 4, still irregular patch
            ori_dots = dots4;
        }
    }
}

// to sum up
void element_energy_force(bool isInnerLayer, bool isInsertionPatch, Row<int> one_ring_nodesin, Param param, double Cin, double& Ebendingin, mat& fbein, mat vertexin, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0){      
    // regular patch
    if ( one_ring_nodesin(12) == -1 ) { // for regular patch
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3);  mat finS(12,3); // bending or curvature term, area term
        element_energy_force_regular(isInnerLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, gqcoeff, shape_functions, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j);
        } 

    // irregular-minus patch
    }else if ( one_ring_nodesin(12) == -2 ){
        mat dotsin(11,3);  // one ring vertices
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(11,3);  mat finS(11,3); // bending or curvature term, area term
        element_energy_force_irregularMinus(isInnerLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j);
        } 

    // irregular-plus patch
    }else if ( one_ring_nodesin(12) >= 0 ){  
        mat dotsin(13,3);  // one ring vertices
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(13,3);  mat finS(13,3); // bending or curvature term, area term
        element_energy_force_irregularPlus(isInnerLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j);
        } 
            
    // pseudo-regular patch 1
    }else if ( one_ring_nodesin(12) == -4 ){  
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3);  mat finS(12,3); // bending or curvature term, area term
        element_energy_force_pseudoRegular1(isInnerLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j);
        } 

    // pseudo-regular patch 2
    }else if ( one_ring_nodesin(12) == -5 ){  
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3);  mat finS(12,3); // bending or curvature term, area term
        element_energy_force_pseudoRegular2(isInnerLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j);
        } 
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// thickness energy and forces

void element_energy_force_thickness_regular(bool isInsertionPatch, bool isInnerLayer, mat dotsin, mat dotsout, Param param, double& E, mat& Fin, mat& Fout, rowvec gqcoeff, cube shape_functions){
    double h0 = param.thickness_in;
    double kt = param.Kthick_in/pow(h0,2.0);
    if ( isInnerLayer == false ){
        h0 = param.thickness_out;
        kt = param.Kthick_out/pow(h0,2.0);
    }
    if ( isInsertionPatch == true ){
        h0 = h0 - param.insert_dH0;
        kt = kt/pow(h0-0.5,2.0);
    }
    //double insertion_depth = param.insertion_depth;
    E = 0.0; Fout.fill(0.0); Fin.fill(0.0);
    int GaussQuadratureN = param.GaussQuadratureN;
    mat VWU = setVMU(GaussQuadratureN); 
    
    for (int i = 0; i < VWU.n_rows; i++) {
        double e = 0.0;
        mat fin(12,3); fin.fill(0);
        mat fout(12,3); fout.fill(0);
        mat sf= shape_functions.slice(i); // 12 shape functions
        // a_1,2,3 covariant vectors; a1,2 contravariant vectors;
        // inner layer
        rowvec x(3); trans_time(sf.col(0),dotsin,x);
        rowvec a_1(3); trans_time(sf.col(1),dotsin,a_1);
        rowvec a_2(3); trans_time(sf.col(2),dotsin,a_2);
        rowvec xa = cross(a_1,a_2);
        double sqa = norm(xa,2);
        rowvec a_3 = xa/sqa;
        rowvec a1in = cross(a_2,a_3)/sqa;
        rowvec a2in = cross(a_3,a_1)/sqa;
        rowvec xin = x;
        double sqain = sqa;
        rowvec din = a_3;
        // outer layer
        trans_time(sf.col(0),dotsout,x);
        trans_time(sf.col(1),dotsout,a_1);
        trans_time(sf.col(2),dotsout,a_2);
        xa = cross(a_1,a_2);
        sqa = norm(xa,2);
        a_3 = xa/sqa;
        rowvec a1out = cross(a_2,a_3)/sqa;
        rowvec a2out = cross(a_3,a_1)/sqa;
        rowvec xout = x;
        double sqaout = sqa;
        rowvec dout = a_3;
 
        rowvec h = xout - xin; 

        if ( isInnerLayer == false ){ // out-layer: the norm is chosen on the out-monolayer
            mat hnorm_matrix = h * strans(dout);
            double hnorm = hnorm_matrix(0,0);
            // thickness elasticity
            e = 0.5 * kt * pow(hnorm-h0,2.0) * (sqain+sqaout)/2.0;
            // nodal force
            for (int j = 0; j < 12; j++) {
                rowvec h_xinj = -sf(j,0)*dout;
                rowvec sqain_xinj = (sf(j,1)*a1in + sf(j,2)*a2in) * sqain;
                fin.row(j) = 0.5*kt*( 2.0*(hnorm-h0)*(sqain+sqaout)/2.0*h_xinj + pow(hnorm-h0,2.0)*sqain_xinj/2.0 );
            
                mat dout_xoutj = -sf(j,1)*kron(strans(a1out),dout) -sf(j,2)*kron(strans(a2out),dout);
                rowvec h_xoutj = sf(j,0)*dout + h*dout_xoutj;
                rowvec sqaout_xj = (sf(j,1)*a1out + sf(j,2)*a2out) * sqaout;
                fout.row(j) = 0.5*kt*( 2.0*(hnorm-h0)*(sqain+sqaout)/2.0*h_xoutj + pow(hnorm-h0,2.0)*sqaout_xj/2.0 );
            }
        }else{                     // in-layer: the norm is chosen on the in-monolayer
            mat hnorm_matrix = h * strans(din);
            double hnorm = hnorm_matrix(0,0);
            // thickness elasticity
            e = 0.5 * kt * pow(hnorm-h0,2.0) * (sqain+sqaout)/2.0;
            // nodal force
            for (int j = 0; j < 12; j++) {
                mat din_xinj = -sf(j,1)*kron(strans(a1in),din) -sf(j,2)*kron(strans(a2in),din);
                rowvec h_xinj = -sf(j,0)*din + h*din_xinj;
                rowvec sqain_xinj = (sf(j,1)*a1in + sf(j,2)*a2in) * sqain;
                fin.row(j) = 0.5*kt*( 2.0*(hnorm-h0)*(sqain+sqaout)/2.0*h_xinj + pow(hnorm-h0,2.0)*sqain_xinj/2.0 );
            
                rowvec h_xoutj = sf(j,0)*din;
                rowvec sqaout_xj = (sf(j,1)*a1out + sf(j,2)*a2out) * sqaout;
                fout.row(j) = 0.5*kt*( 2.0*(hnorm-h0)*(sqain+sqaout)/2.0*h_xoutj + pow(hnorm-h0,2.0)*sqaout_xj/2.0 );
            }
        }
        /////////////////////////////////////////////////////////////
        E = E + 1.0/2.0*gqcoeff(i)*e;
        Fin = Fin + 1.0/2.0*gqcoeff(i)*fin;
        Fout = Fout + 1.0/2.0*gqcoeff(i)*fout;
    }
}

void element_energy_force_thickness_irregularMinus(bool isInsertionPatch, bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the irregular patch
    mat M(17,11), M1(11,17), M2(12,17), M3(12,17), M4(12,17); 
    M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
    int n = param.subDivideTimes;
    //int i = facenum;
    /////////////////////////////////////////////////////////////////
    mat temp = eye(11,11);
    for (int j = 0; j < n; j++) {
        mat newnodes17in = M*ori_dotsin; // 17 new nodes
        mat newnodes17out = M*ori_dotsout; // 17 new nodes

        if (j != 0) {
            temp = (M1*M) * temp;
        }
        mat matrix = M*temp;

        mat dotsin = M4*newnodes17in;    // element 4
        mat dotsout = M4*newnodes17out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes17in;    // element 2
        dotsout = M2*newnodes17out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes17in;    // element 3
        dotsout = M3*newnodes17out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E = E + e;

        mat dots1in = M1*newnodes17in;   // element 1, still irregularMinus patch
        mat dots1out = M1*newnodes17out;
        ori_dotsin = dots1in;
        ori_dotsout = dots1out;
    }
}

void element_energy_force_thickness_irregularPlus(bool isInsertionPatch, bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the complex patch
    mat M(19,13), M1(12,19), M2(12,19), M3(12,19), M4(13,19); 
    M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat temp = eye(13,13);
    for (int j = 0; j < n; j++) {
        mat newnodes19in = M*ori_dotsin; // 19 new nodes
        mat newnodes19out = M*ori_dotsout; // 19 new nodes
        if (j != 0) {
            temp = (M4*M) * temp;
        }
        mat matrix = M*temp;

        mat dotsin = M1*newnodes19in;    // element 1
        mat dotsout = M1*newnodes19out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M1*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes19in;    // element 2
        dotsout = M2*newnodes19out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes19in;    // element 3
        dotsout = M3*newnodes19out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E = E + e;

        mat dots4in = M4*newnodes19in;   // element 4, still irregularPlus patch
        mat dots4out = M4*newnodes19out;
        ori_dotsin = dots4in;
        ori_dotsout = dots4out;
    }
}

void element_energy_force_thickness_pseudoRegular1(bool isInsertionPatch, bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the sudoregular patch 1
    mat Mf(18,12), M1f(11,18), M2f(12,18), M3f(12,18), M4f(13,18); 
    Mf = subMatrix.sudoreg1M; M1f = subMatrix.sudoreg1M1; M2f = subMatrix.sudoreg1M2; M3f = subMatrix.sudoreg1M3; M4f = subMatrix.sudoreg1M4;
    // five matrix used for subdivision of the irregular patch
    mat Mi(17,11), M1i(11,17), M2i(12,17), M3i(12,17), M4i(12,17); 
    Mi = subMatrix.irregM; M1i = subMatrix.irregM1; M2i = subMatrix.irregM2; M3i = subMatrix.irregM3; M4i = subMatrix.irregM4;
    // five matrix used for subdivision of the complex patch
    mat Mc(19,13), M1c(12,19), M2c(12,19), M3c(12,19), M4c(13,19); 
    Mc = subMatrix.comregM; M1c = subMatrix.comregM1; M2c = subMatrix.comregM2; M3c = subMatrix.comregM3; M4c = subMatrix.comregM4; 
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat newnodes18in = Mf*ori_dotsin; // 18 new nodes
    mat newnodes18out = Mf*ori_dotsout; // 18 new nodes
    // two regular patches
    {
        mat dotsin = M2f*newnodes18in;    // element 2
        mat dotsout = M2f*newnodes18out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M2f*Mf);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M3f*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3f*Mf);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E = E + e;
    }   
    // irregular-Minus patch
    {
        mat ori_dotsin1 = M1f*newnodes18in;
        mat ori_dotsout1 = M1f*newnodes18out;
        mat temp = M1f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes17in = Mi*ori_dotsin1; // 17 new nodes
            mat newnodes17out = Mi*ori_dotsout1; // 17 new nodes
            if (j != 0) {
                temp = (M1i*Mi) * temp;
            }
            mat matrix = Mi*temp;

            mat dotsin = M4i*newnodes17in;    // element 4
            mat dotsout = M4i*newnodes17out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m4m = strans(M4i*matrix);
            cunchu1 = cunchu1 + m4m*f1;
            cunchu2 = cunchu2 + m4m*f2;
            E = E + e;

            dotsin = M2i*newnodes17in;    // element 2
            dotsout = M2i*newnodes17out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2i*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3i*newnodes17in;    // element 3
            dotsout = M3i*newnodes17out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3i*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E = E + e;

            mat dots1in = M1i*newnodes17in;   // element 1, still irregular-Minus patch
            mat dots1out = M1i*newnodes17out;
            ori_dotsin1 = dots1in;
            ori_dotsout1 = dots1out;
        }
    }
    // irregular-plus patch
    {
        mat ori_dotsin1 = M4f*newnodes18in;
        mat ori_dotsout1 = M4f*newnodes18out;
        mat temp = M4f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes19in = Mc*ori_dotsin1; // 19 new nodes
            mat newnodes19out = Mc*ori_dotsout1; // 19 new nodes
            if (j != 0) {
                temp = (M4c*Mc) * temp;
            }
            mat matrix = Mc*temp;

            mat dotsin = M1c*newnodes19in;    // element 1
            mat dotsout = M1c*newnodes19out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m1m = strans(M1c*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E = E + e;

            dotsin = M2c*newnodes19in;    // element 2
            dotsout = M2c*newnodes19out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2c*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3c*newnodes19in;    // element 3
            dotsout = M3c*newnodes19out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3c*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E = E + e;

            mat dots4in = M4c*newnodes19in;   // element 4, still irregular-plus patch
            mat dots4out = M4c*newnodes19out;
            ori_dotsin1 = dots4in;
            ori_dotsout1 = dots4out;
        }
    }
}

void element_energy_force_thickness_pseudoRegular2(bool isInsertionPatch, bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);

    // five matrix used for subdivision of the fakeregular patch
    mat Mf(18,12), M1f(11,18), M2f(12,18), M3f(12,18), M4f(13,18); 
    Mf = subMatrix.sudoreg2M; M1f = subMatrix.sudoreg2M1; M2f = subMatrix.sudoreg2M2; M3f = subMatrix.sudoreg2M3; M4f = subMatrix.sudoreg2M4;
    // five matrix used for subdivision of the irregular patch
    mat Mi(17,11), M1i(11,17), M2i(12,17), M3i(12,17), M4i(12,17); 
    Mi = subMatrix.irregM; M1i = subMatrix.irregM1; M2i = subMatrix.irregM2; M3i = subMatrix.irregM3; M4i = subMatrix.irregM4;
    // five matrix used for subdivision of the complex patch
    mat Mc(19,13), M1c(12,19), M2c(12,19), M3c(12,19), M4c(13,19); 
    Mc = subMatrix.comregM; M1c = subMatrix.comregM1; M2c = subMatrix.comregM2; M3c = subMatrix.comregM3; M4c = subMatrix.comregM4; 
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat newnodes18in = Mf*ori_dotsin; // 18 new nodes
    mat newnodes18out = Mf*ori_dotsout; // 18 new nodes
    // two regular patches
    {
        mat dotsin = M2f*newnodes18in;    // element 2
        mat dotsout = M2f*newnodes18out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M2f*Mf);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M3f*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e = 0.0;
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3f*Mf);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E = E + e;
    }  

    // irregular-minus patch
    {
        mat ori_dotsin1 = M1f*newnodes18in;
        mat ori_dotsout1 = M1f*newnodes18out;
        mat temp = M1f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes17in = Mi*ori_dotsin1; // 17 new nodes
            mat newnodes17out = Mi*ori_dotsout1; // 17 new nodes
            if (j != 0) {
                temp = (M1i*Mi) * temp;
            }
            mat matrix = Mi*temp;

            mat dotsin = M4i*newnodes17in;    // element 4
            mat dotsout = M4i*newnodes17out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m4m = strans(M4i*matrix);
            cunchu1 = cunchu1 + m4m*f1;
            cunchu2 = cunchu2 + m4m*f2;
            E = E + e;

            dotsin = M2i*newnodes17in;    // element 2
            dotsout = M2i*newnodes17out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2i*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3i*newnodes17in;    // element 3
            dotsout = M3i*newnodes17out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3i*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E = E + e;

            mat dots1in = M1i*newnodes17in;   // element 1, still irregular-minus patch
            mat dots1out = M1i*newnodes17out;
            ori_dotsin1 = dots1in;
            ori_dotsout1 = dots1out;
        }
    }
    // irregular-plus patch
    {
        mat ori_dotsin1 = M4f*newnodes18in;
        mat ori_dotsout1 = M4f*newnodes18out;
        mat temp = M4f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes19in = Mc*ori_dotsin1; // 19 new nodes
            mat newnodes19out = Mc*ori_dotsout1; // 19 new nodes
            if (j != 0) {
                temp = (M4c*Mc) * temp;
            }
            mat matrix = Mc*temp;

            mat dotsin = M1c*newnodes19in;    // element 1
            mat dotsout = M1c*newnodes19out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m1m = strans(M1c*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E = E + e;

            dotsin = M2c*newnodes19in;    // element 2
            dotsout = M2c*newnodes19out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2c*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3c*newnodes19in;    // element 3
            dotsout = M3c*newnodes19out;
            e = 0.0;
            element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3c*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            E = E + e;

            mat dots4in = M4c*newnodes19in;   // element 4, still irregular-plus patch
            mat dots4out = M4c*newnodes19out;
            ori_dotsin1 = dots4in;
            ori_dotsout1 = dots4out;
        }
    }
}

// to sum up
// for out-layer: vertex1 is the mid-surface, vertex2 is the out-surface
// for in-layer: vertex1 is the in-surface, vertex2 is the mid-surface
void element_energy_force_thickness(bool isInsertionPatch, bool isInnerLayer, Row<int> one_ring_nodes, Param param, double& Ethick, mat& fthick, mat vertex1, mat vertex2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix){
    int numvertex = vertex1.n_rows;
    // regular patch
    if ( one_ring_nodes(12) == -1 ) { 
        mat dots1(12,3); 
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots1.row(j) = vertex1.row(nodenum);
        }
        mat dots2(12,3); // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots2.row(j) = vertex2.row(nodenum);
        }
        double ethick = 0.0; mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_regular(isInsertionPatch, isInnerLayer, dots1, dots2, param, ethick, fH1, fH2, gqcoeff, shape_functions);
        if ( isInnerLayer == false ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else{
            for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH2.row(j);
            } 
        }
        Ethick = ethick;

    // irregular-minus patch
    }else if ( one_ring_nodes(12) == -2 ) { 
        mat dots1(11,3); 
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodes(j);
            dots1.row(j) = vertex1.row(nodenum);
        }
        mat dots2(11,3); // one ring vertices
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodes(j);
            dots2.row(j) = vertex2.row(nodenum);
        }
        double ethick = 0.0; mat fH1(11,3); mat fH2(11,3); 
        element_energy_force_thickness_irregularMinus(isInsertionPatch, isInnerLayer, dots1, dots2, param, ethick, fH1, fH2, gqcoeff, shape_functions, subMatrix);
        if ( isInnerLayer == false ){
            for (int j = 0; j < 11; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 11; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else{
             for (int j = 0; j < 11; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH1.row(j);
            } 
            for (int j = 0; j < 11; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH2.row(j);
            } 
        }
        Ethick = ethick;
        
    // irregular-plus patch
    }else if ( one_ring_nodes(12) >= 0 ) { 
        mat dots1(13,3); 
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodes(j);
            dots1.row(j) = vertex1.row(nodenum);
        }
        mat dots2(13,3); // one ring vertices
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodes(j);
            dots2.row(j) = vertex2.row(nodenum);
        }
        double ethick = 0.0; mat fH1(13,3); mat fH2(13,3); 
        element_energy_force_thickness_irregularPlus(isInsertionPatch, isInnerLayer, dots1, dots2, param, ethick, fH1, fH2, gqcoeff, shape_functions, subMatrix);
        if ( isInnerLayer == false ){
            for (int j = 0; j < 13; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 13; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else{
             for (int j = 0; j < 13; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH1.row(j);
            } 
            for (int j = 0; j < 13; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH2.row(j);
            } 
        }
        Ethick = ethick;
          
    // pseudo-regular patch 1
    }else if ( one_ring_nodes(12) == -4 ) { 
        mat dots1(12,3); 
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots1.row(j) = vertex1.row(nodenum);
        }
        mat dots2(12,3); // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots2.row(j) = vertex2.row(nodenum);
        }
        double ethick = 0.0; mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_pseudoRegular1(isInsertionPatch, isInnerLayer, dots1, dots2, param, ethick, fH1, fH2, gqcoeff, shape_functions, subMatrix);
        if ( isInnerLayer == false ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else{
             for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH2.row(j);
            } 
        } 
        Ethick = ethick;

    // pseudo-regular patch 2
    }else if ( one_ring_nodes(12) == -5 ) { 
        mat dots1(12,3); 
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots1.row(j) = vertex1.row(nodenum);
        }
        mat dots2(12,3); // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodes(j);
            dots2.row(j) = vertex2.row(nodenum);
        }
        double ethick = 0.0; mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_pseudoRegular2(isInsertionPatch, isInnerLayer, dots1, dots2, param, ethick, fH1, fH2, gqcoeff, shape_functions, subMatrix);
                if ( isInnerLayer == false ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else{
             for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH2.row(j);
            } 
        }  
        Ethick = ethick;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
rowvec getdout(mat dotsout, mat sf){
        rowvec x(3); trans_time(sf.col(0),dotsout,x);
        rowvec a_1(3); trans_time(sf.col(1),dotsout,a_1);
        rowvec a_2(3); trans_time(sf.col(2),dotsout,a_2);
        rowvec xa = cross(a_1,a_2);
        double sqa = norm(xa,2);
        rowvec a_3 = xa/sqa;
        rowvec a1out = cross(a_2,a_3)/sqa;
        rowvec a2out = cross(a_3,a_1)/sqa;
        rowvec xout = x;
        double sqaout = sqa;
        rowvec dout = a_3;
        return dout;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// tilt energy and forces
/*
void element_energy_force_tilt_regulars(bool isInnerLayer, mat dotsin, mat dotsout, Param param, double& E, mat& Fin, mat& Fout, rowvec gqcoeff, cube shape_functions){
    double k = param.Ktilt;
    //double insertion_depth = param.insertion_depth;
    E = 0.0; Fout.fill(0.0); Fin.fill(0.0);
    int GaussQuadratureN = param.GaussQuadratureN;
    mat VWU = setVMU(GaussQuadratureN); 
    for (int i = 0; i < VWU.n_rows; i++) {
        double e = 0.0;
        mat fin(12,3); fin.fill(0);
        mat fout(12,3); fout.fill(0);
        mat sf= shape_functions.slice(i); // 12 shape functions
        // a_1,2,3 covariant vectors; a1,2 contravariant vectors;
        // inner layer
        rowvec xin(3); trans_time(sf.col(0),dotsin,xin);
        // outer layer
        rowvec xout(3); trans_time(sf.col(0),dotsout,xout);
        rowvec a_1(3); trans_time(sf.col(1),dotsout,a_1);
        rowvec a_2(3); trans_time(sf.col(2),dotsout,a_2);
        rowvec xa = cross(a_1,a_2);
        double sqa = norm(xa,2);
        rowvec a_3 = xa/sqa;
        rowvec a1out = cross(a_2,a_3)/sqa;
        rowvec a2out = cross(a_3,a_1)/sqa;
        double sqaout = sqa;
        rowvec dout = a_3;
        if ( isInnerLayer == true ){
            dout = -dout;
        }
        /////////////////////////////////////////////////////////////
        // energy
        rowvec h = xout - xin; 
        double hnorm = norm(h,2.0);
        rowvec n = h / hnorm;
        mat ndoutshu = n * strans(dout); double ndout = ndoutshu(0,0);
        rowvec tout = n/ndout - dout; 
        mat eshu = 0.5*k*tout*strans(tout)*sqaout;// element tilt-energy
        e = eshu(0,0);
        /////////////////////////////////////////////////////////////
        // nodal force
        mat I = eye(3,3);
        for (int j = 0; j < 12; j++) { 
            mat dout_xoutj = - sf(j,1)*kron(strans(a1out),dout) - sf(j,2)*kron(strans(a2out),dout);
            mat n_xoutj = sf(j,0)/hnorm * ( I - kron(strans(h),h)/pow(hnorm,2.0));
            rowvec tmp = dout*n_xoutj + n*dout_xoutj;
            mat tout_xoutj = - 1.0/ndout/ndout*kron(strans(n),tmp) + 1.0/ndout*n_xoutj - dout_xoutj;
            rowvec sqaout_xoutj = (sf(j,1)*a1out + sf(j,2)*a2out) * sqaout;
            fout.row(j) = 0.5*k*( 2.0*sqaout*tout*tout_xoutj + tout*strans(tout)*sqaout_xoutj );

            mat n_xinj = - n_xoutj;
            tmp = dout*n_xinj;
            mat tout_xinj = - 1.0/ndout/ndout*kron(strans(n),tmp) + 1.0/ndout*n_xinj;
            fin.row(j) = 0.5*k*( 2.0*sqaout*tout*tout_xinj );
        }
        /////////////////////////////////////////////////////////////
        E = E + 1.0/2.0*gqcoeff(i)*e;
        Fin = Fin + 1.0/2.0*gqcoeff(i)*fin;
        Fout = Fout + 1.0/2.0*gqcoeff(i)*fout;
    }
}

void element_energy_force_tilt_irregulars(bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the irregular patch
    mat M(17,11), M1(11,17), M2(12,17), M3(12,17), M4(12,17); 
    M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
    int n = param.subDivideTimes;
    //int i = facenum;
    /////////////////////////////////////////////////////////////////
    mat temp = eye(11,11);
    for (int j = 0; j < n; j++) {
        mat newnodes17in = M*ori_dotsin; // 17 new nodes
        mat newnodes17out = M*ori_dotsout; // 17 new nodes

        if (j != 0) {
            temp = (M1*M) * temp;
        }
        mat matrix = M*temp;

        mat dotsin = M4*newnodes17in;    // element 4
        mat dotsout = M4*newnodes17out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes17in;    // element 2
        dotsout = M2*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes17in;    // element 3
        dotsout = M3*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        E = E + e;

        mat dots1in = M1*newnodes17in;   // element 1, still irregular patch
        mat dots1out = M1*newnodes17out;
        ori_dotsin = dots1in;
        ori_dotsout = dots1out;
    }
}

void element_energy_force_tilt_regular_irregular(bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the regular patch
    mat M(18,12), M1(12,18), M2(12,18), M3(12,18), M4(12,18); 
    M = subMatrix.regM; M1 = subMatrix.regM1; M2 = subMatrix.regM2; M3 = subMatrix.regM3; M4 = subMatrix.regM4;
    // five matrix used for subdivision of the irregular patch
    mat Mi(17,11), M1i(11,17), M2i(12,17), M3i(12,17), M4i(12,17); 
    Mi = subMatrix.irregM; M1i = subMatrix.irregM1; M2i = subMatrix.irregM2; M3i = subMatrix.irregM3; M4i = subMatrix.irregM4;
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat tempin = eye(12,12);
    mat tempout = eye(11,11);
    for (int j = 0; j < n; j++) {
        mat newnodes18in = M*ori_dotsin; // 18 new nodes
        mat newnodes17out = Mi*ori_dotsout; // 17 new nodes
        if (j != 0) {
            tempin = (M1*M) * tempin;
            tempout = (M1i*Mi) * tempout;
        }
        mat matrixin = M*tempin;
        mat matrixout = Mi*tempout;

        mat dotsin = M4*newnodes18in;    // element 4
        mat dotsout = M4i*newnodes17out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m4m = strans(M4*matrixin);
        cunchu1 = cunchu1 + m4m*f1;
        mat m4mi = strans(M4i*matrixout);
        cunchu2 = cunchu2 + m4mi*f2;
        E = E + e;

        dotsin = M2*newnodes18in;    // element 2
        dotsout = M2i*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrixin);
        cunchu1 = cunchu1 + m2m*f1;
        mat m2mi = strans(M2i*matrixout);
        cunchu2 = cunchu2 + m2mi*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3i*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*matrixin);
        cunchu1 = cunchu1 + m3m*f1;
        mat m3mi = strans(M3i*matrixout);
        cunchu2 = cunchu2 + m3mi*f2;
        E = E + e;

        mat dots1in = M1*newnodes18in;   // element 1, still irregular patch
        mat dots1out = M1i*newnodes17out;
        ori_dotsin = dots1in;
        ori_dotsout = dots1out;
    }
}

void element_energy_force_tilt_regular_complex(bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the regular patch
    mat M(18,12), M1(12,18), M2(12,18), M3(12,18), M4(12,18); 
    M = subMatrix.regM; M1 = subMatrix.regM1; M2 = subMatrix.regM2; M3 = subMatrix.regM3; M4 = subMatrix.regM4;
    // five matrix used for subdivision of the complex patch
    mat Mi(19,13), M1i(12,19), M2i(12,19), M3i(12,19), M4i(13,19); 
    Mi = subMatrix.comregM; M1i = subMatrix.comregM1; M2i = subMatrix.comregM2; M3i = subMatrix.comregM3; M4i = subMatrix.comregM4;
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat tempin = eye(12,12);
    mat tempout = eye(13,13);
    for (int j = 0; j < n; j++) {
        mat newnodes18in = M*ori_dotsin; // 18 new nodes
        mat newnodes19out = Mi*ori_dotsout; // 19 new nodes
        if (j != 0) {
            tempin = (M4*M) * tempin;
            tempout = (M4i*Mi) * tempout;
        }
        mat matrixin = M*tempin;
        mat matrixout = Mi*tempout;

        mat dotsin = M1*newnodes18in;    // element 1
        mat dotsout = M1i*newnodes19out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M1*matrixin);
        cunchu1 = cunchu1 + m1m*f1;
        mat m1mi = strans(M1i*matrixout);
        cunchu2 = cunchu2 + m1mi*f2;
        E = E + e;

        dotsin = M2*newnodes18in;    // element 2
        dotsout = M2i*newnodes19out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrixin);
        cunchu1 = cunchu1 + m2m*f1;
        mat m2mi = strans(M2i*matrixout);
        cunchu2 = cunchu2 + m2mi*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3i*newnodes19out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*matrixin);
        cunchu1 = cunchu1 + m3m*f1;
        mat m3mi = strans(M3i*matrixout);
        cunchu2 = cunchu2 + m3mi*f2;
        E = E + e;

        mat dots4in = M4*newnodes18in;   // element 4, still irregular patch
        mat dots4out = M4i*newnodes19out;
        ori_dotsin = dots4in;
        ori_dotsout = dots4out;
    }
}

void element_energy_force_tilt_regular_fakeregular(bool isInnerLayer, mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E = 0.0;
    cunchu1.fill(0);
    cunchu2.fill(0);
    // five matrix used for subdivision of the regular patch
    mat M(18,12), M1(12,18), M2(12,18), M3(12,18), M4(12,18); 
    M = subMatrix.regM; M1 = subMatrix.regM1; M2 = subMatrix.regM2; M3 = subMatrix.regM3; M4 = subMatrix.regM4;
    // five matrix used for subdivision of the fakeregular patch
    mat Mf(18,12), M1f(11,18), M2f(12,18), M3f(12,18), M4f(13,18); 
    Mf = subMatrix.fakeregM; M1f = subMatrix.fakeregM1; M2f = subMatrix.fakeregM2; M3f = subMatrix.fakeregM3; M4f = subMatrix.fakeregM4;
    // five matrix used for subdivision of the irregular patch
    mat Mi(17,11), M1i(11,17), M2i(12,17), M3i(12,17), M4i(12,17); 
    Mi = subMatrix.irregM; M1i = subMatrix.irregM1; M2i = subMatrix.irregM2; M3i = subMatrix.irregM3; M4i = subMatrix.irregM4;
    // five matrix used for subdivision of the complex patch
    mat Mc(19,13), M1c(12,19), M2c(12,19), M3c(12,19), M4c(13,19); 
    Mc = subMatrix.comregM; M1c = subMatrix.comregM1; M2c = subMatrix.comregM2; M3c = subMatrix.comregM3; M4c = subMatrix.comregM4; 
    int n = param.subDivideTimes;
    /////////////////////////////////////////////////////////////////
    mat newnodes18in = M*ori_dotsin; // 18 new nodes
    mat newnodes18out = Mf*ori_dotsout; // 18 new nodes
    // two regular patches
    {
        mat dotsin = M2*newnodes18in;    // element 2
        mat dotsout = M2f*newnodes18out;
        mat f1(12,3);  f1.fill(0); mat f2 = f1; 
        double e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M2*M);
        cunchu1 = cunchu1 + m1m*f1;
        mat m1mf = strans(M2f*Mf);
        cunchu2 = cunchu2 + m1mf*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e = 0.0;
        element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m3m = strans(M3*M);
        cunchu1 = cunchu1 + m3m*f1;
        mat m3mf = strans(M3f*Mf);
        cunchu2 = cunchu2 + m3mf*f2;
        E = E + e;
    }   
    // irregular patch
    {
        mat ori_dotsin1 = M1*newnodes18in;
        mat ori_dotsout1 = M1f*newnodes18out;
        mat tempin = M1*M;
        mat tempout = M1f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes18in = M*ori_dotsin1; // 18 new nodes
            mat newnodes17out = Mi*ori_dotsout1; // 17 new nodes
            if (j != 0) {
                tempin = (M1*M) * tempin;
                tempout = (M1i*Mi) * tempout;
            }
            mat matrixin = M*tempin;
            mat matrixout = Mi*tempout;

            mat dotsin = M4*newnodes18in;    // element 4
            mat dotsout = M4i*newnodes17out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m4m = strans(M4*matrixin);
            cunchu1 = cunchu1 + m4m*f1;
            mat m4mi = strans(M4i*matrixout);
            cunchu2 = cunchu2 + m4mi*f2;
            E = E + e;

            dotsin = M2*newnodes18in;    // element 2
            dotsout = M2i*newnodes17out;
            e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2*matrixin);
            cunchu1 = cunchu1 + m2m*f1;
            mat m2mi = strans(M2i*matrixout);
            cunchu2 = cunchu2 + m2mi*f2;
            E = E + e;

            dotsin = M3*newnodes18in;    // element 3
            dotsout = M3i*newnodes17out;
            e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3*matrixin);
            cunchu1 = cunchu1 + m3m*f1;
            mat m3mi = strans(M3i*matrixout);
            cunchu2 = cunchu2 + m3mi*f2;
            E = E + e;

            mat dots1in = M1*newnodes18in;   // element 1, still irregular patch
            mat dots1out = M1i*newnodes17out;
            ori_dotsin1 = dots1in;
            ori_dotsout1 = dots1out;
        }
    }
    // complex patch
    {
        mat ori_dotsin1 = M4*newnodes18in;
        mat ori_dotsout1 = M4f*newnodes18out;
        mat tempin = M4*M;
        mat tempout = M4f*Mf;
        for (int j = 0; j < n-1; j++) {
            mat newnodes18in = M*ori_dotsin1; // 18 new nodes
            mat newnodes19out = Mc*ori_dotsout1; // 19 new nodes
            if (j != 0) {
                tempin = (M4*M) * tempin;
                tempout = (M4c*Mc) * tempout;
            }
            mat matrixin = M*tempin;
            mat matrixout = Mc*tempout;

            mat dotsin = M1*newnodes18in;    // element 1
            mat dotsout = M1c*newnodes19out;
            mat f1(12,3);  f1.fill(0); mat f2 = f1; 
            double e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m1m = strans(M1*matrixin);
            cunchu1 = cunchu1 + m1m*f1;
            mat m1mc = strans(M1c*matrixout);
            cunchu2 = cunchu2 + m1mc*f2;
            E = E + e;

            dotsin = M2*newnodes18in;    // element 2
            dotsout = M2c*newnodes19out;
            e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2*matrixin);
            cunchu1 = cunchu1 + m2m*f1;
            mat m2mc = strans(M2c*matrixout);
            cunchu2 = cunchu2 + m2mc*f2;
            E = E + e;

            dotsin = M3*newnodes18in;    // element 3
            dotsout = M3c*newnodes19out;
            e = 0.0;
            element_energy_force_tilt_regulars(isInnerLayer, dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m3m = strans(M3*matrixin);
            cunchu1 = cunchu1 + m3m*f1;
            mat m3mc = strans(M3c*matrixout);
            cunchu2 = cunchu2 + m3mc*f2;
            E = E + e;

            mat dots4in = M4*newnodes18in;   // element 4, still irregular patch
            mat dots4out = M4c*newnodes19out;
            ori_dotsin1 = dots4in;
            ori_dotsout1 = dots4out;
        }
    }
}
*/

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// rgularization energy and forces
void energy_force_regularization(bool isInnerLayer, mat vertex, mat vertexold, Mat<int> face, Param param, Row<int> isinsertionpatch, double& E_regularization, mat& Fre, rowvec& deformnumbers){
    double k  = param.k_regularization;
    //Fre.fill(0.0);
    E_regularization = 0.0;
    deformnumbers.fill(0);
    int deformnumber_shape = 0;
    int deformnumber_area = 0;
    rowvec Ere(face.n_rows); Ere.fill(0.0);
    //cube fre(vertex.n_rows,3,face.n_rows);
    mat fre(vertex.n_rows,3); fre.fill(0.0);
    #pragma omp parallel for reduction(+:fre)
    for (int i = 0; i < face.n_rows; i++){    
        bool isInsertionPatch = false;
        if ( isinsertionpatch(i) == 1 && isInnerLayer == false){
            isInsertionPatch = true; 
            continue;
        }
        int node0 = face(i,0); // three nodes of this face element
        int node1 = face(i,1);
        int node2 = face(i,2);
        rowvec vector0 = vertex.row(node0) - vertex.row(node1);  double side0 = norm(vector0,2.0);
        rowvec vector1 = vertex.row(node1) - vertex.row(node2);  double side1 = norm(vector1,2.0);
        rowvec vector2 = vertex.row(node2) - vertex.row(node0);  double side2 = norm(vector2,2.0);
        double s = (side0 + side1 + side2)/2.0;
        double area = sqrt( s*(s-side0)*(s-side1)*(s-side2) );
        double meanside = 1.0/3.0*(side0 + side1+ side2);
        double gama = 1.0/pow(meanside,2.0) * ( pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0) );
        rowvec vectorold0 = vertexold.row(node0) - vertexold.row(node1);  double sideold0 = norm(vectorold0,2.0);
        rowvec vectorold1 = vertexold.row(node1) - vertexold.row(node2);  double sideold1 = norm(vectorold1,2.0);
        rowvec vectorold2 = vertexold.row(node2) - vertexold.row(node0);  double sideold2 = norm(vectorold2,2.0);
        s = (sideold0 + sideold1 + sideold2)/2.0;
        double areaold = sqrt( s*(s-sideold0)*(s-sideold1)*(s-sideold2) ); //double areaold = S0/face.n_rows;
            
        bool isDeformShape = false;
        if ( gama > param.gama_shape && param.usingRpi == true ){
            isDeformShape = true;
        }
        bool isDeformArea = false;
        double a0 = areaold; 
        if ( abs(area-a0)/a0 >= param.gama_area && param.usingRpi == true ){
            isDeformArea = true;
        }
        
        if ( param.duringStepsToIncreaseInsertDepth == true ){
            deformnumber_shape ++;
            double ktmp = 0.0; // k / 10.0; 
            double meansideold = sqrt( 4.0*area/sqrt(3.0) );
            Ere(i) = ktmp /2.0*(pow(side0-meansideold,2.0) + pow(side1-meansideold,2.0) + pow(side2-meansideold,2.0));
            fre.row(node0) = fre.row(node0) + ktmp*( (side0-meansideold)*(-vector0/side0) + (side2-meansideold)*(vector2/side2) );
            fre.row(node1) = fre.row(node1) + ktmp*( (side1-meansideold)*(-vector1/side1) + (side0-meansideold)*(vector0/side0) );
            fre.row(node2) = fre.row(node2) + ktmp*( (side2-meansideold)*(-vector2/side2) + (side1-meansideold)*(vector1/side1) );
        }else if ( isDeformShape == false && isDeformArea == false ){
            Ere(i) = k/2.0*(pow(side0-sideold0,2.0) + pow(side1-sideold1,2.0) + pow(side2-sideold2,2.0));
            fre.row(node0) = fre.row(node0) + k*( (side0-sideold0)*(-vector0/side0) + (side2-sideold2)*(vector2/side2) );
            fre.row(node1) = fre.row(node1) + k*( (side1-sideold1)*(-vector1/side1) + (side0-sideold0)*(vector0/side0) );
            fre.row(node2) = fre.row(node2) + k*( (side2-sideold2)*(-vector2/side2) + (side1-sideold1)*(vector1/side1) );
        }else if ( isDeformArea == true ){
            deformnumber_area ++;
            double meanside = sqrt( 4.0*a0/sqrt(3.0) );
            Ere(i) = k/2.0*(pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0));
            fre.row(node0) = fre.row(node0) + k*( (side0-meanside)*(-vector0/side0) + (side2-meanside)*(vector2/side2) );
            fre.row(node1) = fre.row(node1) + k*( (side1-meanside)*(-vector1/side1) + (side0-meanside)*(vector0/side0) );
            fre.row(node2) = fre.row(node2) + k*( (side2-meanside)*(-vector2/side2) + (side1-meanside)*(vector1/side1) );
        }else if ( isDeformShape == true && isDeformArea == false ){
            deformnumber_shape ++;
            double meansideold = sqrt( 4.0*area/sqrt(3.0) );
            Ere(i) = k/2.0*(pow(side0-meansideold,2.0) + pow(side1-meansideold,2.0) + pow(side2-meansideold,2.0));
            fre.row(node0) = fre.row(node0) + k*( (side0-meansideold)*(-vector0/side0) + (side2-meansideold)*(vector2/side2) );
            fre.row(node1) = fre.row(node1) + k*( (side1-meansideold)*(-vector1/side1) + (side0-meansideold)*(vector0/side0) );
            fre.row(node2) = fre.row(node2) + k*( (side2-meansideold)*(-vector2/side2) + (side1-meansideold)*(vector1/side1) );
        }
    }
    #pragma omp parallel for reduction(+:E_regularization)
    for (int i = 0; i < face.n_rows; i++){
        E_regularization = E_regularization + Ere(i);
    }
    Fre = fre;
    /////////////////////////////////////////////////////////////
    deformnumbers << deformnumber_area << deformnumber_shape << face.n_rows -deformnumber_area-deformnumber_shape << endr;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// here, A is the one-ring-nodes
void plane_area(mat vertex, Mat<int> face, Mat<int> isBoundaryFace, Mat<int> A, Row<int> Isinsertionpatch, int GaussQuadratureN, rowvec& element_area, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, int n){ // A is face_ring_vertex 
    element_area.fill(0);
    mat VWU = setVMU(GaussQuadratureN); 

    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        if ( isBoundaryFace(i) == 1 )
            continue;
            
        double area = 0.0;

        // regular patch
        if  ( A(i,12) == -1 ){ 
            mat dots(12,3); //dots(12,3); 12 nodes
            for (int j = 0; j < 12; j++){
                int nodenum = A(i,j);
                dots.row(j) = vertex.row(nodenum);
            }
            for (int j = 0; j < VWU.n_rows; j++){
                mat sf = shape_functions.slice(j);
                rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                double sqa = norm(cross(a_1,a_2),2); 
                double s = sqa; 
                area = area + 1.0/2.0*gqcoeff(j)*s; 
            }
       
        // irregular patch
        }else if ( A(i,12) == -2 ){ 
            mat ori_dots(11,3); // ori_dots(11,3);
            for (int j = 0; j < 11; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            } 
            mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17);  
            M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
            for (int j = 0; j < n; j++){
                mat newnodes17 = M*ori_dots; // 17 new nodes
                /////////////////////////////////////////////////
                // element 4
                mat dots = M4*newnodes17; // dots(12,3);
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                mat dots1 = M1*newnodes17;    // element 1, still irregular patch
                ori_dots = dots1;
            }

        // complex patch
        }else if ( A(i,12) >= 0 ){ 
            mat M(19,13); mat M1(12,19); mat M2(12,19); mat M3(12,19); mat M4(13,19); 
            M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
            mat ori_dots(13,3); // 13 nodes
            for (int j = 0; j < 13; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            }
            for (int j = 0; j < n; j++){
                mat newnodes19 = M*ori_dots; // 19 new nodes
                /////////////////////////////////////////////////
                // element 1
                mat dots = M1*newnodes19; // dots(12,3);
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);  
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                }
                mat dots4 = M4*newnodes19;    // element 4, still complex patch
                ori_dots = dots4;
            }

        // sudo-regular patch 1
        }else if ( A(i,12) == -4 || A(i,12) == -5 ){ 
            mat MM(18,12); mat MM1(11,18); mat MM2(12,18); mat MM3(12,18); mat MM4(13,18); 
            if ( A(i,12) == -4 ){
                MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
            }
            if ( A(i,12) == -5 ){
                MM = subMatrix.sudoreg2M; MM1 = subMatrix.sudoreg2M1; MM2 = subMatrix.sudoreg2M2; MM3 = subMatrix.sudoreg2M3; MM4 = subMatrix.sudoreg2M4;
            }
            mat Dots(12,3); // 
            for (int j = 0; j < 12; j++){
                int nodenum = A(i,j);
                Dots.row(j) = vertex.row(nodenum);
            } 
            mat newnodes18 = MM*Dots; // 18 new nodes
            // two regular patches
            {
                mat dots2 = MM2*newnodes18; // dots(12,3);
                for (int j = 0; j < VWU.n_rows; j++){
                    mat sf = shape_functions.slice(j); 
                    rowvec a_1(3); trans_time(sf.col(1), dots2,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots2,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(j)*s; 
                }
                mat dots3 = MM3*newnodes18; // dots(12,3);
                for (int j = 0; j < VWU.n_rows; j++){
                    mat sf = shape_functions.slice(j); 
                    rowvec a_1(3); trans_time(sf.col(1), dots3,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots3,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    double s = sqa; 
                    area = area + 1.0/2.0*gqcoeff(j)*s; 
                }
            }
            // irregular patch
            {
                mat ori_dots(11,3); 
                ori_dots = MM1*newnodes18; // dots(11,3);
                mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17); 
                M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
                for (int j = 0; j < n-1; j++){
                    mat newnodes17 = M*ori_dots; // 17 new nodes
                    /////////////////////////////////////////////////
                    // element 4
                    mat dots = M4*newnodes17; // dots(12,3);
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2); 
                        double s = sqa; 
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                    }
                    ///////////////////////////////////////////////
                    // element 2 
                    dots = M2*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);  
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2);  
                        double s = sqa; 
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                    }
                    //////////////////////////////////////////////////////
                    // element 3
                    dots = M3*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k); 
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2);  
                        double s = sqa; 
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                    }
                    mat dots1 = M1*newnodes17;    // element 4, still irregular patch
                    ori_dots = dots1;
                }
                // complex patch
                {
                    mat ori_dots(13,3); // 13 nodes
                    ori_dots = MM4*newnodes18; // dots(13,3);
                    mat M(19,13); mat M1(12,19); mat M2(12,19); mat M3(12,19); mat M4(13,19); 
                    M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;        
                    for (int j = 0; j < n-1; j++){
                        mat newnodes19 = M*ori_dots; // 19 new nodes
                        /////////////////////////////////////////////////
                        // element 1
                        mat dots = M1*newnodes19; // dots(12,3);
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2);  
                            double s = sqa; 
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                        }
                        ///////////////////////////////////////////////
                        // element 2 
                        dots = M2*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k);  
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            double s = sqa; 
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                        }
                        //////////////////////////////////////////////////////
                        // element 3
                        dots = M3*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            double s = sqa; 
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                        }
                        mat dots4 = M4*newnodes19;    // element 4, still complex patch
                        ori_dots = dots4;
                    }
                }
            }
        }
        element_area(i) = area;
    }
}

rowvec get_insertionpatchArea(Mat<int> insertionpatch, rowvec elementSout){
    rowvec insertionArea(insertionpatch.n_rows);
    for (int i = 0; i < insertionpatch.n_rows; i++){
        double area = 0.0;
        for (int j = 0; j < insertionpatch.n_cols; j++){
            int faceindex = insertionpatch(i,j);
            area = area + elementSout(faceindex);
        }
        insertionArea(i) = area;
    }
    return insertionArea;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// here, A is the one-ring-nodes
double LocalAreaConstraintEnergy(bool isInnerLayer, mat vertex, Mat<int> face, Mat<int> A, Param param, Row<int> Isinsertionpatch, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, rowvec elementsqa){ // A is face_ring_vertex 
    double energy = 0.0;
    int GaussQuadratureN = param.GaussQuadratureN;
    mat VWU = setVMU(GaussQuadratureN); 

    double us = param.us_in;
    if ( isInnerLayer == false ) us = param.us_out;

    int n = param.subDivideTimes;
    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < face.n_rows; i++){
        if ( i < param.isBoundaryFace.n_cols ){
            if ( param.isBoundaryFace(i) == 1 )
                continue;
        }
        //if ( Isinsertionpatch(i) == 1 ) continue;
        double elementE = 0.0;

        // regular patch
        if  ( A(i,12) == -1 ){ 
            mat dots(12,3); //dots(12,3); 12 nodes
            for (int j = 0; j < 12; j++){
                int nodenum = A(i,j);
                dots.row(j) = vertex.row(nodenum);
            }
            for (int j = 0; j < VWU.n_rows; j++){
                mat sf = shape_functions.slice(j);
                rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                double sqa = norm(cross(a_1,a_2),2); 
                double elementa = elementsqa(i);
                double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                elementE = elementE + 1.0/2.0*gqcoeff(j)*s; 
            }
        
        // irregular patch
        }else if ( A(i,12) == -2 ){ 
            mat ori_dots(11,3); // ori_dots(11,3);
            for (int j = 0; j < 11; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            } 
            mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17);  
            M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
            for (int j = 0; j < n; j++){
                mat newnodes17 = M*ori_dots; // 17 new nodes
                /////////////////////////////////////////////////
                // element 4
                mat dots = M4*newnodes17; // dots(12,3);
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa;
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                mat dots1 = M1*newnodes17;    // element 1, still irregular patch
                ori_dots = dots1;
            }
        
        // complex patch
        }else if ( A(i,12) >= 0 ){ 
            mat M(19,13); mat M1(12,19); mat M2(12,19); mat M3(12,19); mat M4(13,19); 
            M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;
            mat ori_dots(13,3); // 13 nodes
            for (int j = 0; j < 13; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            }
            for (int j = 0; j < n; j++){
                mat newnodes19 = M*ori_dots; // 19 new nodes
                /////////////////////////////////////////////////
                // element 1
                mat dots = M1*newnodes19; // dots(12,3);
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);  
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,j+1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                }
                mat dots4 = M4*newnodes19;    // element 4, still complex patch
                ori_dots = dots4;
            }

        // sudo-regular patch 1
        }else if ( A(i,12) == -4 || A(i,12) == -5 ){ 
            mat MM(18,12); mat MM1(11,18); mat MM2(12,18); mat MM3(12,18); mat MM4(13,18); 
            if ( A(i,12) == -4 ){
                MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
            }
            if ( A(i,12) == -5 ){
                MM = subMatrix.sudoreg2M; MM1 = subMatrix.sudoreg2M1; MM2 = subMatrix.sudoreg2M2; MM3 = subMatrix.sudoreg2M3; MM4 = subMatrix.sudoreg2M4;
            }
            mat Dots(12,3); // 
            for (int j = 0; j < 12; j++){
                int nodenum = A(i,j);
                Dots.row(j) = vertex.row(nodenum);
            } 
            mat newnodes18 = MM*Dots; // 18 new nodes
            // two regular patches
            {
                mat dots2 = MM2*newnodes18; // dots(12,3);
                for (int j = 0; j < VWU.n_rows; j++){
                    mat sf = shape_functions.slice(j); 
                    rowvec a_1(3); trans_time(sf.col(1), dots2,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots2,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                    elementE = elementE + 1.0/2.0*gqcoeff(j)*s;
                }
                mat dots3 = MM3*newnodes18; // dots(12,3);
                for (int j = 0; j < VWU.n_rows; j++){
                    mat sf = shape_functions.slice(j); 
                    rowvec a_1(3); trans_time(sf.col(1), dots3,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots3,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    double elementa = elementsqa(i)/pow(4.0,1.0);
                    double s = pow(sqa - elementa,2.0) * 0.5*us/elementa;
                    elementE = elementE + 1.0/2.0*gqcoeff(j)*s;
                }
            }
            // irregular patch
            {
                mat ori_dots(11,3); 
                ori_dots = MM1*newnodes18; // dots(11,3);
                mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17); 
                M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
                for (int j = 0; j < n-1; j++){
                    mat newnodes17 = M*ori_dots; // 17 new nodes
                    /////////////////////////////////////////////////
                    // element 4
                    mat dots = M4*newnodes17; // dots(12,3);
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2); 
                        double elementa = elementsqa(i)/pow(4.0,j+2.0);
                        double s = pow(sqa - elementa,2.0) * 0.5*us/elementa;
                        elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                    }
                    ///////////////////////////////////////////////
                    // element 2 
                    dots = M2*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);  
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2);  
                        double elementa = elementsqa(i)/pow(4.0,j+2.0);
                        double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                        elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                    }
                    //////////////////////////////////////////////////////
                    // element 3
                    dots = M3*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k); 
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2);  
                        double elementa = elementsqa(i)/pow(4.0,j+2.0);
                        double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                        elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                    }
                    mat dots1 = M1*newnodes17;    // element 4, still irregular patch
                    ori_dots = dots1;
                }
                // complex patch
                {
                    mat ori_dots(13,3); // 13 nodes
                    ori_dots = MM4*newnodes18; // dots(13,3);
                    mat M(19,13); mat M1(12,19); mat M2(12,19); mat M3(12,19); mat M4(13,19); 
                    M = subMatrix.comregM; M1 = subMatrix.comregM1; M2 = subMatrix.comregM2; M3 = subMatrix.comregM3; M4 = subMatrix.comregM4;        
                    for (int j = 0; j < n-1; j++){
                        mat newnodes19 = M*ori_dots; // 19 new nodes
                        /////////////////////////////////////////////////
                        // element 1
                        mat dots = M1*newnodes19; // dots(12,3);
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2);  
                            double elementa = elementsqa(i)/pow(4.0,j+2.0);
                            double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                            elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                        }
                        ///////////////////////////////////////////////
                        // element 2 
                        dots = M2*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k);  
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            double elementa = elementsqa(i)/pow(4.0,j+2.0);
                            double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                            elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                        }
                        //////////////////////////////////////////////////////
                        // element 3
                        dots = M3*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            double elementa = elementsqa(i)/pow(4.0,j+2.0);
                            double s = pow(sqa - elementa,2.0) * 0.5*us/elementa; 
                            elementE = elementE + 1.0/2.0*gqcoeff(k)*s;
                        }
                        mat dots4 = M4*newnodes19;    // element 4, still complex patch
                        ori_dots = dots4;
                    }
                }
            }
        }
        energy = energy + elementE;
    }
    return energy;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

rowvec determine_spontaneous_curvature(bool isInnerLayer, Param param, Mat<int> face, mat vertex){
    rowvec spontcurv(face.n_rows); 
    Mat<int> insertionpatch = param.insertionpatch;
    if (isInnerLayer == false){
        double c0ins = param.c0out_ins;
        double c0 = param.c0out;
        spontcurv.fill(c0);
        for (int i = 0; i < insertionpatch.n_rows; i++){
            for (int j = 0; j < insertionpatch.n_cols; j++){
                int facenumber = insertionpatch(i,j);
                spontcurv(facenumber) = c0ins;
            }
        }
    }else{
        double c0 = param.c0in;
        spontcurv.fill(c0);
        return spontcurv;
    }
    
    return spontcurv;
}

Row<int> determine_isinsertionpatch(Mat<int> face, Mat<int> insertionpatch){
    int facenumber = face.n_rows;
    Row<int> out(facenumber); out.fill(0);
    for (int j = 0; j < insertionpatch.n_rows; j++){
        for (int m = 0; m < insertionpatch.n_cols; m++){
            int insertnum = insertionpatch(j,m); 
            if ( insertnum <= facenumber-1 ){
                out(insertnum) = 1;
            }
        }
    }
    return out;
}

rowvec define_insertion_shape(double l, double a, double b, double dl){
    rowvec targetSides(3); 
    // index '0', to the partner
    // index '1', to one end node
    // index '2', to the other end node
    
    
    // ellipse shape
    // each insertion patch has ellipse shape with long and short radius: a = 1.25 nm, b = 0.5093 nm, area=pi*a*b
    //double a = 1.25; double b = 0.8;//0.5093;
    double sideitarget = 2.0*b/a*sqrt(a*a-pow(a-dl,2.0));
    targetSides(0) = sideitarget;
    targetSides(1) = sqrt( pow(sideitarget/2.0,2.0) + pow(dl,2.0) );
    targetSides(2) = sqrt( pow(sideitarget/2.0,2.0) + pow(2.0*a-dl,2.0) );
    /*
    // rectanglar shape
    double sideitarget = 0.0;
    if ( dl < sqrt(3.0)*b ){
        sideitarget = 2.0 * dl/sqrt(3.0);
    }else if ( dl > 2.0*a - sqrt(3.0)*b){
        sideitarget = 2.0 * (2.0*a-dl)/sqrt(3.0);
    }else{
        sideitarget = 2.0 * b;
    }
    targetSides(0) = sideitarget;
    targetSides(1) = sqrt( pow(sideitarget/2.0,2.0) + pow(dl,2.0) );
    targetSides(2) = sqrt( pow(sideitarget/2.0,2.0) + pow(2.0*a-dl,2.0) );
    */
    return targetSides;
}

void insertion_shape_constraint(Param param, Mat<int> faceout, mat vertexout, double& Einsert, mat& finsert){
    Mat<int> insertionpatch = param.insertionpatch;
    if ( insertionpatch.n_cols > 0 && insertionpatch.n_rows > 0 )
    {   
        double k = param.K_insertShape;
        //#pragma omp parallel for reduction(+:finsert) 
        for ( int i = 0; i < insertionpatch.n_rows; i++ ) {
            for (int j = 0; j < insertionpatch.n_cols; j++ ){
                int facenum = insertionpatch(i,j);
                int node0 = faceout(facenum,0); // three nodes of this face element
                int node1 = faceout(facenum,1);
                int node2 = faceout(facenum,2);
                rowvec vector0 = vertexout.row(node0) - vertexout.row(node1);  double side0 = norm(vector0,2.0);
                rowvec vector1 = vertexout.row(node1) - vertexout.row(node2);  double side1 = norm(vector1,2.0);
                rowvec vector2 = vertexout.row(node2) - vertexout.row(node0);  double side2 = norm(vector2,2.0);
                double meanside = param.insertionShapeEdgeLength;
                Einsert = Einsert + k/2.0*(pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0));
                finsert.row(node0) = finsert.row(node0) + k*( (side0-meanside)*(-vector0/side0) + (side2-meanside)*(vector2/side2) );
                finsert.row(node1) = finsert.row(node1) + k*( (side1-meanside)*(-vector1/side1) + (side0-meanside)*(vector0/side0) );
                finsert.row(node2) = finsert.row(node2) + k*( (side2-meanside)*(-vector2/side2) + (side1-meanside)*(vector1/side1) );
            }
        }
        // add the constraint on the adjacent patches around the insertionpatch
        k = param.K_adjacentPatch;
        for (int i = 0; i < faceout.n_rows; i++){
            if (param.IsinertionpatchAdjacent(i) != 1) continue;
            int facetmp = i;
            int node0 = faceout(facetmp,0); // three nodes of this face element
            int node1 = faceout(facetmp,1);
            int node2 = faceout(facetmp,2);
            rowvec vector0 = vertexout.row(node0) - vertexout.row(node1);  double side0 = norm(vector0,2.0);
            rowvec vector1 = vertexout.row(node1) - vertexout.row(node2);  double side1 = norm(vector1,2.0);
            rowvec vector2 = vertexout.row(node2) - vertexout.row(node0);  double side2 = norm(vector2,2.0);
            double meanside = param.insertionShapeEdgeLength;
            Einsert = Einsert + k/2.0*(pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0));
            finsert.row(node0) = finsert.row(node0) + k*( (side0-meanside)*(-vector0/side0) + (side2-meanside)*(vector2/side2) );
            finsert.row(node1) = finsert.row(node1) + k*( (side1-meanside)*(-vector1/side1) + (side0-meanside)*(vector0/side0) );
            finsert.row(node2) = finsert.row(node2) + k*( (side2-meanside)*(-vector2/side2) + (side1-meanside)*(vector1/side1) );
        }

        finsert = -finsert;
    } 
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// calculate the thickness
rowvec calculate_thickness(bool IsinnerLayer, Mat<int> face, mat vertexin, mat vertexout, Mat<int> one_ring_nodes, Param param, cube shape_functions, SubMatrix subMatrix){ // A is face_ring_vertex 
    
    rowvec thickness(face.n_rows); thickness.fill(-1.0);

    int GaussQuadratureN = param.GaussQuadratureN;
    //mat VWU = setVMU(GaussQuadratureN); 
    mat sf = shape_functions.slice(0);

    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        if ( param.isBoundaryFace(i) == 1 ){
            continue;
        }

        //double hnorm = 0.0; 
        mat dotsin(12,3);  // one-ring-vertices
        mat dotsout(12,3);

        // regular patch
        if  ( one_ring_nodes(i,12) == -1 ){ 
            for (int j = 0; j < 12; j++) {
                int nodenum = one_ring_nodes(i,j);
                dotsin.row(j) = vertexin.row(nodenum);
                dotsout.row(j) = vertexout.row(nodenum);
            }
       
        // irregular-minus patch
        }else if ( one_ring_nodes(i,12) == -2 ){ 
            mat ori_dotsin(11,3); 
            mat ori_dotsout(11,3); // one ring vertices
            for (int j = 0; j < 11; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dotsin.row(j) = vertexin.row(nodenum);
                ori_dotsout.row(j) = vertexout.row(nodenum);
            }
            mat M(17,11); mat M2(12,17); 
            M = subMatrix.irregM; M2 = subMatrix.irregM2; 
            dotsin = M2 * ( M*ori_dotsin ); 
            dotsout = M2 * ( M*ori_dotsout );    

        // irregular-plus patch
        }else if ( one_ring_nodes(i,12) >= 0 ){ 
            mat ori_dotsin(13,3); 
            mat ori_dotsout(13,3); // one ring vertices
            for (int j = 0; j < 13; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dotsin.row(j) = vertexin.row(nodenum);
                ori_dotsout.row(j) = vertexout.row(nodenum);
            }
            mat M(19,13); mat M2(12,19); 
            M = subMatrix.comregM; M2 = subMatrix.comregM2; 
            dotsin = M2 * ( M*ori_dotsin ); 
            dotsout = M2 * ( M*ori_dotsout );    

        // sudo-regular patch 1
        }else if ( one_ring_nodes(i,12) == -4 || one_ring_nodes(i,12) == -5 ){ 
            mat ori_dotsin(12,3); 
            mat ori_dotsout(12,3); // one ring vertices
            for (int j = 0; j < 12; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dotsin.row(j) = vertexin.row(nodenum);
                ori_dotsout.row(j) = vertexout.row(nodenum);
            }
            mat M(18,12); mat M2(12,18); 
            if ( one_ring_nodes(i,12) == -4 ){
                M = subMatrix.sudoreg1M; M2 = subMatrix.sudoreg1M2; 
            }
            if ( one_ring_nodes(i,12) == -5 ){
                M = subMatrix.sudoreg2M; M2 = subMatrix.sudoreg2M2; 
            }
            dotsin = M2 * ( M*ori_dotsin ); 
            dotsout = M2 * ( M*ori_dotsout ); 
        }

        // inner layer
        rowvec x(3); trans_time(sf.col(0),dotsin,x);
        rowvec a_1(3); trans_time(sf.col(1),dotsin,a_1);
        rowvec a_2(3); trans_time(sf.col(2),dotsin,a_2);
        rowvec xa = cross(a_1,a_2);
        double sqa = norm(xa,2);
        rowvec din = xa/sqa;
        rowvec xin = x;
        // outer layer
        trans_time(sf.col(0),dotsout,x);
        trans_time(sf.col(1),dotsout,a_1);
        trans_time(sf.col(2),dotsout,a_2);
        xa = cross(a_1,a_2);
        sqa = norm(xa,2);
        rowvec dout = xa/sqa;
        rowvec xout = x;
        rowvec h = xout - xin; 
        mat hnorm_matrix(1,1); 
        if ( IsinnerLayer == false ){
            hnorm_matrix = h * strans(dout);
        }else{
            hnorm_matrix = h * strans(din);
        }
        
        thickness(i) = hnorm_matrix(0,0);
    }

    return thickness;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// calculate the triangle centers on the limit surface, to show the shape of the surface.
mat calculate_triangleCenter(Mat<int> face, mat vertex, Mat<int> one_ring_nodes, Param param, cube shape_functions, SubMatrix subMatrix){ // A is face_ring_vertex 
    mat centers(face.n_rows,3); centers.fill(0.0);

    int GaussQuadratureN = param.GaussQuadratureN;
    //mat VWU = setVMU(GaussQuadratureN); 
    mat sf = shape_functions.slice(0);

    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        if ( param.isBoundaryFace(i) == 1 ){
            continue;
        }
        
        //double hnorm = 0.0; 
        mat dots(12,3);  // one-ring-vertices
        // regular patch
        if  ( one_ring_nodes(i,12) == -1 ){ 
            for (int j = 0; j < 12; j++) {
                int nodenum = one_ring_nodes(i,j);
                dots.row(j) = vertex.row(nodenum);
            }
       
        // irregular-minus patch
        }else if ( one_ring_nodes(i,12) == -2 ){ 
            mat ori_dots(11,3); 
            for (int j = 0; j < 11; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            }
            mat M(17,11); mat M2(12,17); 
            M = subMatrix.irregM; M2 = subMatrix.irregM2; 
            dots = M2 * ( M*ori_dots );  

        // irregular-plus patch
        }else if ( one_ring_nodes(i,12) >= 0 ){ 
            mat ori_dots(13,3); 
            for (int j = 0; j < 13; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            }
            mat M(19,13); mat M2(12,19); 
            M = subMatrix.comregM; M2 = subMatrix.comregM2; 
            dots = M2 * ( M*ori_dots );  

        // sudo-regular patch 1
        }else if ( one_ring_nodes(i,12) == -4 || one_ring_nodes(i,12) == -5 ){ 
            mat ori_dots(12,3); 
            for (int j = 0; j < 12; j++) {
                int nodenum = one_ring_nodes(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            }
            mat M(18,12); mat M2(12,18); 
            if ( one_ring_nodes(i,12) == -4 ){
                M = subMatrix.sudoreg1M; M2 = subMatrix.sudoreg1M2; 
            }
            if ( one_ring_nodes(i,12) == -5 ){
                M = subMatrix.sudoreg2M; M2 = subMatrix.sudoreg2M2; 
            }
            dots = M2 * ( M*ori_dots ); 
        }

        rowvec x(3); trans_time(sf.col(0),dots,x);
        
        centers.row(i) = x;

    }

    return centers;
} 