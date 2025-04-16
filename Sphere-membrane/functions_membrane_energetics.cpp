#include <math.h>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include "functions_print_and_read.cpp"

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// the following are bending energy, volume energy, and local area energy 

// bending energy and forces, area constraint energy and forces, volume constriant energy and forces
void element_energy_force_regular(int whichLayer, bool isInsertionPatch, mat dots, Param param, double C0_spont, double& Ebe, mat& F_be, mat& F_s, mat& F_v, rowvec gqcoeff, cube shape_functions, double a0) {
    // F_be is the force related to the curvature; F_s is the force related to the area-constraint; F_v is the force related to the volume-constraint.
    // initialize output parameters
    double c0 = C0_spont;
    if ( whichLayer == 0 ){ // inner layer, the sign of memrbane direction is opposite! 
        c0 = -c0;
        // The bending energy should be: kc/2*(-2H-c0)^2, equals kc/2*(2H+c0)^2
        // Therefore, instead of changing H sign, change c0 sign. 
    }

    Ebe = 0.0;
    F_be.fill(0);
    F_s.fill(0);
    F_v.fill(0);
    
    double S0 = param.S0out; double V0 = param.V0out; 
    double S  = param.Sout;  double V  = param.Vout; 
    double kc = param.kc_out; 
    double us = param.us_out / S0;
    double uv = 0.0; 
    if ( whichLayer == 0 ){ // inner layer
        S0 = param.S0in; V0 = param.V0in; 
        S  = param.Sin;  V  = param.Vin;
        kc = param.kc_in; 
        us = param.us_in / S0;
        uv = param.uv / V0; // volume constraint only works to the inner layer
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
        mat f_conv(12,3); f_conv.fill(0);
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
        rowvec n1_conv = 1.0/3.0*uv*(V-V0)*(x*strans(d)*a1-x*strans(a1)*d);
        rowvec n2_conv = 1.0/3.0*uv*(V-V0)*(x*strans(d)*a2-x*strans(a2)*d);
        for (int j = 0; j < 12; j++) {
            mat da1 = -sf(j,3)*kron(strans(a1),d) - sf(j,1)*kron(strans(a11),d) - sf(j,1)*kron(strans(a1),d_1) - sf(j,6)*kron(strans(a2),d) - sf(j,2)*kron(strans(a21),d) - sf(j,2)*kron(strans(a2),d_1);
            mat da2 = -sf(j,5)*kron(strans(a1),d) - sf(j,1)*kron(strans(a12),d) - sf(j,1)*kron(strans(a1),d_2) - sf(j,4)*kron(strans(a2),d) - sf(j,2)*kron(strans(a22),d) - sf(j,2)*kron(strans(a2),d_2);
            rowvec tempf_be = n1_be*sf(j,1) + m1_be*da1 + n2_be*sf(j,2) + m2_be*da2;
            f_be.row(j) = tempf_be*sqa;
            rowvec tempf_cons = n1_cons*sf(j,1) + n2_cons*sf(j,2);
            f_cons.row(j) = tempf_cons*sqa;
            rowvec tempf_conv = n1_conv*sf(j,1) + n2_conv*sf(j,2) + 1.0/3.0*uv*(V-V0)*d*sf(j,0);
            f_conv.row(j) = tempf_conv*sqa;
        }
        Ebe = Ebe + 1.0/2.0*gqcoeff(i)*ebe;
        F_be = F_be + 1.0/2.0*gqcoeff(i)*f_be;
        F_s = F_s + 1.0/2.0*gqcoeff(i)*f_cons;
        F_v = F_v + 1.0/2.0*gqcoeff(i)*f_conv;
    }
}

void element_energy_force_irregularMinus(int whichLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, mat& cunchu3, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    cunchu3.fill(0.0);
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
        mat f3 = f1;
        double ebe1 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        cunchu3 = cunchu3 + m1m*f3;
        E_bending = E_bending + ebe1;

        dots = M2*newnodes17;    // element 2
        f1.fill(0.0);
        f2.fill(0.0);
        f3.fill(0.0);
        double ebe2 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        cunchu3 = cunchu3 + m2m*f3;
        E_bending = E_bending + ebe2;

        dots = M3*newnodes17;    // element 3
        f1.fill(0.0);
        f2.fill(0.0);
        f3.fill(0.0);
        double ebe3 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        cunchu3 = cunchu3 + m3m*f3;
        E_bending = E_bending + ebe3;

        mat dots1 = M1*newnodes17;   // element 4, still irregular patch
        ori_dots = dots1;
    }
}

void element_energy_force_irregularPlus(int whichLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, mat& cunchu3, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    cunchu3.fill(0.0);
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
        mat f3 = f1;
        double ebe1 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m1m = strans(M1*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        cunchu3 = cunchu3 + m1m*f3;
        E_bending = E_bending + ebe1;

        dots = M2*newnodes19;    // element 2
        f1.fill(0.0);
        f2.fill(0.0);
        f3.fill(0.0);
        double ebe2 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        cunchu3 = cunchu3 + m2m*f3;
        E_bending = E_bending + ebe2;

        dots = M3*newnodes19;    // element 3
        f1.fill(0.0);
        f2.fill(0.0);
        f3.fill(0.0);
        double ebe3 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+1.0));
        mat m3m = strans(M3*matrix);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        cunchu3 = cunchu3 + m3m*f3;
        E_bending = E_bending + ebe3;

        mat dots4 = M4*newnodes19;   // element 4, still irregular patch
        ori_dots = dots4;
    }
}

void element_energy_force_pseudoRegular1(int whichLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, mat& cunchu3, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    cunchu3.fill(0.0); 
    // five matrix used for subdivision of the irregular patch
    mat MM(18,12), MM1(11,18), MM2(12,18), MM3(12,17), MM4(13,18); 
    MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
    int n = param.subDivideTimes;
    mat newnodes18 = MM*Dots; // 18 new nodes
    /////////////////////////////////////////////////////////////////
    // two regular patch
    {
        mat dots = MM2*newnodes18;    // element 2
        mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
        double ebe1 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2,  f3, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m1m = strans(MM2*MM);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        cunchu3 = cunchu3 + m1m*f3;
        E_bending = E_bending + ebe1;
        
        dots = MM3*newnodes18;    // element 3
        double ebe3 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m3m = strans(MM3*MM);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        cunchu3 = cunchu3 + m1m*f3;
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
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
            double ebe1 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M4*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes17;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes17;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            cunchu3 = cunchu3 + m1m*f3;
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
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
            double ebe1 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M1*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes19;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes19;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe3;

            mat dots4 = M4*newnodes19;   // element 4, still irregular-plus patch
            ori_dots = dots4;
        }
    }
}

void element_energy_force_pseudoRegular2(int whichLayer, bool isInsertionPatch, mat Dots, Param param, double c0, double& E_bending, mat& cunchu1, mat& cunchu2, mat& cunchu3, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0) {
    // cunchu1 is the curvature force; cunchu2 is the area-constraint force; cunchu3 is the volume-constraint force.
    // initialize the output parameters
    E_bending = 0.0;
    cunchu1.fill(0.0);
    cunchu2.fill(0.0);
    cunchu3.fill(0.0);
    // five matrix used for subdivision of the irregular patch
    mat MM(18,12), MM1(11,18), MM2(12,18), MM3(12,18), MM4(13,18); 
    MM = subMatrix.sudoreg2M; MM1 = subMatrix.sudoreg2M1; MM2 = subMatrix.sudoreg2M2; MM3 = subMatrix.sudoreg2M3; MM4 = subMatrix.sudoreg2M4;
    int n = param.subDivideTimes;
    mat newnodes18 = MM*Dots; // 18 new nodes
    /////////////////////////////////////////////////////////////////
    // two regular patch
    {
        mat dots = MM2*newnodes18;    // element 2
        mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
        double ebe1 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m1m = strans(MM2*MM);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        cunchu3 = cunchu3 + m1m*f3;
        E_bending = E_bending + ebe1;
        
        dots = MM3*newnodes18;    // element 3
        double ebe3 = 0.0;
        element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,1.0));
        mat m3m = strans(MM3*MM);
        cunchu1 = cunchu1 + m3m*f1;
        cunchu2 = cunchu2 + m3m*f2;
        cunchu3 = cunchu3 + m1m*f3;
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
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
            double ebe1 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M4*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes17;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes17;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            cunchu3 = cunchu3 + m1m*f3;
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
            mat f1(12,3);  f1.fill(0.0); mat f2 = f1; mat f3 = f1;
            double ebe1 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe1, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m1m = strans(M1*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe1;

            dots = M2*newnodes19;    // element 2
            double ebe2 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe2, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m2m = strans(M2*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe2;

            dots = M3*newnodes19;    // element 3
            double ebe3 = 0.0;
            element_energy_force_regular(whichLayer, isInsertionPatch, dots, param, c0, ebe3, f1, f2, f3, gqcoeff, shape_functions, a0/pow(4.0,j+2.0));
            mat m3m = strans(M3*matrix);
            cunchu1 = cunchu1 + m3m*f1;
            cunchu2 = cunchu2 + m3m*f2;
            cunchu3 = cunchu3 + m1m*f3;
            E_bending = E_bending + ebe3;

            mat dots4 = M4*newnodes19;   // element 4, still irregular-plus patch
            ori_dots = dots4;
        }
    }
}

// to sum up
void element_energy_force(int whichLayer, bool isInsertionPatch, Row<int> one_ring_nodesin, Param param, double Cin, double& Ebendingin, mat& fbein, mat vertexin, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, double a0){
    // regular patch
    if ( one_ring_nodesin(12) == -1 ) { // for regular patch
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3); mat finS(12,3); mat finV(12,3);// bending or curvature term, area term
        element_energy_force_regular(whichLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, finV, gqcoeff, shape_functions, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j) + finV.row(j);
        } 

    // irregular-minus patch
    }else if ( one_ring_nodesin(12) == -2 ){
        mat dotsin(11,3);  // one ring vertices
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(11,3);  mat finS(11,3); mat finV(11,3); // bending or curvature term, area term
        element_energy_force_irregularMinus(whichLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, finV, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 11; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j) + finV.row(j);
        } 

    // irregular-plus patch
    }else if ( one_ring_nodesin(12) >= 0 ){  
        mat dotsin(13,3);  // one ring vertices
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(13,3);  mat finS(13,3); mat finV(13,3); // bending or curvature term, area term
        element_energy_force_irregularPlus(whichLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, finV, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 13; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j) + finV.row(j);
        } 
            
    // pseudo-regular patch 1
    }else if ( one_ring_nodesin(12) == -4 ){  
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3);  mat finS(12,3); mat finV(12,3); // bending or curvature term, area term
        element_energy_force_pseudoRegular1(whichLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, finV, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j) + finV.row(j);
        } 

    // pseudo-regular patch 2
    }else if ( one_ring_nodesin(12) == -5 ){  
        mat dotsin(12,3);  // one ring vertices
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            dotsin.row(j) = vertexin.row(nodenum);
        }
        double ebein = 0.0; 
        mat finB(12,3);  mat finS(12,3); mat finV(12,3); // bending or curvature term, area term
        element_energy_force_pseudoRegular2(whichLayer, isInsertionPatch, dotsin, param, Cin, ebein, finB, finS, finV, gqcoeff, shape_functions, subMatrix, a0); 
        Ebendingin = ebein;
        for (int j = 0; j < 12; j++) {
            int nodenum = one_ring_nodesin(j);
            fbein.row(nodenum) = fbein.row(nodenum) + finB.row(j) + finS.row(j) + finV.row(j);
        } 
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// the following are thickness energy and forces

void element_energy_force_thickness_regular(bool isInsertionPatch, int whichLayer, mat dotsin, mat dotsout, Param param, rowvec& E, mat& Fin, mat& Fout, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions){
    // initialize output parameters
    double c0 = C0_spont;
    double S0 = param.S0out; double V0 = param.V0out; 
    double S  = param.Sout;  double V  = param.Vout;
    double kc = param.kc_out;
    double us = param.us_out/S0;
    double uv = 0.0; 
    double h0 = param.H0C_out; //param.thickness_out;
    double kt = param.Kthick_out; // height modulus
    double kg = param.kst_out; // splay-tilt modulus, for tilt divergence
    if (whichLayer == 0){ // 0 means inLayer, 2 means outLayer
        S0 = param.S0in; V0 = param.V0in;
        S  = param.Sin;  V  = param.Vin;
        kc = param.kc_in;
        c0 = -c0; // The bending energy should be: kc/2*(-2H-c0)^2, equals kc/2*(2H+c0)^2. 
                  // Therefore, instead of changing H sign, change c0 sign. 
        us = param.us_in/S0;
        h0 = param.H0C_in; //param.thickness_in;
        kt = param.Kthick_in;
        kg = param.kst_in;
        uv = param.uv / V0; // volume constraint only works to the inner layer
    }
    if ( param.isGlobalAreaConstraint == false ){
        us = us/a0;
    }
    if ( isInsertionPatch == true ){
        h0 = h0 - param.insert_dH0;
    }
    double ktpen = param.Kthick_constraint; 
    double ktinsert = param.Kthick_constraint; // param.Kthick_insertion;
    
    E.fill(0.0); Fout.fill(0.0); Fin.fill(0.0);
    int GaussQuadratureN = param.GaussQuadratureN;
    mat VWU = setVMU(GaussQuadratureN); 
    
    mat fout(12,3); fout.fill(0.0); 
    mat fin(12,3);  fin.fill(0.0);
    for (int i = 0; i < VWU.n_rows; i++) {
        mat sf= shape_functions.slice(i); // 12 shape functions
        // a_1,2,3 covariant vectors; a1,2 contravariant vectors;
        // outlayer
        rowvec xout(3); trans_time(sf.col(0),dotsout,xout);
        rowvec a_1out(3); trans_time(sf.col(1),dotsout,a_1out);
        rowvec a_2out(3); trans_time(sf.col(2),dotsout,a_2out);
        rowvec a_11out(3); trans_time(sf.col(3),dotsout,a_11out);
        rowvec a_22out(3); trans_time(sf.col(4),dotsout,a_22out);
        rowvec a_12out(3); trans_time(sf.col(5),dotsout,a_12out);
        rowvec a_21out(3); trans_time(sf.col(6),dotsout,a_21out);
        rowvec xaout = cross(a_1out,a_2out);
        double sqaout = norm(xaout,2);
        rowvec xa_1out = cross(a_11out,a_2out) + cross(a_1out,a_21out);
        rowvec xa_2out = cross(a_12out,a_2out) + cross(a_1out,a_22out);
        mat oneelement1 = xaout * strans(xa_1out);
        double sqa_1out = 1.0/sqaout * oneelement1(0,0);
        mat oneelement2 = xaout * strans(xa_2out);
        double sqa_2out = 1.0/sqaout * oneelement2(0,0);
        rowvec a_3out = xaout/sqaout;
        rowvec a_31out = 1.0/sqaout/sqaout *(xa_1out*sqaout-xaout*sqa_1out);
        rowvec a_32out = 1.0/sqaout/sqaout *(xa_2out*sqaout-xaout*sqa_2out);
        rowvec dout = a_3out;
        rowvec d_1out = a_31out;
        rowvec d_2out = a_32out;
        rowvec a1out = cross(a_2out, a_3out)/sqaout;
        rowvec a2out = cross(a_3out ,a_1out)/sqaout;
        rowvec a11out = 1.0/sqaout/sqaout *( (cross(a_21out,a_3out)+cross(a_2out,a_31out))*sqaout - cross(a_2out,a_3out)*sqa_1out );
        rowvec a12out = 1.0/sqaout/sqaout *( (cross(a_22out,a_3out)+cross(a_2out,a_32out))*sqaout - cross(a_2out,a_3out)*sqa_2out );
        rowvec a21out = 1.0/sqaout/sqaout *( (cross(a_31out,a_1out)+cross(a_3out,a_11out))*sqaout - cross(a_3out,a_1out)*sqa_1out );
        rowvec a22out = 1.0/sqaout/sqaout *( (cross(a_32out,a_1out)+cross(a_3out,a_12out))*sqaout - cross(a_3out,a_1out)*sqa_2out );
        mat shu = a1out * strans(d_1out) + a2out * strans(d_2out);
        double Hcurv = 0.5*shu(0,0); // mean curvature
        // inlayer 
        rowvec xin(3); trans_time(sf.col(0),dotsin,xin);
        rowvec a_1in(3); trans_time(sf.col(1),dotsin,a_1in);
        rowvec a_2in(3); trans_time(sf.col(2),dotsin,a_2in);
        // height strain
        rowvec h = xout - xin;
        rowvec h_1 = a_1out - a_1in;
        rowvec h_2 = a_2out - a_2in;
        mat hnorm_matrix = h * strans(dout);
        if ( whichLayer == 0 ) hnorm_matrix = -hnorm_matrix;
        double hnorm = hnorm_matrix(0,0); // observed height
        double rho = (hnorm-h0)/h0; // height elasticity or height strain 
        mat rho_1_matrix = 1.0/h0 * ( h_1 * strans(dout) + h * strans(d_1out) ); double rho_1 = rho_1_matrix(0,0);
        mat rho_2_matrix = 1.0/h0 * ( h_2 * strans(dout) + h * strans(d_2out) ); double rho_2 = rho_2_matrix(0,0);
        if ( whichLayer == 0 ){
            rho_1 = -rho_1;
            rho_2 = -rho_2;
        }
        
        // height energy (tilt energy)
        double Eheight = 0.0; // energy; height elasticity
        double eps = -1.0e-2; // criterion for E2
        double A = kt/pow(eps,2.0)*(eps-1.0/3.0) - kt/pow(eps,3.0)*pow(eps-1.0/3.0,2.0) + 1.0/9.0*kt/pow(eps,3.0); // Two coefficient of E2 function
        double B = - kt/eps*(eps-1.0/3.0) + 3.0/2.0*kt/pow(eps,2.0)*pow(eps-1.0/3.0,2.0) - 1.0/6.0*kt/pow(eps,2.0);
        {
            if ( rho <= eps ){ // E1 branch;
                Eheight = ( 0.5 * kt * pow(rho - 1.0/3.0, 2.0) - kt/18.0 ) * sqaout; 
            }else if ( rho > eps && rho <= 0.0 ){ // E2 branch; Aims to make the energy function and its derivation continuous. 
                Eheight = ( A * pow(rho,3.0) + B * pow(rho, 2.0) ) * sqaout;
            }else if ( rho > 0.0 ){ // E3 branch; expanded height
                Eheight = ( 0.5 * ktpen * pow(rho, 2.0) )* sqaout;
            }
        }
        // divergence of tilt
        double dTilt_drho = 0.0;
        double deltaShu = sqrt(3.0*eps*eps - 2.0*eps);
        double A1 = 0.5/eps/deltaShu *(3.0 - pow((3.0*eps-1.0)/deltaShu, 2.0));
        double B1 = (3.0*eps-1.0)/deltaShu - A1*eps*eps;
        if ( param.includeDivTilt == true ){        
            if ( rho <= eps ){ // E1 branch;
                dTilt_drho = (3.0*rho-1.0)/sqrt(3.0*rho*rho-2.0*rho); 
            }else if ( rho > eps ){ // E2 branch; Aims to make the energy function and its derivation continuous. 
                dTilt_drho = A1*rho*rho + B1; 
            }
        }
        double divTilt = dTilt_drho * (rho_1 + rho_2);
        // curvature energy
        double Ebend = 0.5 * kc * pow(2.0*Hcurv + divTilt - c0, 2.0) * sqaout; 
        // curvature-tilt energy
        double EcurvTilt = 0.5 * kg * ( 2.0*Hcurv + divTilt ) * divTilt * sqaout;
        
        // area and volume elasticity. 
        rowvec n1_cons = us*(S-S0)*a1out;
        rowvec n2_cons = us*(S-S0)*a2out;
        if ( param.isGlobalAreaConstraint == false ){
            n1_cons = us*(sqaout-a0)*a1out;
            n2_cons = us*(sqaout-a0)*a2out;
        }
        rowvec n1_conv(3); n1_conv.fill(0.0); 
        rowvec n2_conv(3); n2_conv.fill(0.0); 
        if ( param.isFlatMembrane == false ){
            n1_conv = 1.0/3.0*uv*(V-V0)*(xout*strans(dout)*a1out-xout*strans(a1out)*dout);
            n2_conv = 1.0/3.0*uv*(V-V0)*(xout*strans(dout)*a2out-xout*strans(a2out)*dout);
        }
        
        fout.fill(0.0); fin.fill(0);
        for (int j = 0; j < 12; j++) { // j is the vertex index in one-ring 
            rowvec sqaout_j = sqaout * ( a1out*sf(j,1) + a2out*sf(j,2));
            mat dout_j   = -sf(j,1)*kron(strans(a1out),dout) - sf(j,2)*kron(strans(a2out),dout);
            mat d_1out_j = -sf(j,3)*kron(strans(a1out),dout) - sf(j,1)*kron(strans(a11out),dout) - sf(j,1)*kron(strans(a1out),d_1out) - sf(j,6)*kron(strans(a2out),dout) - sf(j,2)*kron(strans(a21out),dout) - sf(j,2)*kron(strans(a2out),d_1out);
            mat d_2out_j = -sf(j,5)*kron(strans(a1out),dout) - sf(j,1)*kron(strans(a12out),dout) - sf(j,1)*kron(strans(a1out),d_2out) - sf(j,4)*kron(strans(a2out),dout) - sf(j,2)*kron(strans(a22out),dout) - sf(j,2)*kron(strans(a2out),d_2out);
            rowvec Hcurv_j = -0.5*a1out*strans(a1out)*d_1out*sf(j,1) - 0.5*a1out*strans(a2out)*d_1out*sf(j,2) - 0.5*a2out*strans(a1out)*d_2out*sf(j,1) - 0.5*a2out*strans(a2out)*d_2out*sf(j,2) + 0.5*a1out*d_1out_j + 0.5*a2out*d_2out_j;
            rowvec h_out_j = sf(j,0)*dout + (xout-xin)*dout_j;
            rowvec h_in_j  = -sf(j,0)*dout;
            rowvec rho_jout = 1.0/h0 * (sf(j,0)*dout + (xout-xin)*dout_j);
            rowvec rho_jin  = 1.0/h0 * (-sf(j,0)*dout); 
            if ( whichLayer == 0 ){
                rho_jout = -rho_jout;
                rho_jin  = -rho_jin;
            }
            // height force
            // modify the nodal force if it is insertionPatch or expanded height
            {
                if ( rho <= eps ){ // E1 branch; 
                    fout.row(j) = fout.row(j) + kt*(rho-1.0/3.0)*sqaout*rho_jout + (0.5*kt*pow(rho-1.0/3.0,2.0)-kt/18.0)*sqaout_j;    
                    fin.row(j)  = fin.row(j)  + kt*(rho-1.0/3.0)*sqaout*rho_jin; 
                }else if ( rho > eps && rho <= 0.0 ){ // E2 branch; 
                    fout.row(j) = fout.row(j) + (3.0*A*pow(rho,2.0)+2.0*B*rho)*sqaout*rho_jout + (A*pow(rho,3.0)+B*pow(rho,2.0))*sqaout_j;
                    fin.row(j)  = fin.row(j)  + (3.0*A*pow(rho,2.0)+2.0*B*rho)*sqaout*rho_jin;
                }else if ( rho > 0.0 ){ // E3 branch; expanded height
                    fout.row(j) = fout.row(j) + ktpen*rho*sqaout*rho_jout + 0.5*ktpen*pow(rho,2.0)*sqaout_j;    
                    fin.row(j)  = fin.row(j)  + ktpen*rho*sqaout*rho_jin;
                } 
            }
            // derivative of divTilt to rho
            rowvec divTilt_jout(3); divTilt_jout.fill(0);
            rowvec divTilt_jin(3);  divTilt_jin.fill(0);
            if ( param.includeDivTilt == true ){
                rowvec rho_1_jout = 1.0/h0 *( sf(j,1)*dout + h_1*dout_j + sf(j,0)*d_1out + h*d_1out_j );
                rowvec rho_2_jout = 1.0/h0 *( sf(j,2)*dout + h_2*dout_j + sf(j,0)*d_2out + h*d_2out_j );
                rowvec rho_1_jin  = 1.0/h0 *( -sf(j,1)*dout - sf(j,0)*d_1out );
                rowvec rho_2_jin  = 1.0/h0 *( -sf(j,2)*dout - sf(j,0)*d_2out ); 
                if ( whichLayer == 0 ){
                    rho_1_jout = -rho_1_jout;
                    rho_2_jout = -rho_2_jout;
                    rho_1_jin  = -rho_1_jin;
                    rho_2_jin  = -rho_2_jin;
                }
                double ddTilt_drho2 = 0.0;
                if ( rho <= eps ){ // E1 branch;
                    ddTilt_drho2 = 1.0/sqrt(3.0*rho*rho-2.0*rho) * ( 3.0 - pow(dTilt_drho,2.0) ); 
                }else if ( rho > eps ){ // E2 branch; Aims to make the energy function and its derivation continuous. 
                    ddTilt_drho2 = 2.0 * A1 * rho; 
                }
                divTilt_jout = ddTilt_drho2*rho_jout*(rho_1+rho_2) + dTilt_drho*(rho_1_jout+rho_2_jout);
                divTilt_jin  = ddTilt_drho2*rho_jin*(rho_1+rho_2) + dTilt_drho*(rho_1_jin+rho_2_jin);
            }          
            // curvature force
            fout.row(j) = fout.row(j) + kc*(2.0*Hcurv+divTilt-c0)*sqaout*(2.0*Hcurv_j+divTilt_jout) + 0.5*kc*pow(2.0*Hcurv+divTilt-c0,2.0)*sqaout_j;
            fin.row(j)  = fin.row(j)  + kc*(2.0*Hcurv+divTilt-c0)*sqaout*( divTilt_jin );
            // curvature-tilt force
            if ( param.includeDivTilt == true ){
                fout.row(j) = fout.row(j) + 0.5*kg*(2.0*Hcurv_j+divTilt_jout)*divTilt*sqaout + 0.5*kg*(2.0*Hcurv+divTilt)*divTilt_jout*sqaout + 0.5*kg*0.5*kg*(2.0*Hcurv+divTilt)*divTilt*sqaout_j;
                fin.row(j)  = fin.row(j)  + 0.5*kg*( divTilt_jin )*divTilt*sqaout + 0.5*kg*(2.0*Hcurv+divTilt)*divTilt_jin*sqaout; 
            }
            // area and volume constraint
            rowvec tempf_cons = n1_cons*sf(j,1) + n2_cons*sf(j,2);
            fout.row(j) = fout.row(j) + tempf_cons * sqaout;
            if ( param.isFlatMembrane == false ){
                rowvec tempf_conv = n1_conv*sf(j,1) + n2_conv*sf(j,2) + 1.0/3.0*uv*(V-V0)*dout*sf(j,0);
                fout.row(j) = fout.row(j) + tempf_conv*sqaout;
            }
        }
        /////////////////////////////////////////////////////////////
        rowvec threeEnergy(3); threeEnergy << Ebend << Eheight << EcurvTilt << endr;
        E = E + 1.0/2.0*gqcoeff(i)*threeEnergy;
        Fin = Fin + 1.0/2.0*gqcoeff(i)*fin;
        Fout = Fout + 1.0/2.0*gqcoeff(i)*fout;        
    }
}


void element_energy_force_thickness_irregularMinus(bool isInsertionPatch, int whichLayer, mat ori_dotsin, mat ori_dotsout, Param param, rowvec& E, mat& cunchu1, mat& cunchu2, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E.fill(0.0);
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
        rowvec e(3); e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes17in;    // element 2
        dotsout = M2*newnodes17out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes17in;    // element 3
        dotsout = M3*newnodes17out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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

void element_energy_force_thickness_irregularPlus(bool isInsertionPatch, int whichLayer, mat ori_dotsin, mat ori_dotsout, Param param, rowvec& E, mat& cunchu1, mat& cunchu2, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E.fill(0.0);
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
        rowvec e(3); e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m1m = strans(M1*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes19in;    // element 2
        dotsout = M2*newnodes19out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes19in;    // element 3
        dotsout = M3*newnodes19out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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

void element_energy_force_thickness_pseudoRegular1(bool isInsertionPatch, int whichLayer, mat ori_dotsin, mat ori_dotsout, Param param, rowvec& E, mat& cunchu1, mat& cunchu2, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E.fill(0.0);
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
        rowvec e(3); e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m1m = strans(M2f*Mf);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M3f*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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
            rowvec e(3); e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m4m = strans(M4i*matrix);
            cunchu1 = cunchu1 + m4m*f1;
            cunchu2 = cunchu2 + m4m*f2;
            E = E + e;

            dotsin = M2i*newnodes17in;    // element 2
            dotsout = M2i*newnodes17out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m2m = strans(M2i*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3i*newnodes17in;    // element 3
            dotsout = M3i*newnodes17out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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
            rowvec e(3); e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m1m = strans(M1c*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E = E + e;

            dotsin = M2c*newnodes19in;    // element 2
            dotsout = M2c*newnodes19out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m2m = strans(M2c*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3c*newnodes19in;    // element 3
            dotsout = M3c*newnodes19out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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

void element_energy_force_thickness_pseudoRegular2(bool isInsertionPatch, int whichLayer, mat ori_dotsin, mat ori_dotsout, Param param, rowvec& E, mat& cunchu1, mat& cunchu2, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
    // cunchu1 is the inlayer force; cunchu2 is the outlayer force.
    // initialize the output parameters
    E.fill(0.0);
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
        rowvec e(3); e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
        mat m1m = strans(M2f*Mf);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M3f*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e.fill(0.0);
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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
            rowvec e(3); e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m4m = strans(M4i*matrix);
            cunchu1 = cunchu1 + m4m*f1;
            cunchu2 = cunchu2 + m4m*f2;
            E = E + e;

            dotsin = M2i*newnodes17in;    // element 2
            dotsout = M2i*newnodes17out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m2m = strans(M2i*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3i*newnodes17in;    // element 3
            dotsout = M3i*newnodes17out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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
            rowvec e(3); e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m1m = strans(M1c*matrix);
            cunchu1 = cunchu1 + m1m*f1;
            cunchu2 = cunchu2 + m1m*f2;
            E = E + e;

            dotsin = M2c*newnodes19in;    // element 2
            dotsout = M2c*newnodes19out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
            mat m2m = strans(M2c*matrix);
            cunchu1 = cunchu1 + m2m*f1;
            cunchu2 = cunchu2 + m2m*f2;
            E = E + e;

            dotsin = M3c*newnodes19in;    // element 3
            dotsout = M3c*newnodes19out;
            e.fill(0.0);
            element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dotsin, dotsout, param, e, f1, f2, C0_spont, a0, gqcoeff, shape_functions);
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
// for in-layer: vertex1 is the mid-surface, vertex2 is the in-surface
void element_energy_force_thickness(bool isInsertionPatch, int whichLayer, Row<int> one_ring_nodes, Param param, rowvec& Ethick, mat& fthick, mat vertex1, mat vertex2, double C0_spont, double a0, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix){
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
        rowvec ethick(3); mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_regular(isInsertionPatch, whichLayer, dots1, dots2, param, ethick, fH1, fH2, C0_spont, a0, gqcoeff, shape_functions);
        if ( whichLayer == 2 ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else if ( whichLayer == 0 ){
            for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH2.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
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
        rowvec ethick(3); mat fH1(11,3); mat fH2(11,3); 
        element_energy_force_thickness_irregularMinus(isInsertionPatch, whichLayer, dots1, dots2, param, ethick, fH1, fH2, C0_spont, a0, gqcoeff, shape_functions, subMatrix);
        if (  whichLayer == 2 ){
            for (int j = 0; j < 11; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 11; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else if (whichLayer == 0){
             for (int j = 0; j < 11; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH2.row(j);
            } 
            for (int j = 0; j < 11; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
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
        rowvec ethick(3); mat fH1(13,3); mat fH2(13,3); 
        element_energy_force_thickness_irregularPlus(isInsertionPatch, whichLayer, dots1, dots2, param, ethick, fH1, fH2, C0_spont, a0, gqcoeff, shape_functions, subMatrix);
        if ( whichLayer == 2 ){
            for (int j = 0; j < 13; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 13; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else if (whichLayer == 0){
             for (int j = 0; j < 13; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH2.row(j);
            } 
            for (int j = 0; j < 13; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
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
        rowvec ethick(3); mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_pseudoRegular1(isInsertionPatch, whichLayer, dots1, dots2, param, ethick, fH1, fH2, C0_spont, a0, gqcoeff, shape_functions, subMatrix);
        if ( whichLayer == 2 ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else if (whichLayer == 0){
             for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH2.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
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
        rowvec ethick(3); mat fH1(12,3); mat fH2(12,3); 
        element_energy_force_thickness_pseudoRegular2(isInsertionPatch, whichLayer, dots1, dots2, param, ethick, fH1, fH2, C0_spont, a0,  gqcoeff, shape_functions, subMatrix);
        if ( whichLayer == 2 ){
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
            } 
            for (int j = 0; j < 12; j++) { // out-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + 2 * numvertex) = fthick.row(nodenum + 2 * numvertex) + fH2.row(j);
            } 
        }else if (whichLayer == 0){
             for (int j = 0; j < 12; j++) { // in-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum) = fthick.row(nodenum) + fH2.row(j);
            } 
            for (int j = 0; j < 12; j++) { // mid-layer of mesh
                int nodenum = one_ring_nodes(j);
                fthick.row(nodenum + numvertex) = fthick.row(nodenum + numvertex) + fH1.row(j);
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
void element_energy_force_tilt_regulars(mat dotsin, mat dotsout, Param param, double& E, mat& Fin, mat& Fout, rowvec gqcoeff, cube shape_functions){
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
        /////////////////////////////////////////////////////////////
        // energy
        rowvec h = xout - xin; 
        double hnorm = norm(h,2.0);
        rowvec n = h / hnorm;
        mat ndoutshu = n * strans(dout); double ndout = ndoutshu(0,0);
        mat ndinshu = n * strans(din); double ndin = ndinshu(0,0);
        rowvec tin = (-n)/(-ndin) - din;
        rowvec tout = n/ndout - dout; 
        mat eshu = 0.5*k*( tin*strans(tin)*sqain +  tout*strans(tout)*sqaout );// element tilt-energy
        e = eshu(0,0);
        /////////////////////////////////////////////////////////////
        // nodal force
        mat I = eye(3,3);
        mat n_x = 1.0/hnorm * ( I - kron(strans(h),h)/pow(hnorm,2.0));
        for (int j = 0; j < 12; j++) {
            mat din_xinj = - sf(j,1)*kron(strans(a1in),din) - sf(j,2)*kron(strans(a2in),din);
            mat n_xinj = - sf(j,0) * n_x;
            rowvec tmp = dout*n_xinj;
            mat tout_xinj = - 1.0/ndout/ndout*kron(strans(n),tmp) + 1.0/ndout*n_xinj;
            tmp = din*n_xinj + n*din_xinj;
            mat tin_xinj = - 1.0/ndin/ndin*kron(strans(n),tmp) + 1.0/ndin*n_xinj - din_xinj;
            rowvec sqain_xinj = (sf(j,1)*a1in + sf(j,2)*a2in) * sqain;
            fin.row(j) = 0.5*k*( 2.0*sqaout*tout*tout_xinj + 2.0*sqain*tin*tin_xinj + tin*strans(tin)*sqain_xinj);
            
            mat dout_xoutj = - sf(j,1)*kron(strans(a1out),dout) - sf(j,2)*kron(strans(a2out),dout);
            mat n_xoutj = sf(j,0) * n_x;
            tmp = dout*n_xoutj + n*dout_xoutj;
            mat tout_xoutj = - 1.0/ndout/ndout*kron(strans(n),tmp) + 1.0/ndout*n_xoutj - dout_xoutj;
            tmp = din*n_xoutj;
            mat tin_xoutj = - 1.0/ndin/ndin*kron(strans(n),tmp) + 1.0/ndin*n_xoutj; 
            rowvec sqaout_xoutj = (sf(j,1)*a1out + sf(j,2)*a2out) * sqaout;
            fout.row(j) = 0.5*k*( 2.0*sqaout*tout*tout_xoutj + tout*strans(tout)*sqaout_xoutj + 2.0*sqain*tin*tin_xoutj );
        }
        /////////////////////////////////////////////////////////////
        E = E + 1.0/2.0*gqcoeff(i)*e;
        Fin = Fin + 1.0/2.0*gqcoeff(i)*fin;
        Fout = Fout + 1.0/2.0*gqcoeff(i)*fout;
    }
}

void element_energy_force_tilt_irregulars(mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
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
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M4*matrix);
        cunchu1 = cunchu1 + m1m*f1;
        cunchu2 = cunchu2 + m1m*f2;
        E = E + e;

        dotsin = M2*newnodes17in;    // element 2
        dotsout = M2*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrix);
        cunchu1 = cunchu1 + m2m*f1;
        cunchu2 = cunchu2 + m2m*f2;
        E = E + e;

        dotsin = M3*newnodes17in;    // element 3
        dotsout = M3*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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

void element_energy_force_tilt_regular_irregular(mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
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
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m4m = strans(M4*matrixin);
        cunchu1 = cunchu1 + m4m*f1;
        mat m4mi = strans(M4i*matrixout);
        cunchu2 = cunchu2 + m4mi*f2;
        E = E + e;

        dotsin = M2*newnodes18in;    // element 2
        dotsout = M2i*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrixin);
        cunchu1 = cunchu1 + m2m*f1;
        mat m2mi = strans(M2i*matrixout);
        cunchu2 = cunchu2 + m2mi*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3i*newnodes17out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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

void element_energy_force_tilt_regular_complex(mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
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
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M1*matrixin);
        cunchu1 = cunchu1 + m1m*f1;
        mat m1mi = strans(M1i*matrixout);
        cunchu2 = cunchu2 + m1mi*f2;
        E = E + e;

        dotsin = M2*newnodes18in;    // element 2
        dotsout = M2i*newnodes19out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m2m = strans(M2*matrixin);
        cunchu1 = cunchu1 + m2m*f1;
        mat m2mi = strans(M2i*matrixout);
        cunchu2 = cunchu2 + m2mi*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3i*newnodes19out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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

void element_energy_force_tilt_regular_fakeregular(mat ori_dotsin, mat ori_dotsout, Param param, double& E, mat& cunchu1, mat& cunchu2, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix) {
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
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
        mat m1m = strans(M2*M);
        cunchu1 = cunchu1 + m1m*f1;
        mat m1mf = strans(M2f*Mf);
        cunchu2 = cunchu2 + m1mf*f2;
        E = E + e;

        dotsin = M3*newnodes18in;    // element 3
        dotsout = M3f*newnodes18out;
        e = 0.0;
        element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m4m = strans(M4*matrixin);
            cunchu1 = cunchu1 + m4m*f1;
            mat m4mi = strans(M4i*matrixout);
            cunchu2 = cunchu2 + m4mi*f2;
            E = E + e;

            dotsin = M2*newnodes18in;    // element 2
            dotsout = M2i*newnodes17out;
            e = 0.0;
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2*matrixin);
            cunchu1 = cunchu1 + m2m*f1;
            mat m2mi = strans(M2i*matrixout);
            cunchu2 = cunchu2 + m2mi*f2;
            E = E + e;

            dotsin = M3*newnodes18in;    // element 3
            dotsout = M3i*newnodes17out;
            e = 0.0;
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m1m = strans(M1*matrixin);
            cunchu1 = cunchu1 + m1m*f1;
            mat m1mc = strans(M1c*matrixout);
            cunchu2 = cunchu2 + m1mc*f2;
            E = E + e;

            dotsin = M2*newnodes18in;    // element 2
            dotsout = M2c*newnodes19out;
            e = 0.0;
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
            mat m2m = strans(M2*matrixin);
            cunchu1 = cunchu1 + m2m*f1;
            mat m2mc = strans(M2c*matrixout);
            cunchu2 = cunchu2 + m2mc*f2;
            E = E + e;

            dotsin = M3*newnodes18in;    // element 3
            dotsout = M3c*newnodes19out;
            e = 0.0;
            element_energy_force_tilt_regulars(dotsin, dotsout, param, e, f1, f2, gqcoeff, shape_functions);
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
// for the locally-finer mesh, don't use r-adaptive scheme, because it will make the suface unsmooth, especailly around the edge of the local-finer area.
void energy_force_regularization(mat vertex, mat vertexold, Mat<int> face, Param param, double& E_regularization, mat& Fre, rowvec& deformnumbers){
    double k  = param.k_regularization;
    //Fre.fill(0.0);
    E_regularization = 0.0;
    deformnumbers.fill(0);
    int deformnumber_shape = 0;
    int deformnumber_area = 0;
    rowvec Ere(face.n_rows); Ere.fill(0.0);
    mat fre(vertex.n_rows,3); fre.fill(0.0);
    #pragma omp parallel for reduction(+:fre)
    for (int i = 0; i < face.n_rows; i++){    
        if ( param.isInsertionPatch.outlayer[i] == true ){
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
        double a0 = areaold;     

        bool isDeformShape = false;
        bool isDeformArea = false;

        if ( param.usingRpi == true ){
            if ( gama > param.gama_shape && param.isLocallyFinerFace[i] == false ){
                isDeformShape = true;
            }
            if ( abs(area-a0)/a0 >= param.gama_area  && param.isLocallyFinerFace[i] == false ){
                //isDeformArea = true;
            }
        }
     
        if ( isDeformShape == false && isDeformArea == false ){
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
// the following functions for calculating the area and volume of the sphere
// here, A is the one-ring-nodes
void cell_area_volume(Mat<int> face, mat vertex, Mat<int> A, int GaussQuadratureN, rowvec& element_area, rowvec& element_volume, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, int n){ // A is face_ring_vertex 
    element_area.fill(0);
    element_volume.fill(0);
    mat VWU = setVMU(GaussQuadratureN); 
    #pragma omp parallel for
    for (int i = 0; i < face.n_rows; i++){
        double area = 0.0;
        double volume = 0.0;

        // regular patch
        if  ( A(i,12) == -1 ){ // regular patch
            mat dots(12,3); //dots(12,3); 12 nodes
            for (int j = 0; j < 12; j++){
                int nodenum = A(i,j);
                dots.row(j) = vertex.row(nodenum);
            }
            for (int j = 0; j < VWU.n_rows; j++){
                mat sf = shape_functions.slice(j);
                rowvec x(3); trans_time(sf.col(0),dots,x); 
                rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                double  sqa = norm(cross(a_1,a_2),2); 
                rowvec d(3); d = cross(a_1,a_2)/sqa; 
                double s = sqa; 
                double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                double v = 1.0/3.0*shu*sqa;
                area = area + 1.0/2.0*gqcoeff(j)*s; 
                volume = volume + 1.0/2.0*gqcoeff(j)*v;
            }

        // irregular-minus patch    
        }else if ( A(i,12) == -2 ){ 
            mat ori_dots(11,3); // ori_dots(11,3);
            for (int j = 0; j < 11; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            } 
            mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17); 
            //subdivision_matrix(M, M1, M2, M3, M4);  
            M = subMatrix.irregM; M1 = subMatrix.irregM1; M2 = subMatrix.irregM2; M3 = subMatrix.irregM3; M4 = subMatrix.irregM4;
            for (int j = 0; j < n; j++){
                mat newnodes17 = M*ori_dots; // 17 new nodes
                /////////////////////////////////////////////////
                // element 4
                mat dots = M4*newnodes17; // dots(12,3);
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec x(3); trans_time(sf.col(0),dots,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec x(3); trans_time(sf.col(0),dots,x);  
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes17;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec x(3); trans_time(sf.col(0),dots,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                mat dots1 = M1*newnodes17;    // element 1, still irregular patch
                ori_dots = dots1;
            }

        // irregular-plus patch    
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
                    rowvec x(3); trans_time(sf.col(0),dots,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                ///////////////////////////////////////////////
                // element 2 
                dots = M2*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec x(3); trans_time(sf.col(0),dots,x);  
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                //////////////////////////////////////////////////////
                // element 3
                dots = M3*newnodes19;  
                for (int k =0; k < VWU.n_rows; k++){
                    mat sf = shape_functions.slice(k);
                    rowvec x(3); trans_time(sf.col(0),dots,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                    double sqa = norm(cross(a_1,a_2),2); 
                    rowvec d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(k)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(k)*v;
                }
                mat dots4 = M4*newnodes19;    // element 4, still complex patch
                ori_dots = dots4;
            }

        // pseudo-regular patch    
        }else if ( A(i,12) == -4 || A(i,12) == -5 ){ 
            mat MM(18,12); mat MM1(11,18); mat MM2(12,18); mat MM3(12,18); mat MM4(13,18); 
            if (  A(i,12) == -4 ){
                MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
            }else if (  A(i,12) == -5 ){
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
                    rowvec x(3); trans_time(sf.col(0),dots2,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots2,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots2,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    rowvec d(3); d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(j)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(j)*v;
                }
                mat dots3 = MM3*newnodes18; // dots(12,3);
                for (int j = 0; j < VWU.n_rows; j++){
                    mat sf = shape_functions.slice(j);
                    rowvec x(3); trans_time(sf.col(0),dots3,x); 
                    rowvec a_1(3); trans_time(sf.col(1), dots3,a_1); 
                    rowvec a_2(3); trans_time(sf.col(2),dots3,a_2); 
                    double  sqa = norm(cross(a_1,a_2),2); 
                    rowvec d(3); d = cross(a_1,a_2)/sqa; 
                    double s = sqa; 
                    double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                    double v = 1.0/3.0*shu*sqa;
                    area = area + 1.0/2.0*gqcoeff(j)*s; 
                    volume = volume + 1.0/2.0*gqcoeff(j)*v;
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
                        rowvec x(3); trans_time(sf.col(0),dots,x); 
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2); 
                        rowvec d = cross(a_1,a_2)/sqa; 
                        double s = sqa; 
                        double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                        double v = 1.0/3.0*shu*sqa;
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                        volume = volume + 1.0/2.0*gqcoeff(k)*v;
                    }
                    ///////////////////////////////////////////////
                    // element 2 
                    dots = M2*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);
                        rowvec x(3); trans_time(sf.col(0),dots,x);  
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2); 
                        rowvec d = cross(a_1,a_2)/sqa; 
                        double s = sqa; 
                        double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                        double v = 1.0/3.0*shu*sqa;
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                        volume = volume + 1.0/2.0*gqcoeff(k)*v;
                    }
                    //////////////////////////////////////////////////////
                    // element 3
                    dots = M3*newnodes17;  
                    for (int k =0; k < VWU.n_rows; k++){
                        mat sf = shape_functions.slice(k);
                        rowvec x(3); trans_time(sf.col(0),dots,x); 
                        rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                        rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                        double sqa = norm(cross(a_1,a_2),2); 
                        rowvec d = cross(a_1,a_2)/sqa; 
                        double s = sqa; 
                        double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                        double v = 1.0/3.0*shu*sqa;
                        area = area + 1.0/2.0*gqcoeff(k)*s; 
                        volume = volume + 1.0/2.0*gqcoeff(k)*v;
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
                            rowvec x(3); trans_time(sf.col(0),dots,x); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1);  
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            rowvec d = cross(a_1,a_2)/sqa; 
                            double s = sqa; 
                            double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                            double v = 1.0/3.0*shu*sqa;
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                            volume = volume + 1.0/2.0*gqcoeff(k)*v;
                        }
                        ///////////////////////////////////////////////
                        // element 2 
                        dots = M2*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k);
                            rowvec x(3); trans_time(sf.col(0),dots,x);  
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            rowvec d = cross(a_1,a_2)/sqa; 
                            double s = sqa; 
                            double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                            double v = 1.0/3.0*shu*sqa;
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                            volume = volume + 1.0/2.0*gqcoeff(k)*v;
                        }
                        //////////////////////////////////////////////////////
                        // element 3
                        dots = M3*newnodes19;  
                        for (int k =0; k < VWU.n_rows; k++){
                            mat sf = shape_functions.slice(k);
                            rowvec x(3); trans_time(sf.col(0),dots,x); 
                            rowvec a_1(3); trans_time(sf.col(1), dots,a_1); 
                            rowvec a_2(3); trans_time(sf.col(2),dots,a_2); 
                            double sqa = norm(cross(a_1,a_2),2); 
                            rowvec d = cross(a_1,a_2)/sqa; 
                            double s = sqa; 
                            double shu = x(0)*d(0) + x(1)*d(1) + x(2)*d(2);
                            double v = 1.0/3.0*shu*sqa;
                            area = area + 1.0/2.0*gqcoeff(k)*s; 
                            volume = volume + 1.0/2.0*gqcoeff(k)*v;
                        }
                        mat dots4 = M4*newnodes19;    // element 4, still complex patch
                        ori_dots = dots4;
                    }
                }
            }
        }
        element_area(i) = area;
        element_volume(i) = volume;
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

// here, A is the one-ring-nodes
double LocalAreaConstraintEnergy(int whichLayer, mat vertex, Mat<int> face, Mat<int> A, Param param, vector<bool> Isinsertionpatch, rowvec gqcoeff, cube shape_functions, SubMatrix subMatrix, rowvec elementsqa){ // A is face_ring_vertex 
    double energy = 0.0;
    int GaussQuadratureN = param.GaussQuadratureN;
    mat VWU = setVMU(GaussQuadratureN); 
    double us;
    if (  whichLayer == 0 ){
        us = param.us_in;
    }else if(  whichLayer == 2 ){
        us = param.us_out;
    }
    int n = param.subDivideTimes;
    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < face.n_rows; i++){
        //if ( Isinsertionpatch[i] == true ) continue;
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

        // irregular-minus patch   
        }else if ( A(i,12) == -2 ){ 
            mat ori_dots(11,3); 
            for (int j = 0; j < 11; j++){
                int nodenum = A(i,j);
                ori_dots.row(j) = vertex.row(nodenum);
            } 
            mat M(17,11); mat M1(11,17); mat M2(12,17); mat M3(12,17); mat M4(12,17); 
            //subdivision_matrix(M, M1, M2, M3, M4);  
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

        // irregular-plus patch    
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

        // pseudo-regular patches     
        }else if ( A(i,12) == -4 || A(i,12) == -5 ){ 
            mat MM(18,12); mat MM1(11,18); mat MM2(12,18); mat MM3(12,18); mat MM4(13,18); 
            if (  A(i,12) == -4 ){
                MM = subMatrix.sudoreg1M; MM1 = subMatrix.sudoreg1M1; MM2 = subMatrix.sudoreg1M2; MM3 = subMatrix.sudoreg1M3; MM4 = subMatrix.sudoreg1M4;
            }else if (  A(i,12) == -5 ){
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

rowvec determine_spontaneous_curvature(int whichLayer, Param param, Mat<int> face, mat vertex){
    rowvec spontcurv(face.n_rows); 
    Mat<int> insertionpatch = param.insertionpatch;
    if (whichLayer == 2){ // outlayer
        double c0_ins = param.c0out_ins;
        double c0 = param.c0out;
        spontcurv.fill(c0);
        for (int i = 0; i < insertionpatch.n_rows; i++){
            for (int j = 0; j < insertionpatch.n_cols; j++){
                int facenumber = insertionpatch(i,j);
                spontcurv(facenumber) = c0_ins;
            }
        }
    }else{
        double c0 = param.c0in;
        spontcurv.fill(c0);
    }
    
    return spontcurv;
}

void insertion_shape_constraint(int whichLayer, Param param, Mat<int> faceout, mat vertexout, double& Einsert, mat& finsert){
    Mat<int> insertionpatch = param.insertionpatch;
    if (insertionpatch.n_cols > 0 && insertionpatch.n_rows > 0 ){
        double k = param.K_insertShape;
        for ( int i = 0; i < insertionpatch.n_rows; i++ ){
            for ( int j = 0; j < insertionpatch.n_cols; j++ ){
                int facetmp = insertionpatch(i,j);
                int node0 = faceout(facetmp,0); // three nodes of this face element
                int node1 = faceout(facetmp,1);
                int node2 = faceout(facetmp,2);
                rowvec vector0 = vertexout.row(node0) - vertexout.row(node1);  double side0 = norm(vector0,2.0);
                rowvec vector1 = vertexout.row(node1) - vertexout.row(node2);  double side1 = norm(vector1,2.0);
                rowvec vector2 = vertexout.row(node2) - vertexout.row(node0);  double side2 = norm(vector2,2.0);
                double meanside = param.insertionShapeEdgeLength(whichLayer);
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
            double meanside = param.insertionShapeEdgeLength(whichLayer);
            Einsert = Einsert + k/2.0*(pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0));
            finsert.row(node0) = finsert.row(node0) + k*( (side0-meanside)*(-vector0/side0) + (side2-meanside)*(vector2/side2) );
            finsert.row(node1) = finsert.row(node1) + k*( (side1-meanside)*(-vector1/side1) + (side0-meanside)*(vector0/side0) );
            finsert.row(node2) = finsert.row(node2) + k*( (side2-meanside)*(-vector2/side2) + (side1-meanside)*(vector1/side1) );
        }
        finsert = -finsert;
    }
}
/*
void insertion_shape_constraint_IN(Param param, Mat<int> facein, mat vertexin, double& Einsert, mat& finsert){
    Mat<int> insertionVertex = param.insertionVertex;
    Mat<int> insertionpatch = param.insertionpatch;
    //if ( param.usingInsertNick == true && insertionVertex.n_cols > 2 && insertionVertex.n_rows > 0 )
    //{ 
    //}else 
    if (param.usingInsertNick == false && insertionpatch.n_cols > 0 && insertionpatch.n_rows > 0 ){
        double k = param.K;
        double Radiusout = sqrt(param.S0mid/4.0/M_PI) + param.thickness/2.0;
        double Radiusin  = Radiusout - param.thickness;
        double meanside = param.meanL * Radiusin/Radiusout;
        for ( int i = 0; i < insertionpatch.n_rows; i++ ){
            for ( int j = 0; j < insertionpatch.n_cols; j++ ){
                int facetmp = insertionpatch(i,j);
                int node0 = facein(facetmp,0); // three nodes of this face element
                int node1 = facein(facetmp,1);
                int node2 = facein(facetmp,2);
                rowvec vector0 = vertexin.row(node0) - vertexin.row(node1);  double side0 = norm(vector0,2.0);
                rowvec vector1 = vertexin.row(node1) - vertexin.row(node2);  double side1 = norm(vector1,2.0);
                rowvec vector2 = vertexin.row(node2) - vertexin.row(node0);  double side2 = norm(vector2,2.0);
                Einsert = Einsert + k/2.0*(pow(side0-meanside,2.0) + pow(side1-meanside,2.0) + pow(side2-meanside,2.0));
                finsert.row(node0) = finsert.row(node0) + k*( (side0-meanside)*(-vector0/side0) + (side2-meanside)*(vector2/side2) );
                finsert.row(node1) = finsert.row(node1) + k*( (side1-meanside)*(-vector1/side1) + (side0-meanside)*(vector0/side0) );
                finsert.row(node2) = finsert.row(node2) + k*( (side2-meanside)*(-vector2/side2) + (side1-meanside)*(vector1/side1) );
            }
        }
        finsert = -finsert;
    } 
}
*/
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// calculate the thickness
mat calculate_thickness(int whichLayer, Mat<int> face, mat vertexin, mat vertexout, Mat<int> one_ring_nodes, Param param, cube shape_functions, SubMatrix subMatrix){ // A is face_ring_vertex 
    double h0 = param.thickness_out;
    if ( whichLayer == 0 ){ // in-layer
        h0 = param.thickness_in;
    }

    mat thickness(face.n_rows,2); thickness.fill(-1.0);

    int GaussQuadratureN = param.GaussQuadratureN;
    //mat VWU = setVMU(GaussQuadratureN); 
    mat sf = shape_functions.slice(0);

    //#pragma omp parallel for
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
        if ( whichLayer == 2 ){
            hnorm_matrix = h * strans(dout);
        }else if (whichLayer == 0){
            hnorm_matrix = h * strans(din);
        }
        
       double hnorm = hnorm_matrix(0,0); // observed height

        //mat shu = aout1*strans(dout_1) + aout2*strans(dout_2);
        //double H_curv_out = 0.5*shu(0,0); // mean curvature
        double h0_curv = h0; //h0 * (1.0 + h0*(1.0*H_curv_out)); // curvature-modified height. NOTE: it is NOT h0*(1.0-h0*(2.0*H_curv_out))    
        
        if (param.isInsertionPatch.outlayer[i] == true && whichLayer == 2 ){
            h0_curv = h0 - param.insert_dH0;
        }

        thickness(i,0) = h0_curv;
        thickness(i,1) = hnorm;
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