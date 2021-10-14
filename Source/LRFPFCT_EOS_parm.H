#ifndef LRFPFCT_EOS_PARM_H_
#define LRFPFCT_EOS_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>


// The code runs on SI units. However, the user needs to input data corresponding to the 
// units specified in the input files

struct EOSParm
{
    const amrex::Real Ru = amrex::Real(8.314462618); // in J K^-1 mol^-1

    // Specifying gamma and the molecular weight of the gas
    amrex::Real eos_gamma = 1.4;
    amrex::Real eos_mu = 0.02897;  // mean molecular weight in kg/mol
    
    // Specific gas constant
    amrex::Real Rsp = Ru/eos_mu;

    // Specifying variable for heat release
    amrex::Real q_nd = 0.0;
    amrex::Real q_dim = 0.0;

    // Specififying kappa_0 ( kappa = kappa_0 * Cp * T^(0.7) )
    amrex::Real kappa_0 = 2.54e-5; // units in CGS

    // Specifying activation energy and Arrhenius pre-exponential
    amrex::Real Ea_nd = 1e10;
    amrex::Real Ea_dim = 1e20;
    amrex::Real pre_exp = 0.0;
    amrex::Real temp_exp = 0.7;

    // This is the reference temperature used to dimensionalize the heat release and activation energy
    amrex::Real Tref = 298.0;
    amrex::Real pref = 1.01325e5;

    amrex::Real cv;
    amrex::Real cp;

    amrex::Real Pr  = 1.0;     // Prandtl number
    amrex::Real Le  = 1.0;      // Lewis number
    amrex::Real C_S = 1.458e-5; // constant in Sutherland's law
    amrex::Real T_S = 110.4;    // Sutherland temperature

    void Initialize ();

};

#endif
