#ifndef CNS_PARM_H_
#define CNS_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct Parm
{
    // const amrex::Real Ru = amrex::Real(8.314462618); // in J K^-1 mol^-1
    // amrex::Real eos_gamma = 1.4;
    // amrex::Real eos_mu = 0.02897;  // mean molecular weight in kg/mol

    // amrex::Real cv;
    // amrex::Real cp;

    // amrex::Real Pr  = 0.72;     // Prandtl number
    // amrex::Real C_S = 1.458e-5; // constant in Sutherland's law
    // amrex::Real T_S = 110.4;    // Sutherland temperature

    amrex::Real smallr = 1.e-19;
    amrex::Real smallp = 1.e-10;

    amrex::Real minro = 1E-10;
    amrex::Real minp  = 1.0;
    amrex::Real minT  = 100; 

    // amrex::Real minrofrac = 1e-4;
    // amrex::Real minpfrac  = 1e-4;

    // int do_minp = 0;

    int scaledt_pres = 0;

#ifdef LRFPFCT_REACTION
    amrex::Real start_sfoil_time = 0.0;
#endif
    
    void Initialize ();
};

#endif
