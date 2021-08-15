#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real width = 0.1;
    amrex::Real p1 = 80*1.01325e5;
    amrex::Real p0 = 1.01325e5;
    amrex::Real u1 = 0.0;
    amrex::Real u0 = 0.0;
    amrex::Real T1 = 3000;
    amrex::Real T0 = 298; 
    amrex::Real Y1 = 1.0;
    amrex::Real Y0 = 0.0;
};

#endif
