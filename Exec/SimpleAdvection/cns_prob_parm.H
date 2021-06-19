#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_l = 1.0;
    amrex::Real p_r = 1.0;
    amrex::Real rho_l = 1.0;
    amrex::Real rho_r = 0.125;
    amrex::Real u_l = 1.0;
    amrex::Real u_r = 1.0;
    amrex::Real v_l = 0.0;
    amrex::Real v_r = 0.0;
    amrex::Real width = 0.1;
    std::string geom_type;

#if AMREX_SPACEDIM==3
    amrex::Real w_l = 0.0;
    amrex::Real w_r = 0.0;
#endif
    
#ifdef AMREX_USE_GPU
    int geom_type_flag = 0;
#endif
};

#endif
