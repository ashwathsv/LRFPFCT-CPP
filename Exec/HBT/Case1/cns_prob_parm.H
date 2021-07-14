#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_0 = 1.0;
    amrex::Real p_r = 1.0;
    amrex::Real rho_0 = 1.0;
    amrex::Real rho_r = 1.0;
    amrex::Real u_0 = 0.0;
    amrex::Real u_r = 1.0;
    amrex::Real v_0 = 0.0;
    amrex::Real v_r = 0.0;
    amrex::Real width = 0.1;
    amrex::Real Mach_shock = 2.0;
    amrex::Real Mach_in = 2.0;  
    amrex::Real inflow_time = 0.0;
    amrex::Real gamma = 1.4;
    amrex::Real Rsp = amrex::Real(287.058);
    amrex::Real sh_speed = 0.0;

    std::string pres_file = "pressuredata.txt";
};

#endif
