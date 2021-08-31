
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        pp.get("rho_0", CNS::h_prob_parm->rho_0);
        pp.get("rho_r", CNS::h_prob_parm->rho_r);
        pp.get("p_0", CNS::h_prob_parm->p_0);
        pp.get("p_r", CNS::h_prob_parm->p_r);
        pp.query("u_0", CNS::h_prob_parm->u_0);
        pp.query("u_r", CNS::h_prob_parm->u_r);
        pp.query("v_0", CNS::h_prob_parm->v_0);
        pp.get("width", CNS::h_prob_parm->width);
        pp.get("exit_width", CNS::h_prob_parm->exit_width);
        pp.get("Mach_shock", CNS::h_prob_parm->Mach_shock);
        pp.get("Mach_in", CNS::h_prob_parm->Mach_in);
        pp.get("inflow_time", CNS::h_prob_parm->inflow_time);

        // get y and z coordinates for the center of the tube
        pp.get("ycent", CNS::h_prob_parm->ycent);
#if AMREX_SPACEDIM==3
        pp.get("zcent", CNS::h_prob_parm->zcent);
#endif

        pp.get("pressure_file", CNS::h_prob_parm->pres_file);

        amrex::Real a1 = sqrt(CNS::h_prob_parm->gamma*CNS::h_prob_parm->p_0/CNS::h_prob_parm->rho_0);
        CNS::h_prob_parm->sh_speed = CNS::h_prob_parm->Mach_shock*a1;

        Print() << "sh_speed (w) = " << CNS::h_prob_parm->sh_speed
                << ", u_0 = " << CNS::h_prob_parm->sh_speed*
                ( Real(1.0) - (Real(1.0)/CNS::h_prob_parm->rho_r) )<< "\n";

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
        // amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
        //                  CNS::d_prob_parm);
#endif

    }
}
