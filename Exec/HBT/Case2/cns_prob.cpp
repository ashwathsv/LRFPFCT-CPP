
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
        pp.get("Mach_shock", CNS::h_prob_parm->Mach_shock);
        pp.get("Mach_in", CNS::h_prob_parm->Mach_in);
        pp.get("inflow_time", CNS::h_prob_parm->inflow_time);
        pp.get("out_width", CNS::h_prob_parm->out_width);
        pp.get("out_loend", CNS::h_prob_parm->out_loend);
        pp.get("iprobe", CNS::h_prob_parm->iprobe);
        pp.get("jprobe", CNS::h_prob_parm->jprobe);
        pp.get("isout_lox", CNS::h_prob_parm->isout_lox);

        pp.get("pressure_file", CNS::h_prob_parm->pres_file);

        Print() << "pres_file = " << CNS::h_prob_parm->pres_file << "\n";
 
#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
        // amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
        //                  CNS::d_prob_parm);
#endif

    }
}
