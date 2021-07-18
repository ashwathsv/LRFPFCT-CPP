
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

        pp.query("p_l", CNS::h_prob_parm->p_l);
        pp.query("p_r", CNS::h_prob_parm->p_r);
        pp.query("rho_l", CNS::h_prob_parm->rho_l);
        pp.query("rho_r", CNS::h_prob_parm->rho_r);
        pp.query("u_l", CNS::h_prob_parm->u_l);
        pp.query("u_r", CNS::h_prob_parm->u_r);
        pp.query("v_l", CNS::h_prob_parm->v_l);
        pp.query("v_r", CNS::h_prob_parm->v_r);
#if AMREX_SPACEDIM==3
        pp.query("w_l", CNS::h_prob_parm->w_l);
        pp.query("w_r", CNS::h_prob_parm->w_r);
#endif
        pp.query("Y_l", CNS::h_prob_parm->Y_l);
        pp.query("Y_r", CNS::h_prob_parm->Y_r);

        pp.query("width", CNS::h_prob_parm->width);

        std::string geom_type;
        pp.query("geom_type", geom_type);

        if(geom_type == "sphere") CNS::h_prob_parm->geom_type_flag = 1;

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
                         CNS::d_prob_parm);
        
    }
}
