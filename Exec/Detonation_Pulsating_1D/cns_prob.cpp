
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

        pp.query("width", CNS::h_prob_parm->width);

        pp.query("Y1", CNS::h_prob_parm->Y1);
        pp.query("Y0", CNS::h_prob_parm->Y0);

        pp.query("T1", CNS::h_prob_parm->T1);
        pp.query("T0", CNS::h_prob_parm->T0);

        pp.query("u1", CNS::h_prob_parm->u1);
        pp.query("u0", CNS::h_prob_parm->u0);


        pp.query("p1", CNS::h_prob_parm->p1);
        pp.query("p0", CNS::h_prob_parm->p0);

        pp.get("pressure_file", CNS::h_prob_parm->pres_file);
        pp.get("print_pres_data", CNS::h_prob_parm->print_pres_data);
        pp.get("print_append", CNS::h_prob_parm->print_append);
        pp.get("pres_int", CNS::h_prob_parm->pres_int);
        pp.get("get_maxgrad", CNS::h_prob_parm->get_maxgrad);

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#endif

    }
}
