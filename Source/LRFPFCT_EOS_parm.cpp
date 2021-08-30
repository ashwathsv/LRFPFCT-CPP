
#include <LRFPFCT_EOS_parm.H>

void EOSParm::Initialize ()
{
    constexpr amrex::Real Ru = amrex::Real(8.314462618); // in J K^-1 mol^-1
    Rsp = Ru / eos_mu;
    cv = Rsp / ((eos_gamma-amrex::Real(1.0)));
    cp = eos_gamma * Rsp / ((eos_gamma-amrex::Real(1.0)));
}