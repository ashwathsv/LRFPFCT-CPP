
#include <CNS_parm.H>

void Parm::Initialize ()
{
    constexpr amrex::Real Ru = amrex::Real(8.314462618); // in J K^-1 mol^-1
    cv = Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    cp = eos_gamma * Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
}
