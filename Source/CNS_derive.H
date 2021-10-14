#ifndef CNS_DERIVE_H_
#define CNS_DERIVE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

void cns_dervel (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dermac     (const amrex::Box& bx, amrex::FArrayBox& macfab, int dcomp, int ncomp,
                    const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                    amrex::Real time, const int* bcrec, int level);
#ifdef LRFPFCT_REACTION
void cns_dermassfrac (const amrex::Box& bx, amrex::FArrayBox& yfab, int dcomp, int ncomp,
                    const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                    amrex::Real time, const int* bcrec, int level);
#endif

#endif
