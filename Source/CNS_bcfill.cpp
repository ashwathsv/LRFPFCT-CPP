
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include "CNS.H"
#include "cns_prob.H"

using namespace amrex;

struct CnsFillExtDir
{
    ProbParm const* lprobparm;

    AMREX_GPU_HOST
    constexpr explicit CnsFillExtDir(const ProbParm* d_prob_parm)
        : lprobparm(d_prob_parm)
    {
    }

    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& data,
                     const int /*dcomp*/, const int numcomp,
                     GeometryData const& geom, const Real time,
                     const BCRec* bcr, const int /*bcomp*/,
                     const int /*orig_comp*/) const
        {
            using namespace amrex;

            const int* domlo = geom.Domain().loVect();
            const int* domhi = geom.Domain().hiVect();
            const amrex::Real* prob_lo = geom.ProbLo();
            // const amrex::Real* prob_hi = geom.ProbHi();
            const amrex::Real* dx = geom.CellSize();
            const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
            prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
            prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
            prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};

            const int* bc = bcr->data();

            amrex::Real s_int[NGROW][NCONS] = {0.0};
            amrex::Real s_ext[NCONS] = {0.0};

             // xlo and xhi
            int idir = 0;
            if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
                for (int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(domlo[idir]+ng, iv[1], iv[2]));
                    for (int n = 0; n < NCONS; n++) {
                        s_int[ng][n] = data(loc, n);
                    }
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[idir], -1, time, geom, *lprobparm);
                for (int n = 0; n < NCONS; n++) {
                    data(iv, n) = s_ext[n];
                }
            } else if (
              (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
              (iv[idir] > domhi[idir])) {
                for(int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(domhi[idir]-ng, iv[1], iv[2]));
                    for (int n = 0; n < NCONS; n++) {
                        s_int[ng][n] = data(loc, n);
                    }                    
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[idir], 1, time, geom, *lprobparm);
                for (int n = 0; n < NCONS; n++) {
                    data(iv, n) = s_ext[n];
                }
            }
#if AMREX_SPACEDIM > 1
            // ylo and yhi
            idir = 1;
            if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
                for(int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir]+ng, iv[2]));
                    for (int n = 0; n < NCONS; n++) {
                        s_int[ng][n] = data(loc, n);
                    }
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[idir], -1, time, geom, *lprobparm);
                for (int n = 0; n < NCONS; n++) {
                    data(iv, n) = s_ext[n];
                }
            } else if (
            (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
            (iv[idir] > domhi[idir])) {
                for(int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir]-ng, iv[2]));
                    for (int n = 0; n < NCONS; n++) {
                        s_int[ng][n] = data(loc, n);
                    }
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[idir], 1, time, geom, *lprobparm);
                for (int n = 0; n < NCONS; n++) {
                    data(iv, n) = s_ext[n];
                }
            }
#endif          
        }
};

// bx                  : Cells outside physical domain and inside bx are filled.
// data, dcomp, numcomp: Fill numcomp components of data starting from dcomp.
// bcr, bcomp          : bcr[bcomp] specifies BC for component dcomp and so on.
// scomp               : component index for dcomp as in the desciptor set up in CNS::variableSetUp.

void cns_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
    // Print() << "entered cns_bcfill()\n";
    const ProbParm* lprobparm = CNS::d_prob_parm;
    GpuBndryFuncFab<CnsFillExtDir> gpu_bndry_func(CnsFillExtDir{lprobparm});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}