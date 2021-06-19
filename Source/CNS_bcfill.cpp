
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct CnsFillExtDir
{
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& data,
                     const int /*dcomp*/, const int numcomp,
                     GeometryData const& geom, const Real /*time*/,
                     const BCRec* bcr, const int /*bcomp*/,
                     const int /*orig_comp*/) const
        {
            using namespace amrex;
//             const Box& domain = geom.Domain();

//             const auto domlo = amrex::lbound(domain);
//             const auto domhi = amrex::ubound(domain);

//             const int ilo = domlo.x;
//             const int ihi = domhi.x;

// #if AMREX_SPACEDIM >= 2
//             const int jlo = domlo.y;
//             const int jhi = domhi.y;
// #if AMREX_SPACEDIM==2
//             const int k = domlo.z;
// #endif
// #if AMREX_SPACEDIM==3
//             const int k = iv[2];
//             const int klo = domlo.z;
//             const int khi = domhi.z;
// #endif
// #endif 
//             const int i = iv[0];
//             const int j = iv[1];

//     for (int n = 0; n < numcomp; ++n){
//         Array4<Real> q(data,n);
//         BCRec const& bc = bcr[n];
//         if (i < ilo) {
//             // if (bc.lo(0) == BCType::int_dir) {
//             //     q(i,j,k) = q(ihi+i+1,j,k);
//             // }
//         }
//         if (i > ihi) {
//             // if (bc.hi(0) == BCType::int_dir) {
//             //     q(i,j,k) = q(ilo+ihi-i-1,j,k);
//             // } 
//         }
// #if AMREX_SPACEDIM >= 2
//         if (j < jlo) {
//             if (bc.lo(1) == BCType::int_dir) {
//                 q(i,j,k) = q(i,jhi+j+1,k);
//             }          
//         }

//         if (iv[1] > jhi) {
//             // if (bc.hi(1) == BCType::int_dir) {
//             //     q(i,j,k) = q(i,jlo+jhi-j-1,k);
//             // } 
//         }

// #if AMREX_SPACEDIM==3
//         if (k < klo) {
//             // if (bc.lo(2) == BCType::int_dir) {
//             //     q(i,j,k) = q(i,j,)
//             // }            
//         }

//         if (k > khi) {
//             // if (bc.hi(2) == BCType::int_dir) {
//             //     // Do nothing.
//             // } 
//         }
// #endif
// #endif              
//     }           
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
    Print() << "entered cns_bcfill()\n";
    GpuBndryFuncFab<CnsFillExtDir> gpu_bndry_func(CnsFillExtDir{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}

// // Function to fill physical domain boundary (fill ghost cells)
// AMREX_FORCE_INLINE
// void 
// FillDomBoundary (amrex::MultiFab& phi, const amrex::Geometry& geom, 
//                  const amrex::Vector<amrex::BCRec>& bc, amrex::Real cur_time)
// {
//     using namespace amrex;
//     BL_PROFILE_VAR("FillDomainBoundary()", dombndry);
//     // int myproc = ParallelDescriptor::MyProc();
//     // Print(myproc) << "rank= " << myproc << ", entered LRFPFCT::FillDomainBoundary()" << "\n";
//     if (geom.isAllPeriodic()) return;
//     if (phi.nGrow() == 0) return;

//     AMREX_ALWAYS_ASSERT(phi.ixType().cellCentered());

//     // Print() << " entered FillDomBoundary() " << "\n";

// #if !(defined(AMREX_USE_CUDA) && defined(AMREX_USE_GPU_PRAGMA) && defined(AMREX_GPU_PRAGMA_NO_HOST))
//     if (Gpu::inLaunchRegion())
//     {
// #endif  
//         GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
//         PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbcf
//             (geom, bc, gpu_bndry_func);
//         physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), cur_time, 0);
//         // Print(myproc) << "rank= " << myproc << ", reached GpuBndryFuncFab()" << "\n";
// #if !(defined(AMREX_USE_CUDA) && defined(AMREX_USE_GPU_PRAGMA) && defined(AMREX_GPU_PRAGMA_NO_HOST))
//     }
//     else
//     {
//         CpuBndryFuncFab cpu_bndry_func(nullptr);;
//         PhysBCFunct<CpuBndryFuncFab> physbcf(geom, bc, cpu_bndry_func);
//         physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), cur_time, 0);
//         // Print(myproc) << "rank= " << myproc << ", reached CpuBndryFuncFab()" << "\n";
//     }
// #endif
// }

// void FillDomBoundary (MultiFab& phi, const Geometry& geom, const Vector<BCRec>& bc)
// {
//     if (phi.nGrow() == 0) return;

//     AMREX_ALWAYS_ASSERT(phi.ixType().cellCentered());

    // if (Gpu::inLaunchRegion())
    // {
        // GpuBndryFuncFab<CnsFillExtDir> gpu_bndry_func(CnsFillExtDir{});
        // PhysBCFunct<GpuBndryFuncFab<dummy_gpu_fill_extdir> > physbcf
        //     (geom, bc, gpu_bndry_func);
        // physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), 0.0, 0);
    // }
    // else
    // {
    //     CpuBndryFuncFab cpu_bndry_func(dummy_cpu_fill_extdir);;
    //     PhysBCFunct<CpuBndryFuncFab> physbcf(geom, bc, cpu_bndry_func);
    //     physbcf(phi, 0, phi.nComp(), phi.nGrowVect(), 0.0, 0);
    // }
// }