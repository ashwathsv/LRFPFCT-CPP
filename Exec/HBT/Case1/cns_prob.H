#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

// #ifdef AMREX_USE_GPU
AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Box& domain = geomdata.Domain();
    const auto domlo = amrex::lbound(domain);

    Real ycm = Real(0.5)*(geomdata.ProbLo(1) + geomdata.ProbHi(1));
    Real tubelo = ycm - Real(0.5)*prob_parm.width;
    Real tubehi = tubelo + prob_parm.width;

    Real y = geomdata.ProbLo(1) + (j+Real(0.5))*geomdata.CellSize(1);

    state(i,j,k,URHO ) = prob_parm.rho_0;
    state(i,j,k,UMX  ) = prob_parm.rho_0*prob_parm.u_0;
    state(i,j,k,UMY  ) = prob_parm.rho_0*prob_parm.v_0;
    Real et = prob_parm.p_0/(parm.eos_gamma-Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UPRE)  = prob_parm.p_0;
    state(i,j,k,UEDEN) = et + Real(0.5)*prob_parm.rho_0*( prob_parm.u_0*prob_parm.u_0 
                       + prob_parm.v_0*prob_parm.v_0 );
    state(i,j,k,UTEMP) = (Real(1.0)/state(i,j,k,URHO)) * state(i,j,k,UEINT) * (Real(1.0)/parm.cv);        
    
    
    
}

AMREX_GPU_DEVICE
inline
void
cns_probspecific_func (amrex::MultiFab& State, amrex::GeometryData const& geomdata, ProbParm& prob_parm,
                        int stepnum, amrex::Real time)
{
    using amrex::Real;

    const Box& domain = geomdata.Domain();
    const auto domhi = amrex::ubound(domain);
    const auto domlo = amrex::lbound(domain);
    const int Ny = domhi.y - domlo.y + 1;

    Real minpre = Real(1.e37);
    Real maxpre = Real(0.0);
    Real meanpre = Real(0.0);

    int myproc = ParallelDescriptor::MyProc();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(State,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& sfab = State.array(mfi);
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);

        if(hi.x == domhi.x){
            for(int k = lo.z; k <= hi.z; ++k){
                for(int j = lo.y; j <= hi.y; ++j){
                    meanpre += sfab(domhi.x,j,k,UPRE);
                    maxpre  = amrex::max(maxpre,sfab(domhi.x,j,k,UPRE));
                    minpre  = amrex::min(minpre,sfab(domhi.x,j,k,UPRE)); 
                }
            }            
        }
    }

    ParallelDescriptor::ReduceRealMin(minpre);
    ParallelDescriptor::ReduceRealMax(maxpre);
    ParallelDescriptor::ReduceRealSum(meanpre);
    meanpre = meanpre/Ny;

    ParallelDescriptor::Barrier();

    if(myproc == 0){
        // Print(myproc) << "myproc = " << myproc << ", meanpre= " << meanpre << 
        // ", minpre = " << minpre << ", maxpre = " << maxpre << "presfile = " << prob_parm.pres_file << "\n";
        
        std::ofstream ofs;

        if(stepnum == 0){
            ofs.open(prob_parm.pres_file, std::ofstream::out);
            if(!ofs)  amrex::Abort("Failed to create file to write pressure data..exiting!");
            Print(myproc,ofs) << "# time minp maxp meanp" << "\n";
        }else{
            ofs.open(prob_parm.pres_file, std::ofstream::app);
        }

        Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
        << std::left << std::setw(12) << minpre  << "\t"  
        << std::left << std::setw(12) << maxpre  << "\t"
        << std::left << std::setw(12) << meanpre  << "\n";

        ofs.close();    
    }

    ParallelDescriptor::Barrier();

}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_probspecific_bc(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NGROW][NCONS],
  amrex::Real s_ext[NCONS],
  const int /*idir*/,
  const int ind,
  const int /*sgn*/,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm)
{
    using amrex::Real;

    Real ycm = Real(0.5)*(geomdata.ProbLo(1) + geomdata.ProbHi(1));
    Real tubelo = ycm - Real(0.5)*prob_parm.width;
    Real tubehi = tubelo + prob_parm.width;
    if(x[1] >= tubelo && x[1] <= tubehi){
        if(time <= prob_parm.inflow_time){
            // high-pressure supersonic inflow BC
             s_ext[URHO]  = prob_parm.rho_r * prob_parm.rho_0;
             s_ext[UPRE]  = prob_parm.p_r * prob_parm.p_0;
             s_ext[UMY]   = Real(0.0);
             s_ext[UEINT] = prob_parm.p_r * prob_parm.p_0 / (prob_parm.gamma - Real(1.0));
             Real cs      = std::sqrt(prob_parm.gamma * prob_parm.p_r * prob_parm.p_0 
                            / (prob_parm.rho_r * prob_parm.rho_0));
             s_ext[UMX]   = prob_parm.rho_r * prob_parm.rho_0 * prob_parm.Mach_in * cs;
             s_ext[UEDEN] = s_ext[UEINT] 
                          + Real(0.5)*( s_ext[UMX]*s_ext[UMX] + s_ext[UMY]*s_ext[UMY])/s_ext[URHO];
             s_ext[UTEMP]  = s_ext[UPRE]/(prob_parm.Rsp * s_ext[URHO]);
        } else{
            // simple first-order extrapolation
            for(int nc = URHO; nc < NCONS; ++nc){
                s_ext[nc] = s_int[0][nc];
            }
        }
    }else{
        // non-slip wall boundary condition
        s_ext[URHO]   = s_int[-(ind+1)][URHO];
        s_ext[UMX]    = -s_int[-(ind+1)][UMX];
        s_ext[UMY]    = s_int[-(ind+1)][UMY];
        s_ext[UEDEN]  = s_int[-(ind+1)][UEDEN];
        s_ext[UEINT]  = s_int[-(ind+1)][UEINT];
        s_ext[UTEMP]  = s_int[-(ind+1)][UTEMP];
        s_ext[UPRE]   = s_int[-(ind+1)][UPRE];
    }
}
#endif
