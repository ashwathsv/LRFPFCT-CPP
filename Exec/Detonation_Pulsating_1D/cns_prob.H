#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "LRFPFCT_EOS_parm.H"
#include "cns_prob_parm.H"

// #ifdef AMREX_USE_GPU
AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, 
              Parm const& parm, EOSParm const& eos_parm,  ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];

    Real Pt, rhot, uxt, uyt, Yt, Temp;
    if(x < prob_lo[0] + prob_parm.width){
        Pt = prob_parm.p1;
        rhot = Pt / (eos_parm.Rsp * prob_parm.T1);
        uxt = prob_parm.u1;
        uyt = Real(0.0);
        Yt  = prob_parm.Y1;
        Temp  = prob_parm.T1;  
    }else{
        Pt = prob_parm.p0;
        rhot = Pt / (eos_parm.Rsp * prob_parm.T0);
        uxt = prob_parm.u0;
        uyt = Real(0.0);       
        Yt = prob_parm.Y0;
        Temp  = prob_parm.T0;
    }
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot*uxt;
    state(i,j,k,UMY  ) = rhot*uyt;
    
    Real et = Pt/(eos_parm.eos_gamma-Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UPRE)  = Pt;

    state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt + uyt*uyt);
#ifdef LRFPFCT_REACTION
    state(i,j,k,URHOY) = rhot*Yt;
    state(i,j,k,SFOIL) = Real(0.0);
#endif
    state(i,j,k,UTEMP) = Temp;
    
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_probspecific_bc(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NGROW][NCONS],
  amrex::Real s_ext[NCONS],
  const int /*idir*/,
  const int /*ind*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParm const& /*prob_parm*/)
{

}

AMREX_GPU_HOST
inline
void
cns_probspecific_func (amrex::MultiFab& State, amrex::GeometryData const& geomdata, ProbParm& prob_parm,
                        int stepnum, amrex::Real time, amrex::Real dt)
{
    using amrex::Real;

   if(prob_parm.print_pres_data == 1 && stepnum % prob_parm.pres_int == 0){

        const Box& domain = geomdata.Domain();
        const auto domlo = amrex::lbound(domain);
        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx      = geomdata.CellSize();

        Real maxpre     = amrex::Real(0.0);
        Real maxpgrad   = 0.0;
        Real maxpglocal = 0.0;
        Real maxplocal  = amrex::Real(0.0);
        Real maxploc    = amrex::Real(-1e10);

        int nproc = ParallelDescriptor::NProcs();

        int minp_rank = nproc + 10;

        int myproc = ParallelDescriptor::MyProc();

        // print out pressure data at top wall 

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(State,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto const& sfab = State.array(mfi);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);

            if(lo.y == domlo.y){
                // Here the shock location is the location of max pressure
                if(prob_parm.get_maxgrad==0){
                    for(int i = lo.x; i <= hi.x; ++i){
                        if(sfab(i,lo.y,domlo.z,UPRE) > maxpre){
                            maxpre = sfab(i,lo.y,domlo.z,UPRE);
                            maxploc = prob_lo[0] + (i+Real(0.5))*dx[0];
                        }
                    }                    
                }else{
                    // Here, the shock location is the location of maximum pressure gradient
                    for(int i = lo.x; i <= hi.x; ++i){
                        if(sfab(i,lo.y,domlo.z,UPRE) > maxpre){
                            maxpre = sfab(i,lo.y,domlo.z,UPRE);
                        }

                        Real grad = std::fabs(sfab(i+1,lo.y,domlo.z,UPRE) - sfab(i-1,lo.y,domlo.z,UPRE));
                        if(grad > maxpgrad){
                            maxpgrad = grad;
                            maxploc = prob_lo[0] + (i+Real(0.5))*dx[0];
                        }
                    }
                }        
            }
            if(prob_parm.get_maxgrad==1){
              maxplocal  = maxpre; maxpglocal = maxpgrad; 
            } else{   maxplocal = maxpre;    }
        }

        ParallelDescriptor::ReduceRealMax(maxpre);
        if(prob_parm.get_maxgrad==1) ParallelDescriptor::ReduceRealMax(maxpgrad);

        if(prob_parm.get_maxgrad==0){
            if(std::fabs(maxplocal-maxpre) < 1e-6 * maxpre){
                minp_rank = myproc;
            }
        }else{
            if(std::fabs(maxpglocal-maxpgrad) < 1e-6 * maxpgrad){
                minp_rank = myproc;
            }
        }
        
        ParallelDescriptor::ReduceIntMin(minp_rank);

        if(minp_rank > nproc){
            amrex::Abort("Could not find rank of minimum pressure....EXITING \n");
        }else{
            if(myproc == minp_rank){
                std::ofstream ofs;
                if(prob_parm.print_append == 0){
                    prob_parm.print_append = 1;
                    ofs.open(prob_parm.pres_file, std::ofstream::out);
                    if(!ofs)  amrex::Abort("Failed to create file to write pressure data..exiting!");
                    Print(myproc,ofs) << "# time maxp location speed(dx_dt)" << "\n";
                }else{
                    ofs.open(prob_parm.pres_file, std::ofstream::app);
                }

                Real sh_sp = 0.0;
                if(time > Real(1E-30)) sh_sp = (maxploc - prob_parm.old_shock_loc)/dt;

                if(prob_parm.get_maxgrad==0){
                   Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                    << std::left << std::setw(12) << maxpre  << "\t"  
                    << std::left << std::setw(12) << maxploc << "\t"
                    << std::left << std::setw(12) << sh_sp   <<"\n"; 
                }else{
                    Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                    << std::left << std::setw(12) << maxpre  << "\t"  
                    << std::left << std::setw(12) << maxploc << "\t"
                    << std::left << std::setw(12) << sh_sp   << "\n";
                }
                

                ofs.close();
                // Print() << "old_shock_loc = " << prob_parm.old_shock_loc << "\n";
                prob_parm.old_shock_loc = maxploc; 
                // Print() << "maxploc = " << maxploc << "\n";
            }
        }
        ParallelDescriptor::Barrier();
    }
}

#endif
