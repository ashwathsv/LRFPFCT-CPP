
#include <CNS.H>
#include <CNS_K.H>
#include <CNS_tagging.H>
#include <CNS_parm.H>
#include <LRFPFCT_EOS_parm.H>

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

#include <climits>

using namespace amrex;

constexpr int CNS::NUM_GROW;

BCRec     CNS::phys_bc;

int       CNS::verbose                  = 0;
IntVect   CNS::hydro_tile_size {AMREX_D_DECL(1024,16,16)};
Real      CNS::cfl                      = 0.3;
int       CNS::do_reflux                = 1;
int       CNS::refine_max_dengrad_lev   = -1;
Real      CNS::refine_gradlim           = 1.0e10;
int       CNS::refine_based_maxgrad     = -1; 
int       CNS::grad_comp                = -1;
Real      CNS::tagfrac                  = 0.0;

Real      CNS::gravity                  = 0.0;
Real      CNS::diff1                    = 1.0;
int       CNS::do_react                 = 0;

CNS::CNS ()
{}

CNS::CNS (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NUM_STATE));
    }

    buildMetrics();
}

CNS::~CNS ()
{}

void
CNS::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<CNS&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);
}

void
CNS::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);
}

void
CNS::initData ()
{
    BL_PROFILE("CNS::initData()");

    const auto geomdata = geom.data();
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();
    MultiFab& S_new = get_new_data(State_Type);

    Parm const* lparm = d_parm;
    EOSParm const* leosparm = d_eos_parm;
    ProbParm const* lprobparm = d_prob_parm;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        auto sfab = S_new.array(mfi);

        amrex::ParallelFor(amrex::grow(box,NUM_GROW),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_initdata(i, j, k, sfab, geomdata, *lparm, *leosparm, *lprobparm);
        });
        
    }

    if(S_new.contains_nan()){
        Print() << "End of CNS::initData(), lev = " << level << ", contains NaN() " << "\n";
        amrex::Error("NaN value found in initial conditions, aborting...");
    }

    FillPatch(*this,S_new,NUM_GROW,0.0,State_Type,0,NUM_STATE);

    if(level == 0){
        ProbParm* lpparm = d_prob_parm;
        cns_probspecific_func(S_new, geomdata, *lpparm, 0, 0.0);
        if(h_parm->do_minp == 1){
            h_parm->minro = h_parm->minrofrac*S_new.min(URHO);
            h_parm->minp  = h_parm->minpfrac*S_new.min(UPRE);
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_parm, h_parm+1, d_parm);
    }
    Print() << "max(pre) = " << S_new.max(UPRE,NUM_GROW) << ", max(rho) = " << S_new.max(URHO,NUM_GROW) << "\n";

}

void
CNS::computeInitialDt (int                    finest_level,
                       int                    /*sub_cycle*/,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& /*ref_ratio*/,
                       Vector<Real>&          dt_level,
                       Real                   stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0) {
        return;
    }

    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
CNS::computeNewDt (int                    finest_level,
                   int                    /*sub_cycle*/,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& /*ref_ratio*/,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep(0);
    }

    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        static Real change_max = 1.1;
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
        }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
CNS::post_regrid (int /*lbase*/, int /*new_finest*/)
{
}

void
CNS::post_timestep (int iteration)
{
    BL_PROFILE("post_timestep");

    if (do_reflux && level < parent->finestLevel()) {
        MultiFab& S = get_new_data(State_Type);
        MultiFab  volfab(S.boxArray(), S.DistributionMap(), 1, 0);
        const Real* dx = geom.CellSize();

        CNS& fine_level = getLevel(level+1);
        int conscomp = CCOMP;
        if(do_react == 1) conscomp = conscomp + 1;

        fine_level.flux_reg->Reflux(S, Real(1.0), 0, 0, conscomp, geom);
        computeTemp(S,0);

        if(S.min(UEDEN,0) < 0.0 || S.min(URHO,0) < 0.0 || S.min(UPRE,0) < 0.0){
            Print() << "energy/pressure/density negative after reflux, lev = " << level << 
            ", min(ro, roE, pre) = " << S.min(URHO,S.nGrow()) << ", " << S.min(UEDEN,S.nGrow()) 
            << ", " << S.min(UPRE,S.nGrow()) << "\n";
            amrex::Error("negative value found in MultiFab, aborting from CNS::post_timestep()...");
        }
    }

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

void
CNS::postCoarseTimeStep (Real time)
{
    BL_PROFILE("postCoarseTimeStep()");

    if(level == 0){
        MultiFab& S = get_new_data(State_Type);
        ProbParm* lpparm = d_prob_parm;
        const auto geomdata = geom.data();
        cns_probspecific_func(S, geomdata, *lpparm, 1, time);
    }

    // This only computes sum on level 0
    if (verbose >= 2) {
        printTotal();
    }
}

void
CNS::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);

    std::array<Real,CCOMP+1> tot;
    for (int comp = URHO; comp <= URHOY; ++comp) {
        tot[comp] = S_new.sum(comp,true) * geom.ProbSize();
    }
    Real romin = S_new.min(URHO,0);
    Real pmin  = S_new.min(UPRE,0);
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            ParallelDescriptor::ReduceRealSum(tot.data(), CCOMP+1, ParallelDescriptor::IOProcessorNumber());
            ParallelDescriptor::ReduceRealMin(romin, ParallelDescriptor::IOProcessorNumber());
            ParallelDescriptor::ReduceRealMin(pmin, ParallelDescriptor::IOProcessorNumber());
            amrex::Print().SetPrecision(17) << "\n[CNS] Total mass          is " << tot[URHO] << "\n"
                                            <<   "      Total x-momentum    is " << tot[UMX] << "\n"
                                            <<   "      Total y-momentum    is " << tot[UMY] << "\n"
#if AMREX_SPACEDIM==3
                                            <<   "      Total z-momentum    is " << tot[UMZ] << "\n"
#endif
                                            <<   "      Total energy        is " << tot[UEDEN] << "\n"
                                            <<   "      Total mass fraction is " << tot[URHOY] << "\n"
                                            <<   "      Minimum density     is  " << romin << "\n"
                                            <<   "      Minimum pressure    is  "<< pmin << "\n";
#ifdef BL_LAZY
        });
#endif
}

void
CNS::post_init (Real)
{
    if (level > 0) return;
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    if (verbose >= 2) {
        printTotal();
    }
}

void
CNS::post_restart ()
{
}

void
CNS::errorEst (TagBoxArray& tags, int, int, Real time, int, int)
{
    BL_PROFILE("CNS::errorEst()");

    const auto geomdata = geom.data();

    if (level < refine_max_dengrad_lev)
    {
        MultiFab& S_new = get_new_data(State_Type);
        // const Real cur_time = state[State_Type].curTime();

        const char   tagval = TagBox::SET;
//        const char clearval = TagBox::CLEAR;

        if(refine_based_maxgrad){
            // tagging criterion based on maximum gradient of the chosen component
#if AMREX_SPACEDIM==2
            int numgradcomp = 3;
#else
            int numgradcomp = 4;
#endif

            MultiFab grad(S_new.boxArray(), S_new.DistributionMap(), numgradcomp, 0);
            int which_comp = grad_comp;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                const auto Sfab = S_new.array(mfi);
                auto gradfab = grad.array(mfi); 

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cns_get_grad(i, j, k, Sfab, gradfab, geomdata, which_comp);
                });
            }

            Real maxgrad =  grad.norm0(numgradcomp-1);

            refine_gradlim = maxgrad;

            const Real grad_threshold = refine_gradlim;
            const Real frac_tag = tagfrac;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                auto gradfab = grad.array(mfi); 
                auto tag = tags.array(mfi);

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cns_tag_maxgrad(i, j, k, tag, gradfab, grad_threshold, tagval, frac_tag);
                });
            }

        }else{
            // tagging criteria based on the difference in the given component being greater than a threshold value 
            const Real threshold_val = refine_gradlim;
            int which_comp = grad_comp;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                const auto Sfab = S_new.array(mfi);
                auto tag = tags.array(mfi);

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cns_tag_graderror(i, j, k, tag, Sfab, geomdata, threshold_val, tagval, which_comp);
                });
            }            
        }
    }
}

void
CNS::read_params ()
{
    ParmParse pp("cns");

    pp.query("v", verbose);

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }

    pp.query("cfl", cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);
    pp.query("do_react", do_react);

    pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
    pp.query("refine_gradlim", refine_gradlim);
    pp.query("refine_based_maxgrad", refine_based_maxgrad);
    pp.query("diff1", diff1);
    if(refine_based_maxgrad){
        pp.get("grad_comp", grad_comp);
        pp.get("tagfrac", tagfrac);
    }

    pp.query("gravity", gravity);

    // pp.query("eos_gamma", h_parm->eos_gamma);
    // do_minp == 1 implies we set a minimum pressure and do not allow pressure/density to go negative 
    // allows for using high cfl number but may reduce accuracy / result in few points with non-physical pressures
    pp.query("do_minp", h_parm->do_minp);
    if(h_parm->do_minp == 1) {
        // read in the fractions for minimum pressure and density (these fractions are related to the minimum 
        // pressure in the initial conditions at the coarsest level)
        pp.query("minpfrac", h_parm->minpfrac);
        pp.query("minrofrac", h_parm->minrofrac);
    }

    h_parm->Initialize();
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_parm, h_parm+1, d_parm);

    read_eos_params(do_react);
}

void
CNS::read_eos_params (int do_react)
{
    ParmParse pp("eos");

    // read in gamma
    pp.query("gamma", h_eos_parm->eos_gamma);
    // read in molecular weight (in CGS)
    pp.query("molecular_weight", h_eos_parm->eos_mu);
    h_eos_parm->eos_mu = h_eos_parm->eos_mu*1e-3;

    // read in heat release q (non-dimensional, as q * Mw / (R * T_0) )
    if(do_react == 1){
        pp.query("heat_release", h_eos_parm->q_nd);
        pp.query("activation_energy", h_eos_parm->Ea_nd);
        pp.query("reference_temperature", h_eos_parm->Tref);
        pp.query("pre_exponential", h_eos_parm->pre_exp);
        pp.query("kappa_0", h_eos_parm->kappa_0);

        // Convert Arrhenius pre-exponential and thermal conductivity to SI units
        h_eos_parm->pre_exp = h_eos_parm->pre_exp * 1e-3;
        h_eos_parm->kappa_0 = h_eos_parm->kappa_0 * 1e-1;
    }

    h_eos_parm->Initialize();

    // Calculate the dimensional heat release and activation energy (in SI units)
    h_eos_parm->q_dim = h_eos_parm->q_nd * h_eos_parm->Rsp * h_eos_parm->Tref;
    h_eos_parm->Ea_dim = h_eos_parm->Ea_nd * h_eos_parm->Ru * h_eos_parm->Tref;

    Print() << "gamma = " << h_eos_parm->eos_gamma << ", mu = " << h_eos_parm->eos_mu
            << ", q = " << h_eos_parm->q_nd << ", " << h_eos_parm->q_dim
            << ", Ea = " << h_eos_parm->Ea_nd << ", " << h_eos_parm->Ea_dim
            << ", A = " << h_eos_parm->pre_exp << "\n";
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_eos_parm, h_eos_parm+1, d_eos_parm);
}

void
CNS::avgDown ()
{
    BL_PROFILE("CNS::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    int conscomp = CCOMP;
    if(do_react == 1) conscomp = conscomp + 1;

    amrex::average_down(S_fine, S_crse, fine_lev.geom, geom,
                        0, conscomp, parent->refRatio(level));

    const int nghost = 0;

    computeTemp(S_crse, nghost);
}   

void
CNS::buildMetrics ()
{
    // make sure dx == dy == dz
    // const Real* dx = geom.CellSize();
    // if (std::abs(dx[0]-dx[1]) > Real(1.e-12)*dx[0] || std::abs(dx[0]-dx[2]) > Real(1.e-12)*dx[0]) {
    //     amrex::Abort("CNS: must have dx == dy == dz\n");
    // }
}

Real
CNS::estTimeStep (int ng)
{
    BL_PROFILE("CNS::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    const int ngrow = ng;

    Parm const* lparm = d_parm;
    EOSParm const* leosparm = d_eos_parm;
#ifdef AMREX_USE_GPU
    prefetchToDevice(S);
#endif
    
    Real estdt = amrex::ReduceMin(S, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real
    {
        return cns_estdt(bx, fab, dx, ngrow, *lparm, *leosparm);
    });

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    return estdt;
}

Real
CNS::initialTimeStep ()
{
    return estTimeStep(NUM_GROW);
}

void
CNS::computeTemp (MultiFab& State, int ng)
{
    BL_PROFILE("CNS::computeTemp()");

    Parm const* lparm = d_parm;
    EOSParm const* leosparm = d_eos_parm;
    // This will reset Eint and compute Temperature
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(State,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);
        auto const& sfab = State.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_compute_temperature(i,j,k,sfab,*lparm,*leosparm);
        });
    }
}

