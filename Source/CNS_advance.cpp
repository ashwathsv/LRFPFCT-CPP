
#include "CNS.H"
#if AMREX_SPACEDIM==2
#include "CNS_hydro_2D_K.H"
#include "LRFPFCT_reaction_2D_K.H"
#include "LRFPFCT_diffusion_2D_K.H"
#else
#include "CNS_hydro_3D_K.H"
#include "LRFPFCT_reaction_3D_K.H"
#include "LRFPFCT_diffusion_3D_K.H"
#endif
#include "CNS_K.H"

using namespace amrex;

Real
CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);

    int numcomp = NUM_STATE;

    MultiFab Sborder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());
    MultiFab Soldtmp(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());

    int conscomp = CCOMP;
    MultiFab dSdt(grids,dmap,conscomp,NUM_GROW,MFInfo(),Factory());

    AMREX_D_TERM(MultiFab flux;, MultiFab fluy;, MultiFab fluz;);
    AMREX_D_TERM(flux.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(0)),
                            S_new.DistributionMap(), conscomp, 0);,
                 fluy.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(1)),
                            S_new.DistributionMap(), conscomp, 0);,
                 fluz.define(amrex::convert(S_new.boxArray(),IntVect::TheDimensionVector(2)),
                            S_new.DistributionMap(), conscomp, 0););
    AMREX_D_TERM(flux.setVal(Real(0.0));, fluy.setVal(0.0);, fluz.setVal(0.0););

    FluxRegister* fr_as_crse = nullptr;
    FluxRegister* fr_as_crse_diff = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
        if(do_diff == 1) fr_as_crse_diff = fine_level.flux_reg_diff.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    FluxRegister* fr_as_fine_diff = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
        if(do_diff == 1) fr_as_fine_diff = flux_reg_diff.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(Real(0.0));
        if(do_diff == 1) fr_as_crse_diff->setVal(Real(0.0));
    }

    // RK2 stage 1
    int rk = 1;
    FillPatch(*this,Sborder,NUM_GROW,time,State_Type,0,numcomp);
    MultiFab::Copy(Soldtmp,Sborder,0,0,numcomp,NUM_GROW);
    MultiFab::Copy(S_new,Sborder,0,0,numcomp,NUM_GROW);

    if(Sborder.contains_nan() || Sborder.min(URHO,NUM_GROW) < Real(0.0) 
        || Sborder.min(UEDEN,NUM_GROW) < Real(0.0) || Sborder.min(UPRE,NUM_GROW) < Real(0.0)){
        Print() << "CNS::advance, after FillPatch(), lev = " << level << ", contains NaN() " << "\n";
        Print() << "min(ueden) = " << Sborder.min(UEDEN,NUM_GROW) << ", min(pre) = " << Sborder.min(UPRE,NUM_GROW) << "\n";
        amrex::Error("NaN value found before RK1 advance, aborting...");
    }

#ifdef LRFPFCT_REACTION
    if(do_react == 1 && Sborder.min(URHOY) < Real(0.0)){
        Print() << "CNS::advance, after FillPatch(), lev = " << level << ", negative mass fraction " << "\n";
        Print() << "min(URHOY) = " << Sborder.min(URHOY,NUM_GROW) << 
        ", min(URHO)= " << Sborder.min(URHO,NUM_GROW) << "\n";
        amrex::Error("Negative mass fraction found before RK1 advance, aborting...");
    }
#endif

    if(do_heun==1){
        // Heun's second order time integration
        // RK step 1, get the RHS and compute Sborder
        compute_dSdt(Soldtmp, Sborder, dSdt, dt, dt, AMREX_D_DECL(flux, fluy, fluz),
                    fr_as_crse, fr_as_fine, 1);  
        // U^* = U^n + dt*(dSdt^n)
        MultiFab::LinComb(S_new, Real(1.0), Sborder, URHO, Real(1.0), dSdt, URHO, URHO, conscomp, 0);
        computeTemp(S_new, 0);
        dSdt.setVal(0.0);

        // RK stage 2
        // After fillpatch, SBorder = U^* = U^n + dt*(dSdt^n)
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, URHO, UPRE+1);
        compute_dSdt(Soldtmp, Sborder, dSdt, Real(0.5)*dt, Real(0.5)*dt, AMREX_D_DECL(flux, fluy, fluz),
                    fr_as_crse, fr_as_fine, 2);
        // U^{n+1} = 0.5*(U^* + U^n)
        MultiFab::LinComb(S_new, Real(0.5), Sborder, URHO, Real(0.5), S_old, URHO, URHO, conscomp, 0);
        // U^{n+1} += 0.5 * dt * (dSdt^*)
        MultiFab::Saxpy(S_new, Real(1.0), dSdt, URHO, URHO, conscomp, 0);
        // Now U^{n+1} = U^n + 0.5*dt*( (dSdt^n) + (dSdt^*) )
    }else{
        // LCPFCT 2nd order time integrator (midpoint method)
        // RK step 1, get the RHS and compute Sborder
        compute_dSdt(Soldtmp, Sborder, dSdt, Real(0.5)*dt, Real(0.5)*dt, AMREX_D_DECL(flux, fluy, fluz),
                    fr_as_crse, fr_as_fine, 1);
        // U^* = U^{n+1/2} = U^n + 0.5*dt*(dSdt^n)
        MultiFab::LinComb(S_new, Real(1.0), Sborder, URHO, Real(1.0), dSdt, URHO, URHO, conscomp, 0); 
        computeTemp(S_new, 0);
        dSdt.setVal(0.0);

        // RK stage 2
        // After fillpatch, SBorder = U^n + 0.5*dt*(dSdt^n)
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, URHO, UPRE+1);
        compute_dSdt(Soldtmp, Sborder, dSdt, dt, dt, AMREX_D_DECL(flux, fluy, fluz),
                    fr_as_crse, fr_as_fine, 2);
        // U^{n+1} = U^n + (dt*(dSdt^*))
        MultiFab::LinComb(S_new, Real(1.0), S_old, URHO, Real(1.0), dSdt, URHO, URHO, conscomp, 0);

        // Now U^{n+1} = U^n + (dt*(dSdt^*))
    }

    computeTemp(S_new, 0);
    FillPatch(*this, S_new, NUM_GROW, time+dt, State_Type, URHO, UPRE+1);

    // add reaction source terms (energy and mass fraction)
#ifdef LRFPFCT_REACTION
    if(do_sootfoil==1 && time > h_parm->start_sfoil_time){
        get_soot_foil(S_old, S_new);
        FillPatch(*this,S_new,NUM_GROW,time+dt,State_Type,SFOIL,1,SFOIL);
    }  
#endif

    if(do_diff == 1){
        FillPatch(*this,Sborder,NUM_GROW,time+dt,State_Type,0,NUM_STATE);
        FillPatch(*this,S_new,NUM_GROW,time+dt,State_Type,0,NUM_STATE);
        add_physical_diffusion(Soldtmp, Sborder, S_new, AMREX_D_DECL(flux, fluy, fluz),
                 dt, fr_as_crse_diff, fr_as_fine_diff, rk);
        computeTemp(S_new,0);
    }
    // FillPatch(*this,S_new,NUM_GROW,time+dt,State_Type,0,NUM_STATE);

    return dt;
}

#ifdef LRFPFCT_REACTION
void
CNS::get_soot_foil(const MultiFab& S_old, MultiFab& S_new)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();

        auto const& snfab  = S_new.array(mfi);
        auto const& sofab  = S_old.array(mfi);

        // ----------------------- Get the numerical soot foil (max pressure trace)---------------------------------------------
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real temp = sofab(i,j,k,SFOIL);
            snfab(i,j,k,SFOIL) = amrex::max(temp, snfab(i,j,k,UPRE));
        });
    }
}
#endif

void
CNS::add_physical_diffusion(const MultiFab& S_old, MultiFab& S, MultiFab& S_new, 
                            AMREX_D_DECL(MultiFab& flux, MultiFab& fluy, MultiFab& fluz),
                            Real dt, FluxRegister* fr_as_crse_diff, FluxRegister* fr_as_fine_diff, int rk)
{
    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    Real dtinv = Real(1.0)/dt;

    EOSParm const* leosparm = d_eos_parm;

    FArrayBox fltmp[BL_SPACEDIM], vel;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();
        auto const& sofab = S_old.array(mfi);
        auto const& sfab  = S.array(mfi);
        auto const& snfab = S_new.array(mfi);

        const int react_do = do_react;

        int conscomp = CCOMP;
        // if(react_do == 1) conscomp = conscomp + 1;

        const int ng = S_new.nGrow();

        Elixir fltmpeli[BL_SPACEDIM], veleli;
        for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            fltmp[dir].resize(amrex::grow(bxtmp,2),conscomp);  
            fltmpeli[dir] = fltmp[dir].elixir(); 
        }

        vel.resize(amrex::grow(bx,ng),BL_SPACEDIM);
        veleli = vel.elixir();

        AMREX_D_TERM(auto const& fxfab = flux.array(mfi);,
                     auto const& fyfab = fluy.array(mfi);,
                     auto const& fzfab = fluz.array(mfi););

        GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL( fltmp[0].array(), 
                                                fltmp[1].array(), fltmp[2].array()) };

        auto const& velarr = vel.array();

        // --------------------- Compute the physical diffusion fluxes ------------------------------

        // Compute the face velocities in the x and y directions
        amrex::ParallelFor(amrex::grow(bx,ng),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   Real rhoinv = Real(1.0) / snfab(i,j,k,URHO);
            for(int nc = 0; nc < BL_SPACEDIM; ++nc){
                velarr(i,j,k,nc) = snfab(i,j,k,UMX+nc) * rhoinv;
            }
        });

        // Compute the diffusion fluxes in x-direction
        const Box& bxnd2 = amrex::grow(amrex::surroundingNodes(bx,0),2);
        amrex::ParallelFor(bxnd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   physical_diffusion_flux_x(i, j, k, fltx[0], sofab, snfab, velarr, 
                                     AMREX_D_DECL(dxinv[0], dxinv[1], dxinv[2]), *leosparm);  });

        // Compute the diffusion fluxes in y-direction
        const Box& bynd2 = amrex::grow(amrex::surroundingNodes(bx,1),2);
        amrex::ParallelFor(bynd2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   physical_diffusion_flux_y(i, j, k, fltx[1], sofab, snfab, velarr, 
                                     AMREX_D_DECL(dxinv[0], dxinv[1], dxinv[2]), *leosparm);  });

        // Update the diffused solution by adding the diffusion fluxes
        amrex::ParallelFor(amrex::grow(bx,2), conscomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   snfab(i,j,k,n) = snfab(i,j,k,n) + dt*dxinv[0]*(fltx[0](i+1,j,k,n) - fltx[0](i,j,k,n))
                          + dt*dxinv[1]*(fltx[1](i,j+1,k,n) - fltx[1](i,j,k,n));
            sfab(i,j,k,n) = snfab(i,j,k,n);     });

        // Store the scaled diffusion fluxes in fluxes MultiFab
        const Box& bxd = amrex::surroundingNodes(bx,0);
        amrex::ParallelFor(bxd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   
#if AMREX_SPACEDIM==2
                fxfab(i,j,k,n) = -fltx[0](i,j,k,n)*dx[1]*dt;
#else
                fxfab(i,j,k,n) = -fltx[0](i,j,k,n)*dx[1]*dx[2]*dt;
#endif
            });

        const Box& byd = amrex::surroundingNodes(bx,1);
        amrex::ParallelFor(byd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   
#if AMREX_SPACEDIM==2
                fyfab(i,j,k,n) = -fltx[1](i,j,k,n)*dx[0]*dt;
#else
                fyfab(i,j,k,n) = -fltx[1](i,j,k,n)*dx[0]*dx[2]*dt;
#endif
            });

#if AMREX_SPACEDIM==3
        const Box& bzd = amrex::surroundingNodes(bx,2);
        amrex::ParallelFor(bzd, conscomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {   fzfab(i,j,k,n) = -fltx[2](i,j,k,n)*dx[0]*dx[1]*dt; });
#endif

    }

    if(rk == 2){
        int conscomp = CCOMP;
        // if(do_react == 1) conscomp = conscomp + 1;

        if (fr_as_crse_diff) {
                AMREX_D_TERM(fr_as_crse_diff->CrseInit(flux, 0, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD);,
                             fr_as_crse_diff->CrseInit(fluy, 1, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD);,
                             fr_as_crse_diff->CrseInit(fluz, 2, 0, 0, conscomp, Real(-1.0), FluxRegister::ADD););
        }

        if (fr_as_fine_diff) {
                AMREX_D_TERM(fr_as_fine_diff->FineAdd(flux, 0, 0, 0, conscomp, Real(1.0));,
                             fr_as_fine_diff->FineAdd(fluy, 1, 0, 0, conscomp, Real(1.0));,
                             fr_as_fine_diff->FineAdd(fluz, 2, 0, 0, conscomp, Real(1.0)););
        }        
    } 

}
