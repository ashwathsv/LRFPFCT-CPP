#ifndef CNS_INDEX_H_
#define CNS_INDEX_H_

#define URHO  0
#define UMX   1
#define UMY   2

#if AMREX_SPACEDIM==2
#define UEDEN 3

#ifdef LRFPFCT_REACTION
#define URHOY 4
#define UEINT 5
#define UTEMP 6
#define UPRE  7
#define SFOIL 8
#define NCONS 9
#define CCOMP 5
#else
#define UEINT 4
#define UTEMP 5
#define UPRE  6
#define NCONS 7
#define CCOMP 4
#endif

#elif AMREX_SPACEDIM==3
#define UMZ   3
#define UEDEN 4

#ifdef LRFPFCT_REACTION
#define URHOY 5
#define UEINT 6
#define UTEMP 7
#define UPRE  8
#define SFOIL 9
#define NCONS 10
#define CCOMP 6
#else
#define UEINT 5
#define UTEMP 6
#define UPRE  7
#define NCONS 8
#define CCOMP 5
#endif
#endif

#define QRHO   0
#define QU     1
#define QV     2
#define QW     3
#define QPRES  4
#define QCS    5
#define QEINT  6
#define QTEMP  7
#define NPRIM  8
#define NGROW  4

#endif
