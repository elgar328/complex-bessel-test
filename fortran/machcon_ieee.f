C     IEEE 754 double precision machine constants for TOMS 644
C     Replaces I1MACH and D1MACH with hardcoded IEEE values.
C
      INTEGER FUNCTION I1MACH(I)
      INTEGER I
      INTEGER IMACH(16)
      DATA IMACH( 1) /     5 /
      DATA IMACH( 2) /     6 /
      DATA IMACH( 3) /     6 /
      DATA IMACH( 4) /     6 /
      DATA IMACH( 5) /    32 /
      DATA IMACH( 6) /     4 /
      DATA IMACH( 7) /     2 /
      DATA IMACH( 8) /    31 /
      DATA IMACH( 9) / 2147483647 /
      DATA IMACH(10) /     2 /
      DATA IMACH(11) /    24 /
      DATA IMACH(12) /  -125 /
      DATA IMACH(13) /   128 /
      DATA IMACH(14) /    53 /
      DATA IMACH(15) / -1021 /
      DATA IMACH(16) /  1024 /
      I1MACH = IMACH(I)
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION D1MACH(I)
      INTEGER I
      DOUBLE PRECISION DMACH(5)
C     DMACH(1) = smallest normalized: 2^(-1022)
      DATA DMACH(1) / 2.2250738585072014D-308 /
C     DMACH(2) = largest: (1-2^(-53))*2^1024
      DATA DMACH(2) / 1.7976931348623157D+308 /
C     DMACH(3) = 2^(-53) (half ulp)
      DATA DMACH(3) / 1.1102230246251565D-016 /
C     DMACH(4) = 2^(-52) (unit roundoff = epsilon)
      DATA DMACH(4) / 2.2204460492503131D-016 /
C     DMACH(5) = log10(2)
      DATA DMACH(5) / 3.0102999566398120D-001 /
      D1MACH = DMACH(I)
      RETURN
      END
C
C     Note: XERROR stub is in zbsubs.f, not needed here.
C
