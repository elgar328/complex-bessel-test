C     Edge-test driver, double precision (TOMS 644 ZBSUBS).
C     Input (stdin), one case per line:
C       FN KODE N FNU ZR ZI
C     FN: J, Y, I, K, H1, H2 (N >= 1) or AI, AP, BI, BP (Airy, N = 1)
C     Output (stdout), one line per case:
C       IERR NZ RE(1) IM(1) ... RE(N) IM(N)
C     ZBIRY has no NZ argument; NZ is reported as 0.
      PROGRAM EDGE64
      IMPLICIT NONE
      INTEGER NMAX
      PARAMETER (NMAX=256)
      CHARACTER*2 FN
      INTEGER KODE, N, NZ, IERR, I, IOS
      DOUBLE PRECISION FNU, ZR, ZI, AR, AI
      DOUBLE PRECISION CYR(NMAX), CYI(NMAX), CWR(NMAX), CWI(NMAX)

   10 CONTINUE
      READ(*, *, IOSTAT=IOS) FN, KODE, N, FNU, ZR, ZI
      IF (IOS .NE. 0) GO TO 99
      IF (N .LT. 1 .OR. N .GT. NMAX) STOP 'EDGE64: N OUT OF RANGE'
      DO 20 I = 1, N
        CYR(I) = 0.0D0
        CYI(I) = 0.0D0
   20 CONTINUE
      NZ = 0
      IERR = 0

      IF (FN .EQ. 'J ') THEN
        CALL ZBESJ(ZR, ZI, FNU, KODE, N, CYR, CYI, NZ, IERR)
      ELSE IF (FN .EQ. 'Y ') THEN
        CALL ZBESY(ZR, ZI, FNU, KODE, N, CYR, CYI, NZ, CWR, CWI,
     &             IERR)
      ELSE IF (FN .EQ. 'I ') THEN
        CALL ZBESI(ZR, ZI, FNU, KODE, N, CYR, CYI, NZ, IERR)
      ELSE IF (FN .EQ. 'K ') THEN
        CALL ZBESK(ZR, ZI, FNU, KODE, N, CYR, CYI, NZ, IERR)
      ELSE IF (FN .EQ. 'H1') THEN
        CALL ZBESH(ZR, ZI, FNU, KODE, 1, N, CYR, CYI, NZ, IERR)
      ELSE IF (FN .EQ. 'H2') THEN
        CALL ZBESH(ZR, ZI, FNU, KODE, 2, N, CYR, CYI, NZ, IERR)
      ELSE IF (FN .EQ. 'AI' .OR. FN .EQ. 'AP') THEN
        AR = 0.0D0
        AI = 0.0D0
        IF (FN .EQ. 'AI') CALL ZAIRY(ZR, ZI, 0, KODE, AR, AI, NZ, IERR)
        IF (FN .EQ. 'AP') CALL ZAIRY(ZR, ZI, 1, KODE, AR, AI, NZ, IERR)
        CYR(1) = AR
        CYI(1) = AI
      ELSE IF (FN .EQ. 'BI' .OR. FN .EQ. 'BP') THEN
        AR = 0.0D0
        AI = 0.0D0
        IF (FN .EQ. 'BI') CALL ZBIRY(ZR, ZI, 0, KODE, AR, AI, IERR)
        IF (FN .EQ. 'BP') CALL ZBIRY(ZR, ZI, 1, KODE, AR, AI, IERR)
        CYR(1) = AR
        CYI(1) = AI
      ELSE
        STOP 'EDGE64: UNKNOWN FUNCTION'
      END IF

C     ES26.17E3: 17 significant digits round-trip a double; the forced
C     3-digit exponent keeps the 'E' for exponents beyond +-99.
      WRITE(*, '(I2, 1X, I4, 512(1X, ES26.17E3))')
     &  IERR, NZ, (CYR(I), CYI(I), I = 1, N)
      GO TO 10

   99 CONTINUE
      END
