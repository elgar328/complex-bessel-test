C     Compute Bessel/Airy values on a grid of (function, nu, z) points.
C     Input (stdin): one line per point:
C       function_id  nu  z_re  z_im  kode
C     function_id: 1=J, 2=Y, 3=I, 4=K, 5=H1, 6=H2, 7=Ai, 8=Ai', 9=Bi, 10=Bi'
C     kode: 1=Unscaled, 2=Scaled
C
C     Output (stdout): one line per point:
C       function_id  nu  z_re  z_im  kode  result_re  result_im  ierr
C
      PROGRAM COMPUTE_GRID
      IMPLICIT NONE

      INTEGER FID, KODE, IERR, NZ, M
      DOUBLE PRECISION FNU, ZR, ZI, CYR(1), CYI(1)
      DOUBLE PRECISION CWRKR(1), CWRKI(1)
      DOUBLE PRECISION AIR, AII
      INTEGER IOSTAT
      INTEGER I1MACH
      DOUBLE PRECISION D1MACH

C     Read lines until EOF
 10   CONTINUE
      READ(*, *, IOSTAT=IOSTAT) FID, FNU, ZR, ZI, KODE
      IF (IOSTAT .NE. 0) GOTO 999

      IERR = 0
      NZ = 0
      CYR(1) = 0.0D0
      CYI(1) = 0.0D0
      AIR = 0.0D0
      AII = 0.0D0

      IF (FID .EQ. 1) THEN
C       J function: ZBESJ
        CALL ZBESJ(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)

      ELSE IF (FID .EQ. 2) THEN
C       Y function: ZBESY (separate workspace arrays)
        CALL ZBESY(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ,
     &             CWRKR, CWRKI, IERR)

      ELSE IF (FID .EQ. 3) THEN
C       I function: ZBESI
        CALL ZBESI(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)

      ELSE IF (FID .EQ. 4) THEN
C       K function: ZBESK
        CALL ZBESK(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)

      ELSE IF (FID .EQ. 5) THEN
C       H^(1): ZBESH with M=1
        M = 1
        CALL ZBESH(ZR, ZI, FNU, KODE, M, 1, CYR, CYI, NZ, IERR)

      ELSE IF (FID .EQ. 6) THEN
C       H^(2): ZBESH with M=2
        M = 2
        CALL ZBESH(ZR, ZI, FNU, KODE, M, 1, CYR, CYI, NZ, IERR)

      ELSE IF (FID .EQ. 7) THEN
C       Ai: ZAIRY with ID=0
        CALL ZAIRY(ZR, ZI, 0, KODE, AIR, AII, NZ, IERR)
        CYR(1) = AIR
        CYI(1) = AII

      ELSE IF (FID .EQ. 8) THEN
C       Ai': ZAIRY with ID=1
        CALL ZAIRY(ZR, ZI, 1, KODE, AIR, AII, NZ, IERR)
        CYR(1) = AIR
        CYI(1) = AII

      ELSE IF (FID .EQ. 9) THEN
C       Bi: ZBIRY with ID=0
        CALL ZBIRY(ZR, ZI, 0, KODE, AIR, AII, IERR)
        CYR(1) = AIR
        CYI(1) = AII

      ELSE IF (FID .EQ. 10) THEN
C       Bi': ZBIRY with ID=1
        CALL ZBIRY(ZR, ZI, 1, KODE, AIR, AII, IERR)
        CYR(1) = AIR
        CYI(1) = AII

      ELSE
        IERR = -1
      END IF

      WRITE(*, '(I3, 1X, E23.16, 1X, E23.16, 1X, E23.16, 1X,
     &  I2, 1X, E23.16, 1X, E23.16, 1X, I2)')
     &  FID, FNU, ZR, ZI, KODE, CYR(1), CYI(1), IERR

      GOTO 10

 999  CONTINUE
      END
