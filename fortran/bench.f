C     Benchmark Fortran TOMS 644 Bessel/Airy functions.
C     Same input format as compute_grid.f.
C     Output: function_id  nu  z_re  z_im  time_ns
C
      PROGRAM BENCH_FORTRAN
      IMPLICIT NONE

      INTEGER FID, KODE, IERR, NZ, M
      DOUBLE PRECISION FNU, ZR, ZI, CYR(1), CYI(1)
      DOUBLE PRECISION AIR, AII
      INTEGER IOSTAT
      INTEGER I1MACH
      DOUBLE PRECISION D1MACH

      INTEGER(8) COUNT1, COUNT2, COUNT_RATE
      DOUBLE PRECISION ELAPSED_NS
      INTEGER IREP, NREP

C     Get clock rate
      CALL SYSTEM_CLOCK(COUNT_RATE=COUNT_RATE)

      NREP = 1000

C     Warmup: 100 dummy calls
      DO IREP = 1, 100
        CALL ZBESJ(1.0D0, 2.0D0, 0.5D0, 1, 1, CYR, CYI, NZ, IERR)
        CALL ZBESK(1.0D0, 2.0D0, 0.5D0, 1, 1, CYR, CYI, NZ, IERR)
      END DO

 10   CONTINUE
      READ(*, *, IOSTAT=IOSTAT) FID, FNU, ZR, ZI, KODE
      IF (IOSTAT .NE. 0) GOTO 999

C     Warmup this specific point
      CALL DO_CALL(FID, FNU, ZR, ZI, KODE, CYR, CYI, AIR, AII,
     &             NZ, IERR, M)

C     Time the call
      CALL SYSTEM_CLOCK(COUNT1)
      DO IREP = 1, NREP
        CALL DO_CALL(FID, FNU, ZR, ZI, KODE, CYR, CYI, AIR, AII,
     &               NZ, IERR, M)
      END DO
      CALL SYSTEM_CLOCK(COUNT2)

      ELAPSED_NS = DBLE(COUNT2 - COUNT1) / DBLE(COUNT_RATE)
     &             * 1.0D9 / DBLE(NREP)

      WRITE(*, '(I3, 1X, E23.16, 1X, E23.16, 1X, E23.16, 1X,
     &  I12)') FID, FNU, ZR, ZI, NINT(ELAPSED_NS)

      GOTO 10

 999  CONTINUE
      END

C     Subroutine to dispatch a single Bessel/Airy call
      SUBROUTINE DO_CALL(FID, FNU, ZR, ZI, KODE, CYR, CYI,
     &                   AIR, AII, NZ, IERR, M)
      IMPLICIT NONE
      INTEGER FID, KODE, IERR, NZ, M
      DOUBLE PRECISION FNU, ZR, ZI, CYR(1), CYI(1), AIR, AII
      DOUBLE PRECISION CWRKR(1), CWRKI(1)

      IERR = 0
      NZ = 0

      IF (FID .EQ. 1) THEN
        CALL ZBESJ(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)
      ELSE IF (FID .EQ. 2) THEN
        CALL ZBESY(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ,
     &             CWRKR, CWRKI, IERR)
      ELSE IF (FID .EQ. 3) THEN
        CALL ZBESI(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)
      ELSE IF (FID .EQ. 4) THEN
        CALL ZBESK(ZR, ZI, FNU, KODE, 1, CYR, CYI, NZ, IERR)
      ELSE IF (FID .EQ. 5) THEN
        M = 1
        CALL ZBESH(ZR, ZI, FNU, KODE, M, 1, CYR, CYI, NZ, IERR)
      ELSE IF (FID .EQ. 6) THEN
        M = 2
        CALL ZBESH(ZR, ZI, FNU, KODE, M, 1, CYR, CYI, NZ, IERR)
      ELSE IF (FID .EQ. 7) THEN
        CALL ZAIRY(ZR, ZI, 0, KODE, AIR, AII, NZ, IERR)
        CYR(1) = AIR
        CYI(1) = AII
      ELSE IF (FID .EQ. 8) THEN
        CALL ZAIRY(ZR, ZI, 1, KODE, AIR, AII, NZ, IERR)
        CYR(1) = AIR
        CYI(1) = AII
      ELSE IF (FID .EQ. 9) THEN
        CALL ZBIRY(ZR, ZI, 0, KODE, AIR, AII, IERR)
        CYR(1) = AIR
        CYI(1) = AII
      ELSE IF (FID .EQ. 10) THEN
        CALL ZBIRY(ZR, ZI, 1, KODE, AIR, AII, IERR)
        CYR(1) = AIR
        CYI(1) = AII
      END IF

      RETURN
      END
