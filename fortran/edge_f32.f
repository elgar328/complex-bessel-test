C     Edge-test driver, single precision (TOMS 644 CBSUBS).
C     Same input/output format as edge_f64.f.
C     CBIRY has no NZ argument; NZ is reported as 0.
      PROGRAM EDGE32
      IMPLICIT NONE
      INTEGER NMAX
      PARAMETER (NMAX=256)
      CHARACTER*2 FN
      INTEGER KODE, N, NZ, IERR, I, IOS
      REAL FNU, ZR, ZI
      COMPLEX Z, A, CY(NMAX), CW(NMAX)

   10 CONTINUE
      READ(*, *, IOSTAT=IOS) FN, KODE, N, FNU, ZR, ZI
      IF (IOS .NE. 0) GO TO 99
      IF (N .LT. 1 .OR. N .GT. NMAX) STOP 'EDGE32: N OUT OF RANGE'
      DO 20 I = 1, N
        CY(I) = (0.0E0, 0.0E0)
   20 CONTINUE
      Z = CMPLX(ZR, ZI)
      NZ = 0
      IERR = 0

      IF (FN .EQ. 'J ') THEN
        CALL CBESJ(Z, FNU, KODE, N, CY, NZ, IERR)
      ELSE IF (FN .EQ. 'Y ') THEN
        CALL CBESY(Z, FNU, KODE, N, CY, NZ, CW, IERR)
      ELSE IF (FN .EQ. 'I ') THEN
        CALL CBESI(Z, FNU, KODE, N, CY, NZ, IERR)
      ELSE IF (FN .EQ. 'K ') THEN
        CALL CBESK(Z, FNU, KODE, N, CY, NZ, IERR)
      ELSE IF (FN .EQ. 'H1') THEN
        CALL CBESH(Z, FNU, KODE, 1, N, CY, NZ, IERR)
      ELSE IF (FN .EQ. 'H2') THEN
        CALL CBESH(Z, FNU, KODE, 2, N, CY, NZ, IERR)
      ELSE IF (FN .EQ. 'AI' .OR. FN .EQ. 'AP') THEN
        A = (0.0E0, 0.0E0)
        IF (FN .EQ. 'AI') CALL CAIRY(Z, 0, KODE, A, NZ, IERR)
        IF (FN .EQ. 'AP') CALL CAIRY(Z, 1, KODE, A, NZ, IERR)
        CY(1) = A
      ELSE IF (FN .EQ. 'BI' .OR. FN .EQ. 'BP') THEN
        A = (0.0E0, 0.0E0)
        IF (FN .EQ. 'BI') CALL CBIRY(Z, 0, KODE, A, IERR)
        IF (FN .EQ. 'BP') CALL CBIRY(Z, 1, KODE, A, IERR)
        CY(1) = A
      ELSE
        STOP 'EDGE32: UNKNOWN FUNCTION'
      END IF

C     ES16.9E2: 9 significant digits round-trip a single.
      WRITE(*, '(I2, 1X, I4, 512(1X, ES16.9E2))')
     &  IERR, NZ, (REAL(CY(I)), AIMAG(CY(I)), I = 1, N)
      GO TO 10

   99 CONTINUE
      END
