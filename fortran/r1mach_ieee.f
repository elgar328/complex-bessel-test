C     R1MACH for IEEE 754 single precision (needed by CBSUBS).
C     machcon_ieee.f provides I1MACH and D1MACH only.
      REAL FUNCTION R1MACH(I)
      INTEGER I
      REAL RMACH(5)
C     1: smallest normal, 2: largest, 3: 2**-24, 4: 2**-23, 5: log10(2)
      DATA RMACH / 1.17549435E-38, 3.40282347E+38, 5.96046448E-08,
     &             1.19209290E-07, 0.301029996 /
      IF (I .LT. 1 .OR. I .GT. 5) STOP 'R1MACH: I OUT OF RANGE'
      R1MACH = RMACH(I)
      RETURN
      END
