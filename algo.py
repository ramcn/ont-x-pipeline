def Network(X, W):
    istate = Convolution(X, W)
    ostate1 = GruModBck(X, W, istate)
    X1 = feedforward_linear(X,ostate1)
    ostate2 = GruModFwd(X1, W, ostate1)
    X2 = feedforward_linear(X1,ostate2)
    ostate3 = GruModBck(X2, W, ostate2)
    X3 = feedforward_linear(X2,ostate3)
    ostate4 = GruModFwd(X3, W, ostate3)
    X4 = feedforward_linear(X3,ostate4)
    ostate5 = GruModBck(X4, W, ostate4)
    trans = GlobalNormFlipFlop(ostate5)

def GruModFwd(X, W, istate):
    for (i=0;i<N;i=i+1):
       GruModStep(X, W, istate, ostate)

def GruModBck(X, W, istate):
    for (i=N;i>0;i=i-1):
       GruModStep(X, W, istate, ostate)

def GruModStep(Cin, Cout, A, Bnext):
     Cout = Cin
     cblas_sgemv(768, 256, 1.0, A, B, 1, 1.0, Cout, 1)

     for ( i = 0; i < 256; i = i+1):
       Cout[i] = LOGISTICF(Cout[i]) // Update gate u(t) 
       Cout[size+i] = LOGISTICF(Cout[size+i]) // RESET gate r(t)
       Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] 
			   + Cin[i+size+size]) // ~O(t)
       Bnext[i] = (-1) * Cout[i] * Cout[i+size+size] 
			   + Cout[i+size+size]  // O(t) part 2
       Bnext[i] = Cout[i] * B[i] + Bnext[i]  // O(t) part 1

