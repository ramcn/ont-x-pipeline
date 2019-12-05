def network(X, W):
    istate = Convolution(X, W, winlen, fun=tanh)
    ostate1 = GruModBck(X, W, istate)
    ostate2 = GruModFwd(X, W, ostate1)
    ostate3 = GruModBck(X, W, ostate2)
    ostate4 = GruModFwd(X, W, ostate3)
    ostate5 = GruModBck(X, W, ostate4)
    GlobalNormFlipFlop(size, nbase)

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

