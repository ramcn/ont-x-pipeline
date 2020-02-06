def Network(X, W):
    conv = Convolution(X, W)
    X1 = feedforward_linear(conv,W1)
    ostate1 = GruModBck(X1,sW1)
    X2 = feedforward_linear(X1,W2,ostate1)
    ostate2 = GruModFwd(X2,sW2)
    X3 = feedforward_linear(X2, W3,ostate2)
    ostate3 = GruModBck(X3,sW3)
    X4 = feedforward_linear(X3,W4,ostate3)
    ostate4 = GruModFwd(X4,sW4)
    X5 = feedforward_linear(X4,W5,ostate4)
    ostate5 = GruModBck(X5,sW5)
    trans = GlobalNormFlipFlop(ostate5)

def GruModFwd(X,sW ):
    for (i=0;i<N;i=i+1):
       GruModStep(W,X[i],ostate)

def GruModBck(X, sW):
    for (i=N;i>0;i=i-1):
       GruModStep(W,X[i],ostate)

def GruModStep(W,istate,ostate):
     cblas_sgemv(768,256,1.0,W,istate,1,1.0,Cout,1)

     for ( i = 0; i < 256; i = i+1):
       Cout[i] = LOGISTICF(Cout[i]) // Update gate u(t) 
       Cout[size+i] = LOGISTICF(Cout[size+i]) // RESET gate r(t)
       Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] 
			   + Cin[i+size+size]) // ~O(t)
       ostate[i] = (-1) * Cout[i] * Cout[i+size+size] 
			   + Cout[i+size+size]  // O(t) part 2
       ostate[i] = Cout[i] * istate[i] + ostate[i]  // O(t) part 1

