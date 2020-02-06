def Network(X, W):
    conv = Convolution(X, W)
    X1 = GruModBck(X,sW1,W1)
    X2 = GruModFwd(X1,sW2,W2)
    X3 = GruModBck(X2,sW3,W3)
    X4 = GruModFwd(X3,sW4,W4)
    X5 = GruModBck(X3,sW5,W5)
    trans = GlobalNormFlipFlop(X5)

def GruModFwd(X,sW,W):
    for (i=0;i<N;i=i+1):
       GruModStep(sW,W,X[i],ostate)

def GruModBck(X,sW,W):
    for (i=N;i>0;i=i-1):
       GruModStep(sW,W,X[i],ostate)

def GruModStep(sW,W,istate,ostate):
     cblas_sgemv(768,256,1.0,sW,istate,1,1.0,Cout,1)

     for ( i = 0; i < 256; i = i+1):
       Cout[i] = LOGISTICF(Cout[i]) // Update gate u(t) 
       Cout[size+i] = LOGISTICF(Cout[size+i]) // RESET gate r(t)
       Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] 
			   + Cin[i+size+size]) // ~O(t)
       ostate[i] = (-1) * Cout[i] * Cout[i+size+size] 
			   + Cout[i+size+size]  // O(t) part 2
       ostate[i] = Cout[i] * istate[i] + ostate[i]  // O(t) part 1

     cblas_sgemv(766,256,1.0,W,ostate,1,1.0,XnextBuf+bias,1);


