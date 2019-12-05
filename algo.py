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
       GruModStep(X, Y, istate, ostate)
