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


     y = d_g_y + index * 768; // load from device memory
     // GEMV1
     for (int jj = 0 ; jj < cols ; jj ++)
         c1 += sW [ (row*cols)+jj ] * x[ jj ];
         c2 += sW [ ((row+256)*cols)+jj ] * x[ jj ];
         c3 += sW [ ((row+512)*cols)+jj ] * x[ jj ];
      y[row] += c1 ;
      y[row+256] += c2 ;
      y[row+512] += c3 ;

      // ACTIVATION
      y[row] = gpu_logisticf(y[row]);
      y[row+256] = gpu_logisticf(y[row+256]);
      y[row+512] = gpu_tanhf(y[row+256] * y[row+512] + cinlocal);
      y[row+512] = (-1) * y[row] * y[row+512] + y[row+512];
      y[row] = y[row] * x[row] + y[row+512];
      __syncthreads();

      // GEMV2
      for (int jj = 0 ; jj < cols ; jj ++)
         c1 += W [ (row*cols)+jj ] * y[ jj ];
         c2 += W [ ((row+256)*cols)+jj ] * y[ jj ];
         c3 += W [ ((row+512)*cols)+jj ] * y[ jj ];

      // next layer data update directly in device memory
      xnext[row] = c1; xnext[row+256] = c2 ; xnext[row+512] = c3;
      x[row] = y[row]; // next invocation istate is from current ostate
      y[row] = c1; y[row+256] = c2 ; y[row+512] = c3;

