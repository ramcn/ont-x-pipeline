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

