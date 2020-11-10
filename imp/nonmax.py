from __future__ import division
from numpy import array, zeros
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, figure, gray, title, axis
from numpy.fft import fft2, ifft2
from numpy import array, zeros, abs, sqrt, arctan2, arctan, pi, real
from scipy import ndimage
import numpy as np


def non_max_suppression(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    Kx=[[-1,0,+1] ,[-1,0,+1] ,[-1,0,+1]]
    Ky=[[-1,-1,-1], [0,0,0], [+1,+1,+1]]
    Gx = ndimage.filters.convolve(img, Kx)
    Gy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Gx, Gy)
    G = G / G.max() 
    theta = np.arctan2(Gy, Gx)  

    M, N = img.shape
    Z = np.zeros((M,N))
    angle = theta * 180. / np.pi
    #angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                if (angle[i,j]>=0 and angle[i,j]<=45) or (angle[i,j]<-135 and angle[i,j]>=-180):
                  yBot = [G[i,j+1], G[i+1,j+1]]
                  yTop = [G[i,j-1], G[i-1,j-1]]
                  x_est = abs(Gy[i,j]/G[i,j])
                  if (G[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and G[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    Z[i,j] = G[i,j]
                if (angle[i,j]>45 and angle[i,j]<=90) or (angle[i,j]<-90 and angle[i,j]>=-135):
                  yBot = [G[i+1,j], G[i+1,j+1]]
                  yTop = [G[i-1,j], G[i-1,j-1]]
                  x_est = abs(Gx[i,j]/G[i,j])
                  if (G[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and G[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    Z[i,j]= G[i,j]
                if (angle[i,j]>90 and angle[i,j]<=135) or (angle[i,j]<-45 and angle[i,j]>=-90):
                  yBot = [G[i+1,j], G[i+1,j-1]]
                  yTop = [G[i-1,j], G[i-1,j+1]]
                  x_est = abs(Gx[i,j]/G[i,j])
                  if G[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and G[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0]):
                    #print('x_est is', G[i,j],((yBot[1]-yBot[0])*x_est+yBot[0]), ((yTop[1]-yTop[0])*x_est+yTop[0]) )
                    Z[i,j]= G[i,j]
                if (angle[i,j]>135 and angle[i,j]<=180) or (angle[i,j]<0 and angle[i,j]>=-45):
                  yBot = [G[i,j-1], G[i+1,j-1]]
                  yTop = [G[i,j+1], G[i-1,j+1]]
                  x_est = abs(Gx[i,j]/G[i,j])
                  if (G[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and G[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    Z[i,j]= G[i,j]
            except IndexError as e:
                print('oHHHHH NNOOOOO')
                pass
    
    return Z





def gradient(im):
    # Sobel operator
    op1 = array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
    op2 = array([[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]])
    kernel1 = zeros(im.shape)
    kernel1[:op1.shape[0], :op1.shape[1]] = op1
    kernel1 = fft2(kernel1)

    kernel2 = zeros(im.shape)
    kernel2[:op2.shape[0], :op2.shape[1]] = op2
    kernel2 = fft2(kernel2)

    fim = fft2(im)
    Gx = real(ifft2(kernel1 * fim)).astype(float)
    Gy = real(ifft2(kernel2 * fim)).astype(float)

    G = sqrt(Gx**2 + Gy**2)
    Theta = arctan2(Gy, Gx) * 180 / pi
    return G, Theta

def non_max_sup(im):
  det, phase = gradient(im)
  gmax = zeros(det.shape)

  for i in range(gmax.shape[0]):
    for j in range(gmax.shape[1]):
      if phase[i][j] < 0:
        phase[i][j] += 360

      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
        # 0 degrees
        if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
          if det[i][j] > det[i][j + 1] and det[i][j] > det[i][j - 1]:
            gmax[i][j] = im[i][j]
        # 45 degrees
        if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
          if det[i][j] > det[i - 1][j + 1] and det[i][j] > det[i + 1][j - 1]:
            gmax[i][j] = im[i][j]
        # 90 degrees
        if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
          if det[i][j] > det[i - 1][j] and det[i][j] > det[i + 1][j]:
            gmax[i][j] = im[i][j]
        # 135 degrees
        if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
          if det[i][j] > det[i - 1][j - 1] and det[i][j] > det[i + 1][j + 1]:
            gmax[i][j] = im[i][j]
  return gmax

if __name__ == '__main__':
    im = array(Image.open('./out.png'))
    print('image shape', im.min(), im.max())
    gmax = maximum(im)

    subplot(1, 2, 1)
    imshow(im)
    axis('off')
    title('Original')

    subplot(1, 2, 2)
    imshow(gmax)
    axis('off')
    title('Non-Maximum suppression')

    show()
