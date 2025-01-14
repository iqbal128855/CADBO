from __future__ import division
import autograd.numpy as anp
import numpy as np
import math
#from scipy.interpolate import interp1d
from copy import deepcopy

class ObjectiveSynthetic:
    """This class is used to create objective space for synthetic cases"""
    def __init__(self):
        print ("[STATUS]: initializing objectivesynthetic class")

    def ZDT1(self, x, n_var):
        """This function is used to define ZDT1 synthetic function"""
        
        f1 = x[0]
        g = 1 + 9.0 / (n_var - 1) * np.sum(x[1:])
        f2 = g * (1 - anp.power((f1 / g), 0.5))
        
        return (f1, f2)
   
    def Branin(self, params, n_var):

        x = params[0]
        y = params[1]

        print 'Evaluating at (%f, %f)' % (x, y)

        obj1 = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)
 
        obj2 = -obj1
	
        return (obj1, obj2)

    def ZDT6(self, x, n_var):
        """This function is used to define ZDT6 synthetic function"""
        f1 = 1 - anp.exp(-4 * x[0]) * anp.power(anp.sin(6 * anp.pi * x[0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[1:], axis=1) / (n_var - 1.0), 0.25)
        f2 = g * (1 - anp.power(f1 / g, 2))
        return (f1, f2)

    def Kursawe(self, x, n_var):
        """This function is used to define Kursawe synthetic function"""
        l = []
        print (x)
        for i in range(2):
            l.append(-10 * anp.exp(-0.2 * anp.sqrt(anp.square(x[:, i]) + anp.square(x[:, i + 1]))))
        f1 = anp.sum(anp.column_stack(l), axis=1)

        f2 = anp.sum(anp.power(anp.abs(x), 0.8) + 5 * anp.sin(anp.power(x, 3)), axis=1)
        return (f1, f2)
    

    def Currin(self, x, d):
        return -1*float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

    def branin(self, x1,d):
        x=deepcopy(x1)
        x[0]= 15* x[0]-5
        x[1]=15*x[1]
        return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)




    def Powell(self, xx,d):

        vmin=-4
        vmax=5

        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        f_original=0
        for i in range(1,int(math.floor(d/4)+1)):
            f_original=f_original+pow(x[4*i-3]+10*x[4*i-2],2)+5*pow(x[4*i-1]-x[4*i],2)+pow(x[4*i-2]-2*x[4*i-1],4)+10*pow(x[4*i-3]-2*x[4*i],4)
        return -1*float(f_original)

    def Perm(xx,d):
        vmin=-1*d
        vmax=d
        beta=10
        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        f_original=0
        for i in range(1,d+1):
            sum1=0
            for j in range(1,d+1):
                sum1=sum1+(j+beta)*(x[j]-math.pow(j,-1*i))        
            f_original=f_original+math.pow(sum1,2)
        return -1*f_original

    def Dixon(xx,d):
        vmin=-10
        vmax=10
        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        f_original=0
        for i in range(2,d+1):    
            f_original=f_original+i*math.pow(2*math.pow(x[i],2)-x[i-1],2)
        f_original=f_original+math.pow(x[1]-1,1)
        return -1*f_original

    def ZAKHAROV(xx,d):
        vmin=-5
        vmax=10
        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        term1=0
        term2=0
        for i in range(1,d+1):
            term1=term1+x[i]**2
            term2=term2+0.5*i*x[i]
        f_original=term1+math.pow(term2,2)+math.pow(term2,4)
        return -1*f_original
    def RASTRIGIN(xx,d):
        vmin=-5.12
        vmax=5.12
        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        f_original=0
        for i in range(1,d+1):
            f_original=f_original+(x[i]**2-10*math.cos(2*x[i]*math.pi))
        f_original=f_original+10*d
        return -1*f_original

    def SumSquares(xx,d):
        vmin=-5.12
        vmax=5.12
        x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
        f_original=0
        for i in range(1,d+1):
            f_original=f_original+(i*math.pow(x[i],2))
        return -1*f_original

    def oka21(xx, d):
        x=deepcopy(xx)
        x[0]=x[0]*(2*3.14)-3.14
        f_original = x[0]
        return -1*f_original

    def oka22(xx, d):
        x=deepcopy(xx)
        x[0]=x[0]*(2*3.14)-3.14
        x[1]=x[1]*(2*5)-5
        x[2]=x[2]*(2*5)-5
        f_original=1-1./(4*pow(math.pi,2))*pow(x[0]+math.pi,2)+pow(np.abs(x[1]-5*math.cos(x[0])),1./3)+pow(np.abs(x[2]-5*math.sin(x[0])),1./3)
        return -1*f_original

    def DTLZ14f_1(x, d):
        g=0
        for i in range(d):
            g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
        g=100*(d+g)
        y1=(1+g)*0.5*x[0]*x[1]*x[2]
        return -1*y1
    
    def DTLZ14f_2(x, d):
        g=0
        for i in range(d):
            g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
        g=100*(d+g)
        y2=(1+g)*0.5*(1-x[2])*x[0]*x[1]
        return -1*y2

    def DTLZ14f_3(x, d):
        g=0
        for i in range(d):
            g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
        g=100*(d+g)
        y3=(1+g)*0.5*(1-x[1])*x[0]
        return -1*y3

    def DTLZ14f_4(x, d):
        g=0
        for i in range(d):
            g=g+pow(x[i]-0.5,2)-math.cos(20*math.pi*(x[i]-0.5))    
        g=100*(d+g)
        y4=(1+g)*0.5*(1-x[0])
        return -1*y4    

 
