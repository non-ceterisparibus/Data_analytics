import math
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib as plt
from functools import partial
from scipy.special import factorial
from math import factorial as mfac


class DistFunctions():
    def __init__(self, n):
        """
        n : maximum trial
        """
        self.n = n

    def ProbFunc(self,f):
        self.func = lambda x: f(x)
        return self.func

    def ProbLinspace(self, X):
        """
        X = None for discrete distribution
        """
        # Create Linspace
        if X is None:
            X = np.array(range(0,self.n+1))
        # Mass distribution function
        mass = self.func(X)
        # Cumulative distribution function
        cum = np.empty(X.shape)
        cum = [sum(mass[:i]) for i in range(X.shape[0])]

        if X is None:
            data = {"x": list(range(0,self.n+1)), "Mass Probability": mass ,"Cumulative Probability": cum}
        else:
           data = {"x": X, "Mass Probability": mass ,"Cumulative Probability": cum}

        self.df = pd.DataFrame(data)
        return self.df
        
    def MassDist(self):
        "Mass Distribution"

        sns.set(font_scale = 1.2)
        g = sns.relplot(x="x", y="Mass Probability", kind="line", data=self.df)
        g.figure.set_size_inches(9, 6.5)
        return g
    
    def CumDist(self):
        "Cumulative Distribution"
        sns.set(font_scale = 1.2)
        g = sns.relplot(x="x", y="Cumulative Probability", kind="line", data=self.df)
        g.figure.set_size_inches(9, 6.5)
        return g


class Geometric(DistFunctions):
    def __init__(self,parameter,n):
        """
        p : probability parameter
        n : maximum trial
        """
        DistFunctions.__init__(self,n)
        self.p = parameter

    def GeoProb(self,x):
        return self.p*(1-self.p)**(x-1)
        
    def MeanGeo(self):
        self.mean = 1/self.p
        return
    
    def VarGeo(self):
        self.var = self.p
        return

class Poisson(DistFunctions):
    def __init__(self,parameter,n):
        """
        p : lambda
        n : maximum trial
        """
        DistFunctions.__init__(self,n)
        self.p = parameter
        
    def PoisProb(self,x):
        return np.power(self.p,x,dtype=np.int64) * (np.exp(-self.p)) /factorial(x)

    def MeanPois(self):
        self.mean = self.p
        return
    
    def VarPois(self):
        self.var = self.p
        return

class Binomial(DistFunctions):
    def __init__(self,parameter,n):
        """
        p : probability parameter
        n : maximum trial
        """
        DistFunctions.__init__(self,n)
        self.p = parameter

    def BinoProb(self,k):
        combination = math.factorial(self.n)/(factorial(k)*factorial(self.n-k))
        return combination * (self.p**k) * (1-self.p)**(self.n-k)

    def MeanBin(self):
        self.mean = self.p* self.n
        return

class Beta(DistFunctions):
    def __init__(self,alpha, beta, n):
        super().__init__(n)
        self.alpha = alpha
        self.beta = beta

    def BetaDensity(self,x):
        return mfac(self.alpha + self.beta - 1)/(mfac(self.alpha-1) * mfac(self.beta-1)) * (x**(self.alpha-1)) * (1-x)**(self.beta-1) 
    
    def BetaMean(self):
        return self.alpha/(self.alpha+ self.beta)
    
    def BetaVar(self):
        return (self.alpha* self.beta)/((self.alpha+self.beta+1)*(self.alpha+self.beta)**2)

class Gamma(DistFunctions):
    def __init__(self,alpha, beta, n):
        super().__init__(n)
        self.alpha = alpha
        self.beta = beta

    def GammaDensity(self,y):
        return np.power(self.beta, self.alpha)/mfac(self.alpha-1) * np.power(y,self.alpha-1) * np.exp(-self.beta*y) 
    
    def GammaMean(self):
        return self.alpha/self.beta
    
    def GammaVar(self):
        return self.alpha/(self.beta**2)