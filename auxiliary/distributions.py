import numpy as np
import seaborn as sns
import pandas as pd
from ipywidgets import widgets

from scipy.special import factorial
from math import factorial as mfac
import plotly.graph_objects as go

class DistFunctions():
    def __init__(self, n):
        if n is None:
            self.n = 20
        else:
            self.n = n
        
        self.alpha = widgets.IntSlider(
            value=2.0,
            min=1.0,
            max=7.0,
            step=1.0,
            description='Alpha:',
            continuous_update=True)
        self.beta = widgets.IntSlider(
            value=5.0,
            min=1.0,
            max=7.0,
            step=1,
            description='Beta:',
            continuous_update=True)

        self.p = widgets.FloatSlider(
            value = 0.5,
            min = 0.0,
            max =1.0,
            step = 0.1,
            description='p:',
            continuous_update=True)

        self.l = widgets.FloatSlider(
            value = 3,
            min= 1.0,
            max= 15.0,
            step= 0.5,
            description='Lambda:',
            continuous_update=True)

    def ProbFunc(self,f):
        self.func = lambda x: f(x)
        return

    def ProbLinspace(self, X):
        """
        Note:
        X = None - for discrete distribution
        """
        # Create Linspace
        if X is None:
            self.X = np.array(range(1, self.n+1))
        else:
            self.X = X

        # Mass distribution function
        self.mass = self.func(self.X)

        # Cumulative distribution function
        # cum = np.empty(X.shape)
        # cum = [sum(mass[:i]) for i in range(X.shape[0])]

        data = {
            "x": self.X, 
            "Mass Probability": self.mass,
            # "Cumulative Probability": cum
            }

        self.df = pd.DataFrame(data)

        return self.df
        
    def MassDistribution(self):
        "Mass Distribution"

        sns.set(font_scale = 1.2)
        g = sns.relplot(x="x", y="Mass Probability", kind="line", data=self.df)
        g.figure.set_size_inches(9, 6.5)
        return 
    
    def CumDistribution(self):
        "Cumulative Distribution"
        sns.set(font_scale = 1.2)
        g = sns.relplot(x="x", y="Cumulative Probability", kind="line", data=self.df)
        g.figure.set_size_inches(9, 6.5)
        return 

    def event_handler(self, change):
            new_data = self.NewData()
            with self.fig.batch_update():
                self.fig.data[0].x = new_data['x']
                self.fig.data[0].y = new_data['Mass Probability']
    
    def InteractivePlot(self, y_axis):
        """"
        """
        # Default dataframe
        df = self.NewData()

        # Assign Figure
        trace = go.Scatter(x=df['x'], y=df['Mass Probability'])

        if y_axis == None:
            self.fig = go.FigureWidget(
            data=[trace],
            layout=go.Layout(
                title='Density Probability',
                width=900,
                height=500,
                )
            )
        else:
            self.fig = go.FigureWidget(
                data=[trace],
                layout=go.Layout(
                    title='Density Probability',
                    width=900, 
                    height=500,
                    yaxis=y_axis
                )
            )
        return
    
    def InteractivePlotP(self):
        """
        """
        self. InteractivePlot(y_axis=dict(range=[0.0, 1.0]))

        self.p.observe(self.event_handler, names ='value')
        self.container = widgets.HBox(
            children=[self.p]
            )
        return

    def InteractivePlotLambd(self):
        """
        """
        self. InteractivePlot(y_axis=None)

        self.l.observe(self.event_handler, names ='value')
        self.container = widgets.HBox(
            children=[self.l], 
            )
        return
    
    def InteractivePlotAB(self):
        """
        """
        self. InteractivePlot(y_axis=None)

        self.alpha.observe(self.event_handler, names ='value')
        self.beta.observe(self.event_handler, names ='value')

        self.container = widgets.HBox(
            children=[
                self.alpha, 
                self.beta,
                ]
            )
        return

# Discrete Distribution
class Geometric(DistFunctions):

    def Prob(self,x):
        return self.p.value * (1-self.p.value)**(x-1)
        
    def Mean(self):
        self.mean = 1/self.p.value
        return self.mean
    
    def Var(self):
        self.var = self.p.value
        return self.var

    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = None)
        return new_data

class Binomial(DistFunctions):

    def Prob(self,x):
        combination = factorial(self.n)/(factorial(x)*factorial(self.n-x))
        return combination * (self.p.value** x) * (1-self.p.value)**(self.n-x)

    def Mean(self):
        self.mean = self.p.value* self.n
        return self.mean
    
    def Var(self):
        self.var = self.n * self.p.value * (1 - self.p.value)
        return self.var

    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = None)
        return new_data

class Poisson(DistFunctions):
        
    def Prob(self,x):
        return np.float_power(self.l.value, x) * np.exp(-self.l.value)/factorial(x)

    def Mean(self):
        self.mean = self.l.value
        return self.mean
    
    def Var(self):
        self.var = self.l.value
        return self.var
    
    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = None)
        return new_data

# Continuous Distribution
class Exponential(DistFunctions):

    def Prob(self,x):
        return self.l.value* np.exp(-self.l.value* x)
    
    def Mean(self):
        self.mean = 1/self.l.value
        return self.mean
    
    def Var(self):
        self.var = 1/(self.l.value**2)
        return self.var

    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = np.linspace(0, self.n , self.n*10))
        return new_data

class Beta(DistFunctions):

    def Prob(self,x):
        return mfac(self.alpha.value + self.beta.value - 1)/(mfac(self.alpha.value-1) * mfac(self.beta.value-1)) * (x**(self.alpha.value-1)) * (1-x)**(self.beta.value-1) 
    
    def Mean(self):
        self.mean = self.alpha/(self.alpha.value + self.beta.value)
        return self.mean
    
    def Var(self):
        self.var = (self.alpha.value * self.beta.value)/((self.alpha.value + self.beta.value + 1)*(self.alpha.value + self.beta.value )**2)
        return self.var
    
    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = np.linspace(0,1,self.n))
        return new_data

class Gamma(DistFunctions):

    def Prob(self,y):
        return np.power(self.beta.value, self.alpha.value)/mfac(self.alpha.value-1) * np.power(y,self.alpha.value - 1) * np.exp(-self.beta.value * y) 
    
    def Mean(self):
        self.mean = self.alpha.value/self.beta.value
        return self.mean
    def Var(self):
        self.var = self.alpha.value/(self.beta.value**2)
        return self.var
    
    def NewData(self):
        self.ProbFunc(self.Prob)
        new_data = self.ProbLinspace(X = np.linspace(0, self.n ,self.n*10))
        return new_data
