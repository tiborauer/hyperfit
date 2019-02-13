import numpy, random
import matplotlib.pyplot as plt
import hyperopt
import pickle
import inspect

class fundef:

    def __init__(self, basis='discrete_sinus_set', number_of_parameters=4):
        self.number_of_parameters = number_of_parameters
        self.function = getattr(self, '_{}__{}'.format(self.__class__.__name__, basis))
	

    def generate_signal(self,parameters=[0,0,0,0], noiseLevel=0, sample_per_second=1000, sample=1000):
        x = numpy.arange(0, sample/sample_per_second, 1/sample_per_second)
        y = self.function(parameters=parameters)(x)+noiseLevel*(numpy.random.rand(sample)*2-1)
        return x,y

    # bases
    def __discrete_sinus_set(self,parameters):
        s = "lambda x: "
        s += " + ".join(["parameters[" + str(i*2) + "]*sin(2*pi*" + str(i+1) + "*(x)+parameters[" + str(i*2+1) + "])" for i in range(0, int(len(parameters)/2))])
        return eval(s,{'sin': numpy.sin, 'pi':numpy.pi, 'parameters': parameters})
   
class hyperfit:
    def __init__(self,name='',signal=None, function=fundef(basis='discrete_sinus_set', number_of_parameters=4), kind='tpe'):
        self.name = name
        self.kind = kind
        self.max_evals = 1000
        self.__function = function.function
        self.__x = signal[0]
        self.__y = signal[1]
        self.__lossfunction = lambda p: sum((self.__y-self.__function(p)(self.__x))**2)
        self.__space=[hyperopt.hp.quniform('p{}'.format(i),1,10,1) for i in range(function.number_of_parameters)]
        self.__trials = hyperopt.Trials()
        self.duration = None
        self.losses = None
        self.result_parameter = None
        self.result_signal = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.save()

    def fit(self):
        self.best = hyperopt.fmin(self.__lossfunction, space=self.__space, algo=getattr(hyperopt,self.kind).suggest,
            max_evals=self.max_evals, trials=self.__trials)
        self.duration = self.__trials.trials[-1]['refresh_time']-self.__trials.trials[0]['book_time']
        self.losses = self.__trials.losses()
        self.result_parameter = [int(v[0]) for v in list(self.__trials.best_trial['misc']['vals'].values())]
        self.result_signal = self.__function(self.result_parameter)(self.__x)

    def save(self):
        with open('{}_{}_trials.pcl'.format(self.name,self.kind),'wb') as f: pickle.dump(self.__trials, f)
        
    def load(self):
        with open('{}_{}_trials.pcl'.format(self.name,self.kind),'rb') as f: self.__trials = pickle.load(f)

    def plot_loss(self,ax=None):
        toPlot = False
        if ax is None:
            fig, ax = plt.subplots(1,1) 
            toPlot = True
        ax.plot(self.losses, label='{}_{} - {}'.format(self.name,self.kind,self.duration))
        ax.set(xlabel='iter', ylabel='loss')
        ax.grid()
        ax.legend()
        if toPlot: plt.show(block=False) 
        return ax.get_lines()

    def plot_result(self,ax=None):
        toPlot = False;
        if ax is None:
            fig, ax = plt.subplots(1,1) 
            toPlot = True
            ax.plot(self.__x, self.__y, llabel='Data')
        ax.plot(self.__x, self.result_signal, label='{}_{} - {}'.format(self.name,self.kind,self.result_parameter))
        ax.set(xlabel='x', ylabel='y')
        ax.grid()
        ax.legend()
        if toPlot: plt.show(block=False)
        return ax.get_lines()
