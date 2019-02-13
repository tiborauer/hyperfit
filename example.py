import numpy, hyperfit
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rc('lines',linewidth=1)
mpl.rc('figure',dpi=300, figsize=[19.2, 9])
mpl.rc('legend',fontsize='large')
mpl.rc('axes', labelsize='large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')

## Prepare tasks
p0 = [4, 3, 2, 1, 5, 1, 6, 3]

fdef = hyperfit.fundef(basis='discrete_sinus_set', number_of_parameters=len(p0))

for duration in range(1,5):
    for noise in range(11):
        case = 'noise{}-duration{}'.format(noise,duration)
        print('Running: ', case)

        # Generate data
        x, y = fdef.generate_signal(p0,noise,1000,duration*1000)

        # Initialise
        f1 = hyperfit.hyperfit(name=case,signal=(x,y),function=fdef, kind='tpe');
        f2 = hyperfit.hyperfit(name=case,signal=(x,y),function=fdef, kind='rand');

        # Run fit
        f1.fit(); f2.fit(); 
        f1.save(); f2.save()
 
        # Plot
        f1.load(); f2.load()
    
        fig, ax = plt.subplots(2,1,sharex=True, sharey=True)
        f1.plot_loss(ax[0])
        f2.plot_loss(ax[1])
        fig.savefig(case + '_losses.svg')
        plt.close(fig)

        r = numpy.corrcoef(numpy.vstack((y,fdef.function(p0)(x),f1.result_signal,f2.result_signal)))
        r = r[1]

        fig, ax = plt.subplots(1,1)
        ax.plot(x,y, label='Data,r={:.3f}'.format(r[0]))
        ax.plot(x,fdef.function(p0)(x), label='Target: {}'.format(p0))
        h = f1.plot_result(ax)
        h[-1].set_label(h[-1].get_label() + ',r={:.3f}'.format(r[2]))
        h = f2.plot_result(ax)
        h[-1].set_label(h[-1].get_label() + ',r={:.3f}'.format(r[3]))
        ax.legend()
        fig.savefig(case + '_results.svg')
        plt.close(fig)

