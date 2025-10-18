import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
sns.set_context('notebook')

if __name__ == '__main__':
    Wpanel = 4
    Hpanel = 3
    _, axgrid = plt.subplots(
        nrows=1, ncols=2,
        sharex=True, sharey=True, squeeze=True,
        figsize=(Wpanel * 2, Hpanel * 1))

    df = pd.read_csv('traffic_train.csv')
    tdf = pd.read_csv('traffic_test.csv')

    axgrid[0].plot(df.x, df.y, 'k.')
    axgrid[0].set_title('Train data (N=%d)' % df.shape[0])
    axgrid[1].plot(tdf.x, tdf.y, 'r.')
    axgrid[1].set_title('Test data (N=%d)' % (tdf.shape[0]))
    for ii, ax in enumerate(axgrid):
        ax.set_ylim([-0.1, 8.5])
        ax.set_xlim([-1.75, 1.6]) # little to left for legend
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([-24, -12, 0, 12, 24])
        ax.set_yticks([0, 3, 6])
        ax.grid(axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("hours since Mon midnight")
        if ii == 0:
            ax.set_ylabel("traffic count (1000s)")
            #ax.legend(loc='upper left', fontsize=0.9*plt.rcParams['legend.fontsize'])
    plt.subplots_adjust(top=0.9, bottom=0.19)
    plt.savefig("traffic_data.png", bbox_inches='tight', pad_inches=0)
    plt.show()

