import numpy as np
import matplotlib.pyplot as plt

def plot_ndcg_barchart():
    # Mean & 200 & 1.0 & 0.7210 & 0.8715 & 0.9980 & 0.9978 & 0.9442  \\
    # Mean & 100 & 1.0 & 0.6862 & 0.8475 & 0.8329 & 0.9128 & 0.7933 \\
    # train_scores = np.array([0.8715, 0.9966, 0.9979, 0.8732])
    train_scores = np.array([0.8715, 0.9980, 0.9978, 0.9442])
    test_scores = np.array([0.8475, 0.8329,0.9128, 0.7933])

    _min = min(0.7210, 0.6862)
    print(_min)
    '''train_scores -= _min
    test_scores -= _min'''
    opt = np.array([1] * train_scores.size) #- _min

    ind = np.arange(len(train_scores))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, train_scores, width, #yerr=men_std,
                    color='SkyBlue', label='Training Set',edgecolor=['k']*4)
    rects2 = ax.bar(ind + width/2, test_scores, width, #yerr=women_std,
                    color='IndianRed', label='Extrapolation Set',edgecolor=['k']*4)

    ax.plot(opt, '--')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('NDCG Scores')
    ax.set_title('NDCG Scores on Training and Test data')
    ax.set_xticks(ind)
    ax.set_xticklabels(( "Random", "Pair", "Triplet", "Naive Triplet"))
    ax.legend(loc=4)


    ax.set_ylim(bottom=_min)
    ax.set_ylim(ymax=1)

    plt.show()

def plot_spearman_barchart():

    train_scores = np.array([0.8715, 0.9966, 0.9979, 0.8732])
    test_scores = np.array([0.8475, 0.8257,0.9209, 0.7695])

    _min = min(0.7210, 0.6862)
    print(_min)
    '''train_scores -= _min
    test_scores -= _min'''
    opt = np.array([1] * train_scores.size) #- _min

    ind = np.arange(len(train_scores))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, train_scores, width, #yerr=men_std,
                    color='SkyBlue', label='Training Set')
    '''rects2 = ax.bar(ind + width/2, test_scores, width, #yerr=women_std,
                    color='IndianRed', label='Extrapolation Set')'''

    ax.plot(opt, '--')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('NDCG Scores')
    ax.set_title('NDCG Scores on Training and Test data')
    ax.set_xticks(ind)
    ax.set_xticklabels(( "Random", "Pair", "Triplet", "Naive Triplet"))
    ax.legend(loc=4)


    ax.set_ylim(bottom=_min)
    ax.set_ylim(ymax=1)

    plt.show()

def plot_correlation():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('mathtext', default='regular')

    pi = np.pi

    # fake data
    time = np.linspace(0, 25, 50)
    temp = 50 / np.sqrt(2 * pi * 3 ** 2) \
           * np.exp(-((time - 13) ** 2 / (3 ** 2)) ** 2) + 15
    Swdown = 400 / np.sqrt(2 * pi * 3 ** 2) * np.exp(-((time - 13) ** 2 / (3 ** 2)) ** 2)
    Rn = Swdown - 10

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(time, Swdown, '-', label='Swdown')
    ax.plot(time, Rn, '-', label='Rn')
    ax2 = ax.twinx()
    ax2.plot(time, temp, '-r', label='temp')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.grid()
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
    ax2.set_ylabel(r"Temperature ($^\circ$C)")
    ax2.set_ylim(0, 35)
    ax.set_ylim(-20, 100)
    plt.show()

def sub_plots():
    # Some example data to display
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(x, y)
    ax2.plot(x, -y)

    plt.show()

sub_plots()
