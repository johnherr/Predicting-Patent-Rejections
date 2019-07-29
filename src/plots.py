import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def plot_pca_two_components(ax, X, y, title='PCA_2components', save=False):
    """
    Parameters
    ----------
    ax: matplotlib.axis object
    The axis to make the scree plot on.

    X: numpy.array, shape (n, 300)
    A two dimensional array containing the vectorized claims

    y: numpy.array
    The labels of the datapoints.  1 or 0.

    title: str
    A title for the plot.
    """
    n_components =2
    pca = PCA(n_components=n_components) #pca object
    X = pca.fit_transform(X)

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    ax.scatter(X[:, 0], X[:, 1], s=10, c = y, alpha=.5, cmap=cm.coolwarm)
    ax.set_xticks([]),
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])
#     if title is not None:
#         ax.set_title(title, fontsize=16)
    if save:
        plt.savefig('images/{}.png'.format(title), dpi= 300)

def plot_pca_three_components(X, y, title='PCA_3components'):
    '''Function for creating a 3D animatino showing the
    frist three principal componets

    X: numpy.array, shape (n, 300)
    A two dimensional array containing the vectorized claims

    y: numpy.array
    The labels of the datapoints.  1 or 0.

    title: str
    A title for the plot.
    '''

    n_components = 3
    pca = PCA(n_components=n_components) #pca object
    X_pca = pca.fit_transform(X)

    xx = X_pca[:, 0]
    yy = X_pca[:, 1]
    zz = X_pca[:, 2]

    #Creating arays for axes & axes limits
    zeros = np.zeros(100)
    xx_center= np.mean(X_pca[:, 0])
    xx_range = np.max(X_pca[:, 0]) - np.min(X_pca[:, 0])*.5
    yy_center= np.mean(X_pca[:, 1])
    yy_range = np.max(X_pca[:, 1]) - np.min(X_pca[:, 1])*.5
    zz_center= np.mean(X_pca[:, 2])
    zz_range = np.max(X_pca[:, 2]) - np.min(X_pca[:, 2])*.5
    x_axis = np.linspace(xx_center-xx_range, xx_center+xx_range, 100)
    y_axis = np.linspace(yy_center-yy_range, yy_center+yy_range, 100)
    z_axis = np.linspace(zz_center-zz_range, zz_center+zz_range, 100)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)

    def init():
        # Plot the surface.
        ax.scatter(xx, yy, zz, marker='o', cmap=cm.coolwarm, s=10, c=y, alpha=0.3)
        ax.plot(x_axis,zeros,zeros, linewidth=4.0, color='k', alpha =1)
        ax.plot(zeros, y_axis,zeros, linewidth=4.0, color='k', alpha =1)
        ax.plot(zeros, zeros, z_axis, linewidth=4.0, color='k', alpha =1)
        ax.set_title('First 3 principal axis')
        ax.axis('off')
        ax.set_xlim(xx_center-xx_range/2, xx_center+xx_range/2)
        ax.set_ylim(yy_center-yy_range/2, yy_center+yy_range/2)
        ax.set_zlim(zz_center-zz_range/2, zz_center+zz_range/2)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig,

    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        ax.view_init(elev=30, azim=i*2)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=180, blit=True)
    fn = '3d_principal_pca_doc2vec'
    ani.save(fn+'.gif',writer='imagemagick',fps=30,savefig_kwargs={'transparent': True, 'facecolor': 'none'})


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=18, save=False):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    sns.set(font_scale=2)
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap="YlGnBu")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
    plt.ylabel('True', fontsize=40)
    plt.xlabel('Predicted', fontsize=40)

    if save:
        plt.savefig('images/confusion_matrix.png', dpi= 300, bbox_inches="tight")
    return fig

def plot_feature_comparison(df, save=False):
    """Plots graphs shoiwng the difference between vectorized features of
    the allowed/rejected claims

    ARGS: df with vectorized claim columns 'filed_claims_vec' and 'granted_claims_vec'
    """
    if 'claim_vec' not in df.keys():
        return 'df does not have vecortized claim data'

    accepted_mask = df['allowed']==1

    mean_rejected = df[~accepted_mask]['claim_vec'].mean()
    mean_accepted = df[accepted_mask]['claim_vec'].mean()
    mean_diff = np.sort((mean_accepted-mean_rejected), axis=-1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].scatter(range(300), mean_rejected ,color = 'r')
    ax[0].scatter(range(300), mean_accepted ,color = 'b')
    ax[0].legend(['Rejected','Accepted'])
    ax[0].set_xlabel('Feature')
    ax[0].set_ylabel('Mean Value')

    ax[1].scatter(range(300), mean_diff ,color = 'k')
    ax[1].set_ylabel('Avg. Feature Value Difference')
    ax[1].set_xlabel('Features (Ordered by Mean Value Difference)')

    if save:
        plt.savefig('images/feature_comparison.png', dpi= 300)

def word_count(text):
    '''Returns # of words in a string'''
    return len(text.split())

def phrase_counter(text, phrases):
    ''' Counts number of phrases in a list of phrases'''
    count = 0
    for _ in phrases:
        count+=text.count(_)
    return count

def cluster(data, n_clusters):
    '''assignes claims to clusters and orders cluster # by % rejected in each cluster'''

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(list(data['claim_vec'].values))
    data['cluster'] = kmeans.labels_

    df2 = data.groupby('cluster').sum().reset_index()
    df2['percent_rejected']=df2['rejected']/(df2['rejected']+df2['allowed'])
    cluster_indx = np.argsort(df2['percent_rejected'])
    sorting_dict ={k:v for v,k in zip(cluster_indx.index, cluster_indx)}
    data['cluster'] = data.apply(lambda row: sorting_dict[row['cluster']],axis=1)
    return data

def calc_claim_attributes(data):
    ''' Calculate claim attributes for analysis on clusters'''

    narrowing_phrases = ['corresponding','each', 'responsive to', 'containing',
                        'only', 'using the', 'a first', 'a second', 'consisting'
                        , 'more than', 'less than', 'between', 'the']

    vague_phrases = ['apply it', 'substantially', 'approximately', 'appreciablly',
                 'appriciably ', 'configured']

    data['word_count']=data.apply(lambda row: word_count(row['claim']),axis=1)
    data['narrowing_count']=data.apply(lambda row: phrase_counter(row['claim'], narrowing_phrases),axis=1)
    data['vague_count']=data.apply(lambda row: phrase_counter(row['claim'], vague_phrases),axis=1)
    data['a_count']=data.apply(lambda row: phrase_counter(row['claim'], [' a ']),axis=1)
    return data

def cluster_eda_plots(data, save=False, n_clusters=30):
    '''This loong no so pythonic block of code prints out a lot of the cluster
    analysis graphs'''

    data = cluster(data, n_clusters) #cluster claims using vectorized features
    data = calc_claim_attributes(data) #add columns for claim attributes

    df2 = data.groupby('cluster').agg({'allowed':'sum', 'rejected':'sum','word_count':'mean',
                                        'narrowing_count':'mean', 'vague_count':'mean',
                                        'vague_count':'mean', 'a_count':'mean'})
    df2 = df2.reset_index()
    df2['percent_rejected']=df2['rejected']/(df2['rejected']+df2['allowed'])

    #FIG 1
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.35       # the width of the bars
    ax1.bar(df2['cluster']-width/2, df2['rejected'], width, color='red', alpha=.8)
    ax1.bar(df2['cluster']+width/2, df2['allowed'], width, color='royalblue', alpha =.8)
    ax1.legend(['Rejected','Allowed'])
    ax1.set_ylabel('Number of Events')
    ax1.set_xlabel('Cluster')
    plt.tight_layout
    plt.savefig('images/{}_clusters_count.png'.format(n_clusters), bbox_inches='tight'
                                                      , dpi =300)

    #FIG 2
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.bar(df2['cluster'], df2['percent_rejected'], color='k', alpha =.8)
    ax2.set_ylabel('Percent Rejected')
    ax2.set_xlabel('Cluster')
    ax2.set_title('Percent of Claims rejected in each Cluster')
    plt.tight_layout
    plt.savefig('images/{}_clusters_percent.png'.format(n_clusters),bbox_inches='tight', dpi =300)

    #FIG 3
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    ax3.bar(df2['cluster'], df2['word_count'], color='k', alpha =.8)
    ax3.set_ylabel('Average word count')
    ax3.set_xlabel('Cluster')
    ax3.set_title('Word Count by Cluster')
    plt.tight_layout
    plt.savefig('images/{}_clusters_word_count.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)

    #FIG 4
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5))
    ax4.bar(df2['cluster'], df2['narrowing_count'], color='k', alpha =.8)
    ax4.set_ylabel('Avg count of Narrowing Phrases')
    ax4.set_xlabel('Cluster')
    ax4.set_title('Count of Narrowing Phrases by Cluster')
    plt.tight_layout
    plt.savefig('images/{}_clusters_narrowing_count.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)

    #FIG 5
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 5))
    ax5.bar(df2['cluster'], df2['a_count'], color='k', alpha =.8)
    ax5.set_ylabel('Average Number of Features')
    ax5.set_xlabel('Cluster')
    ax5.set_title('Count of Claimed Features by Cluster')
    plt.tight_layout
    plt.savefig('images/{}_clusters_a_count.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)

    #FIG 7-9 showing classes in clusters
    num_groups = 9
    temp = data.groupby(['uspc_class']).count()['claim'].reset_index()
    temp.columns = ['uspc_class', 'count']
    indacies = np.argsort(-temp['count'])[:10]
    top_groups = list(temp['uspc_class'][indacies])
    top_groups

    classes = pd.read_csv('data/uspc_classes.csv', delimiter=',',header=None)
    classes.columns=['uspc','class_descricption']
    converter = {k:v for k,v in zip(classes['uspc'],classes['class_descricption'])}
    converter['other']='Other'

    def replace_col(text, top_groups):
        if text not in top_groups:
            return 'other'
        else:
            return converter[str(text)]

    fig7, ax7 = plt.subplots(1, 1, figsize=(10, 5))
    ax7.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    temp = data.copy()
    temp['uspc_class'] = temp.apply(lambda row: replace_col(row['uspc_class'], top_groups), axis=1)
    df3 = temp.groupby(['cluster','uspc_class']).agg({'claim':'count'}).unstack()
    df3.columns.set_levels(['category'],level=0,inplace=True)
    df3.plot(kind='bar',stacked=True, ax=ax7, colormap='tab20b')
    ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('images/{}_clusters_and_USPC_class.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)

    fig8, ax8 = plt.subplots(1, 1, figsize=(10, 5))
    ax8.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    temp = data.copy()
    temp = temp[temp['allowed']==1]
    temp['uspc_class'] = temp.apply(lambda row: replace_col(row['uspc_class'], top_groups), axis=1)
    df3 = temp.groupby(['cluster','uspc_class']).agg({'claim':'count'}).unstack()
    df3.columns.set_levels(['category'],level=0,inplace=True)
    df3.plot(kind='bar',stacked=True, ax=ax8, colormap='tab20b')
    ax8.legend(['Artificial Intelligence', 'Database and File Management / Data Structures',
                'Generic Control Systems or Specific Applications',
                'Presentation Processing / Operator Interface Processing',
                'Software Development, Installation, and Management', 'Structural Design, Modeling, Simulation, and Emulation',
                'Memory','Support','Virtual mMchine Task or Process Management/Control',
                 'Error Detection/Correction and fault detection/recovery',
                 'Other'], loc='center left', bbox_to_anchor=(.42, 0.68), fancybox=True,framealpha=0.7)
    ax8.set_ylim([0,1650])
    ax8.set_title('Allowed Claims by Cluster and Category')
    plt.savefig('images/{}_clusters_and_USPC_class_allowed.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)


    fig9, ax9 = plt.subplots(1, 1, figsize=(10, 5))
    ax9.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    temp = data.copy()
    temp = temp[temp['allowed']==0]
    temp['uspc_class'] = temp.apply(lambda row: replace_col(row['uspc_class'], top_groups), axis=1)
    df3 = temp.groupby(['cluster','uspc_class']).agg({'claim':'count'}).unstack()
    df3.columns.set_levels(['category'],level=0,inplace=True)
    df3.plot(kind='bar',stacked=True, ax=ax9, colormap='tab20b')
    ax9.legend(['Artificial Intelligence', 'Database and File Management / Data Structures',
                'Generic Control Systems or Specific Applications',
                'Presentation Processing / Operator Interface Processing',
                'Software Development, Installation, and Management', 'Structural Design, Modeling, Simulation, and Emulation',
                'Memory','Support','Virtual mMchine Task or Process Management/Control',
                 'Error Detection/Correction and fault detection/recovery',
                 'Other'], loc='center left', bbox_to_anchor=(0, 0.68), framealpha=0.7, fancybox=True)
    ax9.set_ylim([0,1650])
    ax9.set_title('Rejected Claims by Cluster and Category')
    plt.savefig('images/{}_clusters_and_USPC_class_rejected.png'.format(n_clusters) ,bbox_inches='tight', dpi =300)
