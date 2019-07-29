from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np       
import seaborn as sns
                             
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






