import torch 
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import seaborn as sbn
from sklearn.metrics import r2_score
from adjustText import adjust_text
import os 
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from scipy import stats
from matplotlib.text import Text

import sys 
sys.path.append('../')
import gsnn.models.utils as utils



def grouped_performance_rowwise(groups_list, model1_preds, model2_preds, metric='pearson', test='paired_ttest', min_members_per_group=10, min_num_replicates_per_drug=2, pval_adj_method='fdr_bh'): 
    '''
    Computes the pearson correlation performance of observations within groups specified by `group_variable`.
    NOTE: rowize groups can be drug, target, cell line, etc (grouping by obs sets)

    Args: 
        groups                  list<sets>               list of groups for each model predictions, e.g., groups[0] should be used with model1/2_preds[0]. Assumes that model1/2_preds are ordered such that they use the same test partitions.          
        gsnn_preds              list<dict>               list of dictionaries containing the gsnn predictions for several models
        nn_preds                list<dict>               list of dictionaries containing the nn predictions for several models 
        axis                    int                      which axis to group: 0 is grouped by row or obs, 1 is grouped by column or gene 
        min_members_per_group       int                  min number of observation to compute group performance
        pval_adj_method         str                      method of multiple test correction of p-values, see [multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html) for details.

    Returns: 
        res                     pandas DataFrame        the grouped results 
    '''
    failed = 0

    print('computing row-wise groups (e.g., obs groupings)')

    all_groups = set() 
    for groups in groups_list: all_groups = all_groups.union( set(list(groups.keys())) )

    perf_res1 = {g:[] for g in all_groups}
    perf_res2 = {g:[] for g in all_groups}
    diff_res = {g:[] for g in all_groups}
    for i, (groups, model1_preds, model2_preds) in enumerate(zip(groups_list, model1_preds, model2_preds)): 
        for j, group in enumerate(groups.keys()): 
            print(f'progress: {j}/{len(groups)} [replicate: {i}]', end='\r')
            idx = groups[group]

            if (len(idx) >= min_members_per_group): 
                if metric != 'mse': 
                    r1 = utils.corr_score(model1_preds['y'][idx], model1_preds['yhat'][idx], method=metric)
                    r2 = utils.corr_score(model2_preds['y'][idx], model2_preds['yhat'][idx], method=metric)
                else: 
                    r1 = np.mean((model1_preds['y'][idx] - model1_preds['yhat'][idx])**2)
                    r2 = np.mean((model2_preds['y'][idx] - model2_preds['yhat'][idx])**2)
                perf_res1[group].append( r1 )
                perf_res2[group].append( r2 )
                diff_res[group].append(r1 - r2)


    res = {'group':[], 'model1_mean_perf':[], 'model2_mean_perf':[], 'model1_std_perf':[], 'model2_std_perf':[], 
        'pval':[], 'tstat':[], 'mean_diff_perf':[], 'std_diff_perf':[], 'ci_diff_perf':[], 
        'N_members':[], 'metric':[]}
    for group in all_groups: 

        model1_perfs = np.array(perf_res1[group])
        model2_perfs = np.array(perf_res2[group])
        perf_diffs = diff_res[group]

        if len(model1_perfs) >= min_num_replicates_per_drug: 
            
            if test == 'wilcoxon': 
                test_res = wilcoxon(model1_perfs, model2_perfs, alternative='two-sided', method='exact')
            elif test == 'paired_ttest': 
                test_res = ttest_rel(model1_perfs, model2_perfs, alternative='two-sided')
            else: 
                raise ValueError('unrecognized test method')
            
            res['group'].append(        group)
            res['model1_mean_perf'].append(  np.mean(model1_perfs))
            res['model2_mean_perf'].append(    np.mean(model2_perfs))
            res['model1_std_perf'].append(   np.std(model1_perfs))
            res['model2_std_perf'].append(     np.std(model2_perfs))
            res['mean_diff_perf'].append(  np.mean(perf_diffs) )
            res['std_diff_perf'].append(   np.std(perf_diffs) )
            res['ci_diff_perf'].append(    np.quantile(perf_diffs, q=[0.025, 0.975]) )
            res['pval'].append(         test_res.pvalue)
            res['tstat'].append(        test_res.statistic)
            res['N_members'].append(    len(idx))
            res['metric'].append( metric )
        else: 
            failed += 1
        

    print() 
    print('# failed groups (too few obs per group):', failed)

    res = pd.DataFrame(res)

    # p-value adjustment 
    res = res.assign(pval_adj = multipletests(res.pval.values, method=pval_adj_method)[1])

    return res


def model_comp_performance_by_gene(model1_preds, model2_preds, gene_names, pval_adj_method='fdr_bh'): 
    '''

    Args: 
        gsnn_preds              list<dict>               list of dictionaries containing the gsnn predictions for several models
        nn_preds                list<dict>               list of dictionaries containing the nn predictions for several models 
        axis                    int                      which axis to group: 0 is grouped by row or obs, 1 is grouped by column or gene 
        min_members_per_group       int                  min number of observation to compute group performance
        pval_adj_method         str                      method of multiple test correction of p-values, see [multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html) for details.

    Returns: 
        res                     pandas DataFrame        the grouped results 
    '''
    
    failed = 0
    res = {'group':[], 'model1_mean_perf':[], 
           'model2_mean_perf':[], 'model1_std_perf':[], 
           'model2_std_perf':[], 'pval':[], 
           'tstat':[], 'mean_diff_perf':[], 
           'std_diff_perf':[], 'ci_diff_perf':[]}

    print('computing gene-wise performances')
    # assume computing performance by gene 

    for i,gene in enumerate(gene_names): 

        print(f'progress: {i}/{len(gene_names)}', end='\r')

        r1s = []; r2s = []; diffs=[]
        for preds1, preds2 in zip(model1_preds, model2_preds): 

            assert len(gene_names) == preds1['y'].shape[1], 'length of gene names does not match prediction shape'
            
            r1 = np.corrcoef(preds1['y'][:, i], preds1['yhat'][:, i])[0,1]
            r2 = np.corrcoef(preds2['y'][:, i], preds2['yhat'][:, i])[0,1]
            r1s.append( r1 )
            r2s.append( r2 )
            diffs.append( r1-r2 )

        _ttest = ttest_rel(r1s, r2s, alternative='two-sided')
        
        res['group'].append(        gene)
        res['model1_mean_perf'].append(  np.mean(r1s))
        res['model2_mean_perf'].append(    np.mean(r2s))
        res['model1_std_perf'].append(   np.std(r1s))
        res['model2_std_perf'].append(     np.std(r2s))
        res['mean_diff_perf'].append(  np.mean(diffs) )
        res['std_diff_perf'].append(   np.std(diffs) )
        res['ci_diff_perf'].append(    np.quantile(diffs, q=[0.025, 0.975]) )
        res['pval'].append(         _ttest.pvalue)
        res['tstat'].append(        _ttest.statistic)

    res = pd.DataFrame(res)

    # p-value adjustment 
    res = res.assign(pval_adj = multipletests(res.pval.values, method=pval_adj_method)[1])

    return res

def my_adjust_text(text_objects, ax, fig, x_start=0, y_start=0, magic_scalar=0.2):
    """
    Adjusts the position and orientation of text labels on a scatterplot for non-overlapping and
    equally spaced labels, with correctly adjusted leader lines.

    Parameters:
    text_objects (list): List of Text objects (matplotlib.text.Text).
    ax (matplotlib.axes.Axes): The axes object of the plot.
    fig (matplotlib.figure.Figure): The figure object of the plot.
    """

    # Get the limits of the axes for alignment
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Separate the text objects based on their quadrant
    right_side_texts = []
    top_side_texts = []

    for text in text_objects:
        x, y = text.get_position()
        if x >= y:
            right_side_texts.append((text, (x, y)))
        else:
            top_side_texts.append((text, (x, y)))

    # Sort the texts based on their position
    right_side_texts.sort(key=lambda text: text[1][1])
    top_side_texts.sort(key=lambda text: text[1][0])

    # Adjust text positions for non-overlapping
    def adjust_positions(texts, axis):
        total_space = ylim[1] if axis == 'y' else xlim[1]
        n = len(texts)
        spacing = (total_space * 0.6) / (n + 1)
        for i, (text, orig_pos) in enumerate(texts):
            if axis == 'y':
                pos = (i + 1) * spacing + y_start
                text.set_position((xlim[1], pos))
                text.set_rotation(0)
                text.set_verticalalignment('center')
                text.set_horizontalalignment('right')
            else:
                pos = (i + 1) * spacing + x_start
                text.set_position((pos, ylim[1]))
                text.set_rotation(90)
                text.set_verticalalignment('top')
                text.set_horizontalalignment('center')

    adjust_positions(right_side_texts, 'y')
    adjust_positions(top_side_texts, 'x')

    # Draw leader lines with adjustment for text length
    for text, orig_pos in right_side_texts + top_side_texts:
        orig_x, orig_y = orig_pos
        new_x, new_y = text.get_position()
        text_width, text_height = text.get_window_extent(renderer=fig.canvas.get_renderer()).width, text.get_window_extent(renderer=fig.canvas.get_renderer()).height
        text_width /= fig.dpi
        text_height /= fig.dpi

        if text in [t for t, _ in right_side_texts]:
            ax.plot([orig_x, new_x - text_width*magic_scalar], [orig_y, new_y], color='k', lw=0.5, alpha=0.5)
        else:
            ax.plot([orig_x, new_x], [orig_y, new_y - text_height*magic_scalar], color='k', lw=0.5, alpha=0.5)


def plot_model2model_comparison(res, feature_label, num_points_to_label=10, num_points_to_label2=10, alpha=0.1, figsize=(7,7), adjust_labels=True, 
                                fontsize=12, model1_name='GSNN', model2_name='NN', size=None, markersize=50, leaderline_magic_scalar=0.2): 

    res = res.assign(significant = res.pval_adj < alpha)
    res = res.rename({'pval_adj': 'p-value (adj.)'}, axis=1)

    # Normalization parameters
    vmin, vmax = 0, 1
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Map the 'p-value (adj.)' to colors
    res['colors'] = [cmap(norm(value)) for value in res['p-value (adj.)']]

    # Create the scatter plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    g = sbn.scatterplot(x='model1_mean_perf', y='model2_mean_perf', 
                        size=size, hue='p-value (adj.)', palette='viridis', 
                        hue_norm=(0,1), 
                        data=res, c='k', marker='.', s=markersize, 
                        ax=ax, legend='brief', style='significant')

    # Remove the hue legend since it's now just colors
    ax.legend_.remove()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(res['p-value (adj.)'])
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label(label='p-value (adj.)', size=int(1.25*fontsize))

    _min = max(min(res.model1_mean_perf.values.tolist() + res.model2_mean_perf.values.tolist()), 0)
    _max = min(max(res.model1_mean_perf.values.tolist() + res.model2_mean_perf.values.tolist()), 1) + 0.2
    #_min =0; _max =1
    plt.plot((_min,_max), (_min, _max), 'k--', alpha=0.33)

    if num_points_to_label > 0: 
        rows_to_label = res.sort_values('tstat', ascending=True).reset_index().head(int(num_points_to_label))
        rows_to_label2 = res.sort_values('tstat', ascending=False).reset_index().head(int(num_points_to_label2))
        #rows_to_label = res.sort_values('mean_diff_r', ascending=True).reset_index().head(int(num_points_to_label))
        #rows_to_label2 = res.sort_values('mean_diff_r', ascending=False).reset_index().head(int(num_points_to_label2))
        annotations = []
        for i, row in rows_to_label.iterrows(): 
            annotations.append(ax.annotate(row[feature_label], (row.model1_mean_perf, row.model2_mean_perf), fontsize=fontsize))#, textcoords='offset points', arrowprops=dict(arrowstyle='-', color='r')))
        for i, row in rows_to_label2.iterrows(): 
            annotations.append(ax.annotate(row[feature_label], (row.model1_mean_perf, row.model2_mean_perf), fontsize=fontsize))

        plt.tight_layout() 
        if adjust_labels: my_adjust_text(annotations, ax, f, x_start=_min, y_start=_min, magic_scalar=leaderline_magic_scalar)
    
    ax.spines[['right', 'top']].set_visible(False)

    # for legend text
    #plt.setp(g.get_legend().get_texts(), fontsize=fontsize)  
    # for legend title
    #plt.setp(g.get_legend().get_title(), fontsize=int(2*fontsize)) 
    #plt.legend(loc='upper right', title='p-value (adj.)')

    ax.set_xlabel(f'{model1_name} Performance', fontsize=int(1.15*fontsize))
    ax.set_ylabel(f'{model2_name} Performance', fontsize=int(1.15*fontsize))
    plt.show()