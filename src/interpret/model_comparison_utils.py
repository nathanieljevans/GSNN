import torch 
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
import seaborn as sbn
from sklearn.metrics import r2_score
from adjustText import adjust_text
import os 
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy import stats


import sys 
sys.path.append('../')
import src.models.utils as utils



def grouped_performance(groups, gsnn_preds, nn_preds, axis=0, min_members_per_group=10, pval_adj_method='fdr_bh'): 
    '''
    Computes the pearson correlation performance of observations within groups specified by `group_variable`.

    Args: 
        group_dict              
        gsnn_preds              list<dict>               list of dictionaries containing the gsnn predictions for several models
        nn_preds                list<dict>               list of dictionaries containing the nn predictions for several models 
        axis                    int                      which axis to group: 0 is grouped by row or obs, 1 is grouped by column or gene 
        min_members_per_group       int                  min number of observation to compute group performance
        pval_adj_method         str                      method of multiple test correction of p-values, see [multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html) for details.

    Returns: 
        res                     pandas DataFrame        the grouped results 
    '''
    

    failed = 0
    res = {'group':[], 'gsnn_mean_r':[], 'nn_mean_r':[], 'gsnn_std_r':[], 'nn_std_r':[], 'pval':[], 'tstat':[], 'mean_diff_r':[], 'std_diff_r':[], 'ci_diff_r':[], 'N_members':[]}

    if axis == 1: 
        print('computing column-wise groups')
        # assume computing performance by gene 
        
        for i, group in enumerate(groups.keys()):
            print(f'progress: {i}/{len(groups)}', end='\r')
            idx = groups[group]

            if len(idx) < min_members_per_group: 
                failed += 1
                continue 

            gsnn_rs = []
            nn_rs = []
            r_diffs = []
            for gsnn_pred, nn_pred in zip(gsnn_preds, nn_preds): 
                gsnn_rs.append( utils.corr_score(gsnn_pred['y'][:, idx], gsnn_pred['yhat'][:, idx]) )
                nn_rs.append( utils.corr_score(nn_pred['y'][:, idx], nn_pred['yhat'][:, idx]) )
                r_diffs.append(gsnn_rs[-1] - nn_rs[-1])
            
            res['group'].append(        group)
            res['gsnn_mean_r'].append(  np.mean(gsnn_rs))
            res['nn_mean_r'].append(    np.mean(nn_rs))
            res['gsnn_std_r'].append(   np.std(gsnn_rs))
            res['nn_std_r'].append(     np.std(nn_rs))
            res['mean_diff_r'].append(  np.mean(r_diffs) )
            res['std_diff_r'].append(   np.std(r_diffs) )
            res['ci_diff_r'].append(    np.quantile(r_diffs, q=[0.025, 0.975]) )
            _ttest = ttest_rel(gsnn_rs, nn_rs, alternative='two-sided')
            res['pval'].append(         _ttest.pvalue)
            res['tstat'].append(        _ttest.statistic)
            res['N_members'].append(    len(idx))

    elif axis == 0: 
        print('computing row-wise groups')

        for i, group in enumerate(groups.keys()): 
            print(f'progress: {i}/{len(groups)}', end='\r')
            idx = groups[group]

            if len(idx) < min_members_per_group: 
                failed += 1
                continue 

            gsnn_rs = []
            nn_rs = []
            r_diffs = []
            for gsnn_pred, nn_pred in zip(gsnn_preds, nn_preds): 
                gsnn_rs.append( utils.corr_score(gsnn_pred['y'][idx], gsnn_pred['yhat'][idx]) )
                nn_rs.append( utils.corr_score(nn_pred['y'][idx], nn_pred['yhat'][idx]) )
                r_diffs.append(gsnn_rs[-1] - nn_rs[-1])
            
            res['group'].append(        group)
            res['gsnn_mean_r'].append(  np.mean(gsnn_rs))
            res['nn_mean_r'].append(    np.mean(nn_rs))
            res['gsnn_std_r'].append(   np.std(gsnn_rs))
            res['nn_std_r'].append(     np.std(nn_rs))
            res['mean_diff_r'].append(  np.mean(r_diffs) )
            res['std_diff_r'].append(   np.std(r_diffs) )
            res['ci_diff_r'].append(    np.quantile(r_diffs, q=[0.025, 0.975]) )
            _ttest = ttest_rel(gsnn_rs, nn_rs, alternative='two-sided')
            res['pval'].append(         _ttest.pvalue)
            res['tstat'].append(        _ttest.statistic)
            res['N_members'].append(    len(idx))

    print() 
    print('# failed groups (too few obs per group):', failed)

    res = pd.DataFrame(res)

    # p-value adjustment 
    res = res.assign(pval_adj = multipletests(res.pval.values, method=pval_adj_method)[1])

    # difference in performance 
    #res = res.assign(diff_r = lambda x: x.gsnn_mean_r - x.nn_mean_r)
    
    return res

def plot_model2model_comparison(res, feature_label, num_points_to_label=20, alpha=0.05, figsize=(7,7), adjust_labels=True, 
                                fontsize=15, model1_name='GSNN', model2_name='NN', size=None, markersize=75): 

    res = res.assign(significant = res.pval_adj <= alpha)
    res = res.rename({'pval_adj': 'p-value (adj.)'}, axis=1)

    f,ax = plt.subplots(1,1, figsize=figsize)
    g = sbn.scatterplot(x='gsnn_mean_r', y='nn_mean_r', hue='p-value (adj.)', size=size, style='significant', data=res, c='k', marker='.', s=markersize, palette=sbn.color_palette("flare", as_cmap=True), ax=ax)

    _min = min(res.gsnn_mean_r.values.tolist() + res.nn_mean_r.values.tolist())
    _max = max(res.gsnn_mean_r.values.tolist() + res.nn_mean_r.values.tolist())
    plt.plot((_min,_max), (_min, _max), 'k--', alpha=0.25)

    if num_points_to_label > 0: 
        rows_to_label = res.sort_values('pval').reset_index().head(num_points_to_label)
        annotations = []
        for i, row in rows_to_label.iterrows(): 
            annotations.append(ax.annotate(row[feature_label], (row.gsnn_mean_r, row.nn_mean_r), fontsize=fontsize))#, textcoords='offset points', arrowprops=dict(arrowstyle='-', color='r')))

        plt.tight_layout() 
        if adjust_labels: 
            adjust_text(annotations, 
                        x=res.gsnn_mean_r.values.tolist(), 
                        #x=np.linspace(0,1,_npts), 
                        y=res.nn_mean_r.values.tolist(), 
                        #y=np.linspace(0,1,_npts) + 0.1*np.random.randn(_npts), 
                        arrowprops=dict(arrowstyle='-', color='red'), 
                        avoid_self=True, 
                        lim=1000)
    
    ax.spines[['right', 'top']].set_visible(False)

    # for legend text
    plt.setp(g.get_legend().get_texts(), fontsize=fontsize)  
    
    # for legend title
    plt.setp(g.get_legend().get_title(), fontsize=int(2*fontsize)) 

    ax.set_xlabel(f'{model1_name} Pearson Corr.', fontsize=int(1.25*fontsize))
    ax.set_ylabel(f'{model2_name} Pearson Corr.', fontsize=int(1.25*fontsize))
    plt.show()