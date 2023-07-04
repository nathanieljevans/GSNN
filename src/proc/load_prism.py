import pandas as pd 
import numpy as np 
from src.proc import utils

def load_prism(path): 

    dep2iname = utils.get_dep2iname(path).set_index('DepMap_ID').to_dict()['cell_iname']

    def _load_prism(path, screen_id, dep2iname): 

        #prism_primary = pd.read_csv(f'{path}/primary-screen-replicate-collapsed-logfold-change.csv')
        p = pd.read_csv(path)
        p = p.rename({'Unnamed: 0':'depmap_id'}, axis=1)
        p = p.assign(cell_iname = lambda x: [dep2iname[xx] if xx in dep2iname else None for xx in x.depmap_id])
        p = p.set_index(['depmap_id', 'cell_iname']).stack().reset_index().rename({'level_2':'meta', 0:'log_fold_change'}, axis=1)
        p[['pert_id_long', 'pert_dose','_']] = p.meta.str.split(pat='::', n=2, expand=True)
        p = p.assign(pert_id = lambda x: [xx[:13] for xx in x.pert_id_long])
        p = p.assign(screen_id = screen_id)
        return p 

    p1 = _load_prism(path       = f'{path}/primary-screen-replicate-collapsed-logfold-change.csv', 
                     screen_id  = 'primary', 
                     dep2iname  = dep2iname)

    p2 = _load_prism(path       = f'{path}/secondary-screen-replicate-collapsed-logfold-change.csv', 
                     screen_id  = 'secondary', 
                     dep2iname  = dep2iname)

    prism = pd.concat((p1, p2), axis=0) 

    # average replicates 
    prism = prism.groupby(['pert_id', 'depmap_id', 'cell_iname', 'pert_dose']).agg({'log_fold_change':np.mean, 'screen_id':list}).reset_index()
    prism = prism.assign(num_repl = lambda x: [len(xx) for xx in x.screen_id])

    # create aggregate id 
    prism['sig_id'] = prism[['cell_iname', 'pert_id', 'pert_dose']].agg('::'.join, axis=1)

    # rename `pert_dose` to match behavior of lincs 
    prism = prism.rename({'pert_dose':'conc_um'}, axis=1)

    # add cell viability transformation
    # Calculate viability data as two to the power of replicate-level logfold change data
    prism = prism.assign(cell_viab = lambda x: 2**(x.log_fold_change))

    return prism