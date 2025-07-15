
07/08/2025 

The results of 07_UQ_with_HyperNetworks is interesting, after playing with it a bit, the GSNN tends to do much better than the NN with calibration,overfitting, and performance when working on noisy, small datasets. This aligns with the premise that we if we know the structure, we don't need as much data to learn the function. Should keep this in mind for what problems the GSNN is most applicable for, and perhaps more importantly, how can we improve the model so that it still performs comparably to NN in low noise, large datasetes - because here the NN starts to look  better. I would think that the GSNN should be robust to large datasets, but it is under confident in its predictions, and I'm not sure why, perhaps a information throttling issue due to the latent edges being smaller than the channel edges? 


07/15/25 

The results of the `04_reinforce.ipynb` example suggest that small local changes in the graph structure (additional edges; not missing edges) lead to comparable performance, or even slightly better performance, which has made it challenging to use performance to optimize the graph structure. It could be that the additional edges offer more flexible learning, analagous to having more channels. Or it might be that there are some other limitations of the GSNN that is preventing effective learning - bad weight initialization, gradient vanishing, etc - which is somehow less impacted in the other network structures. 

The other challenge of using performance to optimize network structure beyond just local structure (a few additional false edges) is there are many largely equivalent graphs that would appear "incorrect", e.g., 

i0 -> A -> o0 ; i1 -> B -> o1 
and 
i0 -> B -> o0 ; i1 -> A -> o1  

So IF we were to do full network optimization, we would need to have a measure of interactions rather than just edge labels. For instance, a better metric would be ... is o0 a descendent of i0 ? In other words, we define the "name" or role of a latent node by it's structure, so without any structure, the nodes become unlabeled until some structure is optimized or reintroduced, at which point we would need to try to define the node labels. This of course assumes that the latent nodes don't have observable outcomes, if they do, then the label becomes far more fixed. This fact could suggest that structure optimization will work much better if there are many observables, spread densely through the network (and that they are dependent on each other or locally related). 