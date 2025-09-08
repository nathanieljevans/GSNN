
07/08/2025 

The results of 07_UQ_with_HyperNetworks is interesting, after playing with it a bit, the GSNN tends to do much better than the NN with calibration,overfitting, and performance when working on noisy, small datasets. This aligns with the premise that we if we know the structure, we don't need as much data to learn the function. Should keep this in mind for what problems the GSNN is most applicable for, and perhaps more importantly, how can we improve the model so that it still performs comparably to NN in low noise, large datasetes - because here the NN starts to look  better. I would think that the GSNN should be robust to large datasets, but it is under confident in its predictions, and I'm not sure why, perhaps a information throttling issue due to the latent edges being smaller than the channel edges? 


07/15/25 

The results of the `04_reinforce.ipynb` example suggest that small local changes in the graph structure (additional edges; not missing edges) lead to comparable performance, or even slightly better performance, which has made it challenging to use performance to optimize the graph structure. It could be that the additional edges offer more flexible learning, analagous to having more channels. Or it might be that there are some other limitations of the GSNN that is preventing effective learning - bad weight initialization, gradient vanishing, etc - which is somehow less impacted in the other network structures. 

The other challenge of using performance to optimize network structure beyond just local structure (a few additional false edges) is there are many largely equivalent graphs that would appear "incorrect", e.g., 

i0 -> A -> o0 ; i1 -> B -> o1 
and 
i0 -> B -> o0 ; i1 -> A -> o1  

So IF we were to do full network optimization, we would need to have a measure of interactions rather than just edge labels. For instance, a better metric would be ... is o0 a descendent of i0 ? In other words, we define the "name" or role of a latent node by it's structure, so without any structure, the nodes become unlabeled until some structure is optimized or reintroduced, at which point we would need to try to define the node labels. This of course assumes that the latent nodes don't have observable outcomes, if they do, then the label becomes far more fixed. This fact could suggest that structure optimization will work much better if there are many observables, spread densely through the network (and that they are dependent on each other or locally related). 



06/06/25 

## method for including signed edges into latent structure. 

Given a signed graph (or partially signed graph), we can incorporate the sign significance either as a secondary objective during training, or as a pre-training step. 

The general idea is that for any edge ij, at every layer of the model we have $z_i^l$ and $z_j^l$, and given a sign +/-, we could adopt the premise that positive interactions should have similar latent embeddings while negative relations should have very different latent embeddings. 

f(z_i, z_j) = sign(e_ij) 

where f is some distance metric, like cosine similarity. 

It may need some normalization to account for when there are many edges into/out-of and node. Although if we used a magnitude based metric like dot product or euclidean then we wouldn't have to worry about it. 

Alternative verrsion of this... 
a_i = f_nn(z_i)
a_j = f_nn(z_j)
minimize dist(a_i, a_j) if sign(e_ij) is positive 
maximize dist(a_i, a_j) if sign(e_ij) is negative
where a is just a scalar. 

Imagining this works perfectly and we have similar latent embeddings when a positive interaction nodes are active or different embeddings when a negative interaction is active, then how would this impact the behavior of the model?

nodes driven by positive interactions would have a different latent embedding than nodes driven by negative interactions. That said, it could potentially still result in similar predictions, since the nonlinear nature of the nns get tricky. 

See `10_signed_networks.ipynb` for some implementation trials. 

I think the concept of using correlations between node attentions makes the most sense, particularly for biological networks. However, I'm struggling to find the right objective to use, because we can't expect perfect correlation between attention weights since there are likely to be many inputs to nodes with conflicting signs. 

The general idea though is that, given: 

   +            -
A -> B        C -> D 

when att_A is large, then it's likely that att_B is large (although not necessarily if there are other inputs to B). Meanwhile if att_C is large, then att_D is likely small (although again, could depend on other inputs to D). Additionally if att_A is 0, then it doesn't necessarily have any effect on att_B, especially if there are other inputs. 

So it almost seems like we need some weighted objective like: 

i -sign-> j

L = att_i*(sign*abs(att_i - att_j)) 

06/06/25 

TODO: need to add documentation for graph equivalence testing. Very interesting and important element to the ability to do latent modeling based on structure. 



07/18/25


Concept for inferring input-function, function-function and function-output edges. 

Combining the ideas from inputEdgeInferrer and outputEdgeInferer, we can generalize the methods to any edge by creating a scoring mechanishm such that we can infer how well a given edge latent state ($h_ij$), can predict or correlate with another edge (\grad h_lm). For instance, strong correlation between: 

$h_ij ~ \grad h_lm $ is suggestive that the model would benefit from an edge from nodes i (or j) to m. 

So theoretically, we could infer this with a weight matrix trained on the gradients, e.g., 

W_ij,lm * h_ij ~ \grad h_lm 

or a scoring algorithm, such as: 

score_{ij, lm} ~ abs( corr(h_ij, \grad h_lm) )

Unfortunately, this task scales quadratically with the number of edges in the model. 

# tests ~ E**2 

For typical tasks, we often have on the order of 100,000 edges, which leads to 1e10 possible edges to test. This is unlikely to be tractable for two reasons: 

- memory requirements for 1e10 parameters is going to be very expensive 
- testing each edge group over the full training dataset would be extremely time consuming. 

One thought to address this is to use the work from "How to turn your knowledge graph into a generative model" 

For instance, if we parameterize a low-rank adjacency matrix with node-node decompositions, e.g,. A_ij = z_i * z_j. We can avoid large memory constraints. Additionally, we can use the generative model component to explore likely regions and avoid testing edges that are unlikely. 

Procedure: 

- initialize generative adjacency matrix g
- for batch in data: 
   - sample N edges from g 
   - score each edge 
   - update g to reflect the edge scores (high scores become more probable, low scores become less probable)

- evaluation: compute the probabilities for each edge in the A and select the top most likely edges 