I have been looking at different methods of categorical encoding:
One hot encoding
Target Encoding - encode the category using a Laplace smoothed mean target value (CTR) using parameter `cat_pseudo_count`
Node level Encoding - reencode categories _at each node_ using a Laplace smoothed mean objective value using parameter `cat_reg_lambda`
Tree level Encoding - reencode categories _only at each tree root_ using a Laplace smoothed mean objective value using parameter `cat_reg_lambda`

# Rare categories cutoff (and memory issues)

The comparison is complicated by the different memory requirements of the different methods.
To allow some measure of comparability, we start with a rare category cutoff (`cat_cutoff`).  Any categories with fewer than this level of counts gets merged into a "rare" category. On the one hand, we can view this cutoff as generating a new benchmark dataset and compare the methodologies on the new dataset.  On the other hand, since performance improves as we lower the cutoff, methods that allow us to drop the cat_cutoff on a given hardware have performance advantages. (ie ultimately we should test on the smallest cutoff each method supports with the given hardware)

There is some evidence that overfitting occurs at lower values of `cat_cutoff` (below 100 for `tree` and below 200 for `node`).  However, we tended to hit memory issues, so it has not been fully explored.


<img width="946" height="847" alt="cat_encoding_methods_cutoff" src="https://github.com/user-attachments/assets/4015a444-360f-454a-8fa7-60e8df0d480d" />

## Node level encoding and need for high `reg_lambda`

The very bad validation log loss (above 0.466) of Node level categorical encoding is with `reg_lambda` = 0.  We found that introducing `reg_lambda` was essential to drop the error for this encoding. Note that `cat_cutoff` acts as a regularising influence in this case (ie reducing the number of categories improves validation error, presumably by reducing overfitting).  The improved performance of node encoding (around 0.45) was achieved with high `reg_lambda` of _25,000_ or more. Given that in most of our experiments a value of  `reg_lambda=0` was better, it suggests the need for different `reg_lambda` for different values

## `Cat_cutoff ` from 100 to 1000 - (Tree 1st 0.4420 Node 2nd 0.4485 OHE 3rd 0.4509 at `cat_cutoff=100`)

Over this range, we were able to build trees with all these models. As can be seen tree level encoding was best followed by node, followed by one hot encoding. Target encoding has few results plotted, because most of our target encoding experiments were performed with downsampled data.

## Target encoding results on downsampled data

In the chart below we plot validation log loss against the `sample_every` for different cat_cutoffs on a log-log scale. As can be seen, the error drops as both sample_every and cat_cutoff are reduced. Extrapolating 
<img width="933" height="864" alt="target_encoding_sample_every_cat_cutoff" src="https://github.com/user-attachments/assets/92dda4b2-7fd3-41d7-a805-ef8a8a69a586" /> Extrapolating from the graph, we might expect target encoding to outperform Node and OHE at approx 0.446 at a cat_cutoff of 100 and no downsampling (`sample_every=1`).

## Config settings
| variable                       | vten3efq                     | c6b0b3nz                     | 3a081avp            |
|--------------------------------|------------------------------|------------------------------|---------------------|
| config.cat_order_fix           | tree                         | node                         | ohe                 |
| config.cat_cutoff              | 100                          | 100                          | 100                 |
| project                        | criteo_boost_gpu_categorical | criteo_boost_gpu_categorical | criteo_boost_ohe    |
| config.sample_every            | 1                            | 1                            | 1                   |
| config.subsample               | 1.0                          | 1.0                          | 1.0                 |
| config.n_estimators_est        | 600                          | 300                          | 200                 |
| config.max_bin                 | 256                          | 256                          | 256                 |
| config.max_depth               | 60                           | 60                           | 60                  |
| config.min_split_loss          | 10.0                         | 10.0                         | 10.0                |
| config.reg_lambda              | 10                           | 38400                        | 0                   |
| config.learning_rate           | 0.0500000007                 | 0.075000003                  | 0.200000003         |
| config.min_child_weight        | 1.0                          | 10.0                         | 1.0                 |
| config.cat_reg_lambda          | 100                          | 25                           | null                |
| config.cat_pseudo_count        | 16                           | 16                           | 16                  |
| config.max_cat_threshold       | 1000000                      | 1000000                      | 64                  |
| config.max_cat_to_onehot       | 10                           | 10                           | 4294967295          |
| summary_metrics.best_iteration | 294                          | 251                          | 199                 |
| summary_metrics.epoch          | 304                          | 261                          | 199                 |
| validation log loss            | 0.44198237045074346          | 0.44852812342223497          | 0.45092605074493886 |
