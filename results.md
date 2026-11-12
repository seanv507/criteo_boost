# Criteo Boost results

Criteo_x4
https://github.com/reczoo/RecData/tree/main/Criteo/Criteo_x4
we *randomly* split the data into 8:1:1 as the training set, validation set, and test set, respectively using *stratified sampling* based on target variable.

(ie ignoring timeseries variation)

train 36,672,493 25.6% CTR
validation 4,584,062
4,584,062



Criteo X1 45,840,617
train 33,003,326 	valid 8,250,124 	test 4,587,167



splitting by wandb project


| project                      | validation log loss|
| ---                          | ---                               |
| str                          | f64                               |
|------------------------------|-----------------------------------|
| criteo_boost                 | null                              |
| criteo_boost_sample_cutoff   | null                              |
| criteo_boost_gpu_categorical | 0.442665                          |
| criteo_boost_sample_01       | 0.443078                          |
| criteo_boost_gpu             | 0.444234                          |
| criteo_boost_sweep           | 0.444352                          |
| criteo_boost_cat_order_tree  | 0.444899                          |
| criteo_boost_ohe             | 0.445224                          |
| criteo_boost_enum            | 0.465781                          |

criteo boost categorical has the lowest validation error



the tree building has memory issues depending on the cardinality of the categoricals


shape: (5, 4)
| config.cat_cutoff | config.learner.gradient_booste... | len | summary_metrics.validation_1-l... |
| ---               | ---                               | --- | ---                               |
| i64               | str                               | u32 | f64                               |
|-------------------|-----------------------------------|-----|-----------------------------------|
| 100               | tree                              | 2   | 0.442665                          |
| 200               | tree                              | 8   | 0.443098                          |
| 200               | node                              | 2   | 0.448782                          |
| 500               | tree                              | 4   | 0.444049                          |
| 500               | node                              | 27  | 0.449207                          |


As we see reducing the cat_cutoff improves performance.  Tree seems to perform better than Node

## Figures

### cat_cutoff: 100 cat_order: tree

![cat_cutoff: 100 cat_order: tree](figures/cat_cutoff%3A%20100%20cat_order%3A%20tree.svg)

### cat_cutoff: 200 cat_order: tree

![cat_cutoff: 200 cat_order: tree](figures/cat_cutoff%3A%20200%20cat_order%3A%20tree.svg)

### cat_cutoff: 200 cat_order: node

![cat_cutoff: 200 cat_order: node](figures/cat_cutoff%3A%20200%20cat_order%3A%20node.svg)

### cat_cutoff: 500 cat_order: tree

![cat_cutoff: 500 cat_order: tree](figures/cat_cutoff%3A%20500%20cat_order%3A%20tree.svg)

### cat_cutoff: 500 cat_order: node

![cat_cutoff: 500 cat_order: node](figures/cat_cutoff%3A%20500%20cat_order%3A%20node.svg)


## Modelling

