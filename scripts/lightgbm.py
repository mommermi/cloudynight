""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This script shows how to train the lightGBM model on feature data and
how to make predictions with this model. In its current state,
the script loads data from example_data/features/. If the cloudynight webapp
is setup properly, it can readily use data extracted from the `labeled` table
of the database.

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""
import cloudynight

# initialize model
model = cloudynight.LightGBMModel()

# load feature example data
model.load_data('../example_data/features/fulltrainingsample_features.dat')

# train the model using parameters defined in __init__.py
model.train()

print('model trained with training/test/validation scores:', model.train_score,
      model.test_score, model.val_score)

# save the trained model
model.write_model('../workbench/lightgbm.pickle')

# apply model to predict presence of clouds in a random subregion from
# the training data set
i = 12345
print('Is there a cloud in training example {}? {}.'.format(
    i, model.data_y[i]==1))
print('The lightgbm model finds {} cloud in this subregion.'.format(
    {1: 'a', 0: 'no'}[
        model.predict(model.data_X.iloc[i].values.reshape(1, -1))[0]]))
