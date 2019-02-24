Petfinder notes:

train.csv - main data

train_images - images of the pets (stored by the ID_NUM). Each pet can have multiple pictures.

train_metadata -  annotations of the images(got from google vision API)

train_sentiment - google annotation sentiment

breed_labels 

color_labels

state_labels



Goal: predict the speed of pet adoption.


1. Look at the big picture
2. Get the data.


3. Visualize and analyze the data.
 - There are more dogs then cats
 - Most of the pets has - 4 or 2 - adoption speed.
 - Dogs tend to have slight lower adoption rates.
 - Meaningless named pets have lower adoption rate.
 - Young pets are more likely to be adopted
 - Not named pets have lower possibility for being adopted
 - There are more pure breed pets, then mixed.
 - Maybe it makes sense to split dataset between cats and dogs


 Age possible features:
 	Less then a year.


 Breed possible features:
  Is Mixed

  Color possible features:
  OneColored
  TwoColored
  ThreeColored
  Maybe group by the most popular combinations(e.g. - black&white, brown&white)
  Test if old unnamed is less likely to be adopted
  Test if unnamed young has different likelyhood to be adopted, comparing to young unnamed


 


