# Data Challenge 2
## Team : Paul Aristidou, Guillaume Bocquillion, Guillaume Cazottes, Lucas Guerrot, Olivier Lapabe-Goastat, 

Nous avons fait ce projet à l'aide de fichiers .py et d'un dépôt git. Cela nous a permis d'avoir un code modulaire, facilitant le travail de groupe et pratique à lancer sur des machines distantes.

## EDA

Nous avons réalisé une analyse exploratoire des données dont les résultats se trouvent sur le github (EDA.ipynb).
Dans cette analyse, nous constatons notamment :  
•⁠  ⁠Nombre d'images important : 101 345 images.  
•⁠  ⁠La variable "gender" est continue : elle varie entre 0 (femme) et 1 (homme).  
•⁠  ⁠Répartition H/F plutôt équilibrée : 40% de femmes et 60% d'hommes.  
•⁠  ⁠Analyse de la distribution des occlusions : La distribution semble suivre une exponentielle décroissante.  


## Idées de la solution proposée

•⁠  ⁠Le nombre d'images étant important (I/O intensive), on a décidé d'augmenter le nombre de workers du CPU pour améliorer I/O.  
•⁠  ⁠Nous avons réalisé un pipeline de prétraitement des images (réalisé à la volée sur chaque batch) :  
&nbsp;&nbsp;&nbsp;&nbsp;- RamdomHorizontalFlip pour 50% des images  
&nbsp;&nbsp;&nbsp;&nbsp;- RandomRotation entre -10° et 10°  
&nbsp;&nbsp;&nbsp;&nbsp;- Tranformation en tenseur  
&nbsp;&nbsp;&nbsp;&nbsp;- Normalization  
•⁠  ⁠Choix du modèle  
&nbsp;&nbsp;&nbsp;&nbsp;- Resnet101 pré-train sur ImageNet  
&nbsp;&nbsp;&nbsp;&nbsp;- batch size : 256  
&nbsp;&nbsp;&nbsp;&nbsp;- optimizer : Adam (learning rate = 0.001)  
&nbsp;&nbsp;&nbsp;&nbsp;- Loss function : MSE  
&nbsp;&nbsp;&nbsp;&nbsp;- num_epochs : 141  


## Voies testées et non concluantes

•⁠  ⁠Analyse des erreurs obtenues avec la baseline suivant le genre de la personne -> Pas de différence.  
•⁠  ⁠Nous avons essayé de sauvegarder les fichiers directement sous forme de tensors (avec un format .pt) et aussi le format HDF5. Nous n'avons cependant pas constaté d'améliorations en temps de calcul.  
•⁠  ⁠Nous avons fait plusieurs tentatives concernant la data augmentation : pas de data augmentation, rotation plus ou moins importante des images, crop des images (??), normalization ou pas.  
•⁠  ⁠Nous avons testé de nombreux modèles : MobileNet_V3_small, MobileNet_V3_large, EfficientNet, ShuffleNet, ResNext, différents modèles ResNet, Vision Transformers (DeiTforImageClassification pretrained)...  
De la même manière, nous avons testés plusieurs hyperparamètres (batch size, learning rate... ).   
•⁠  ⁠Nous avons aussi essayé d'utiliser une "loss custom", qui était définie de la même manière que le score à optimiser :

$$
Score = \frac{Err_F + Err_M}{2} + \left | Err_F - Err_M \right |
$$

-> Pas d'amélioration du score.  

## Fonctionnement du code et structure du git

### Train - Train_runner.py
1. Create training and validation dataloaders, including loading and preprocessing (using src/DataLoader/Dataloader.py).  
2. Instanciate solver, train model, plot and log loss data for train and val (using src/Solver/Solver.py).  
3. Save best epoch according to lowest validation loss.  

### Test - Test_runner.py
1. Create training dataloader (composed of training and validation dataset), including loading and preprocessing.  
2. Create testing dataloader including loading and preprocessing but without data augmentation.
3. Tester.py:  
3.1. Instanciate tester and train model on training dataloader (using src/Tester/Tester.py) until the best epoch.    
3.2. Test on testing dataloader in order to create the .csv to submit (using src/Tester/Tester.py).   
