Definition of the API
=====================

The Sphinx documentation can be found on root->doc->_build->html->index.html


Installation
------------

To install the add-on, run

    python setup.py install

or

    pip install .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

or

    pip install -e .


Hierarchy
---------

|- orangecontrib/<br>
|   |- recsystems/<br>
|       |- datasets/<br>
|       |- models/<br>
|           |- model_based/<br>
|               |- brismf/<br>
|       |- tests/<br>




Example
------------

    BRISMF -> 150 ratings, 100 users, 25 items.
    ------------------------------------------
    
    DENSE:
        - RMSE: 0.390
        - Time mean: 0.000s
        - Time: 0.277s
        
        * Mac OSX has a feature to compress the memory, so dense matrix are automatically manage as a kind of dense matrices.
        
    SPARSE:
        - RMSE: 0.379
        - Time mean: 0.002s
        - Time: 1.237s
        
        * Too slow (x4-5), I don't know why. (CSR<CSC<COO)
    
    
Bugs and problems
-----------------        
        
        1. EVERYTHING CORRECT BUT ON RETURNING TO THE MAIN THREAD, THE MODEL PROPERTIES ARE LOST
        ----------------------------------------------------------------------------------------
        
        Code:
            data = Orange.data.Table('ratings.tab')
            learner = brismf.BRISMFLearner()
            
            # learner(data) returns a 'model' but "magically" 'recomender' doesn't 
            # have the 'model' attributes. I've traced it and everything is okay
            # except when it returns to this line. Thus, the 'model' attributes are
            # "magically" lost.
            recommender = learner(data)
            
            # THIS LINES THROWS AN ERROR BECAUSE 'recommender' doesn't have the
            # 'model' attributes
            prediction = recommender.predict(user=0, sort=False, top=None)
            print(prediction[:, 1].T)
        
        Extras:
            - On BRISMFLearner(Learner), 'fit' function returns BRISMFModel(self) -> Correct object
            - Main thread, 'recommender = learner(data)' -> Incorrect object
        
        
        2. WHEN I IMPORT ORANGE INTO IPYTHON, HOW DO I ADD MY ADDON TO THE EXISTING ORANGE?
        ("SOLUTION": The current option is to copy the addon to the library manually)
        
        Terminal inputs:
            -> import Orange -> Okay.
            -> learner = Orange.recsystems.model based.BRISMFLearner() -> Error 'recsystems' doesn't exist (Obvious)
    
    
        
RANDOM CHANGES
--------------

    1. I've temporarily removed the CLiMF module because it had no sense to have several modules with the same problems.
       Therefore, when the problems of BRISMF are solved, I will commit CLiMF module again.

