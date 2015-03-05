# NodeHarvest
Node Harvest selects a small subset of nodes from a Random Forest predictor. This potentially increases both,
interpretability and predictive accuracy.
For details about the algorithm see [1] [(pdf)](http://www.stats.ox.ac.uk/~meinshau/AOAS367.pdf).


This implementation is written in Python 3 and works on top of the 
[RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 
from [scikit-learn](http://scikit-learn.org/).
For solving the quadratic optimization either
[scipy.optimize.minimize](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) (slow) or
[cvxopt.solvers.qp] (http://cvxopt.org/userguide/coneprog.html#quadratic-programming) (preferred) are available.

[1] [N. Meinshausen. "Node Harvest". *The Annals of Applied Statistics*, 4(4), 2010]
    (http://www.stats.ox.ac.uk/~meinshau/AOAS367.pdf)

## Installation
There is no installer yet. Simply copy `nodeharvest.py` into your project source tree.

## Usage Example

``` Python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from nodeharvest import NodeHarvest

model = lambda x: np.prod(np.sin(2 * np.pi * x), axis=1) + np.random.randn(x.shape[0]) * 0.25

x_train = np.random.rand(100, 3)
y_train = model(x_train)

x_test = np.random.rand(10, 3)
y_test = model(x_test)

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20)
rf.fit(x_train, y_train)

nh = NodeHarvest(solver='scipy_robust', verbose=True)
nh.fit(rf, x_train, y_train)

y_est = nh.predict(x_test)

print('     true y    predicted y  ')
print(np.transpose([y_test, y_est]))
```

## Test Suite
Currently, unit tests are included in `nodeharvest.py` directly:

    python nodeharvest.py
    
or

    nosetests nodeharvest.py

## License

This implementation of Node Harvest is [**BSD-licensed** (3 clause)](http://opensource.org/licenses/BSD-3-Clause):

> Copyright (c) 2015, Martin Billinger.
> All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
> 
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> 
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
> 
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
> 
> **This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.**
