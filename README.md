# SEIDR

## Simplified baseline model

```
    S = 9900,
    E = 100,
    I = 0,
    D = 0,
    R = 0,
    beta = 0.75,
    gamma = 0.1,
    mi = 0.0002,
    lambda_ = 0.0003,
    epsilon = 0.2,
    alpha = 0.01,
    no_of_days = 100
```

## Baseline model

```
S = 9900,
E = 100,
I = 0,
D = 0,
R = 0,
beta_bins = [20, 50],
beta_values = [0.75, 0.5, 0.25],
gamma = 0.1,
mi = 0.0002,
lambda_ = 0.0003,
epsilon = 0.2,
alpha = 0.01,
delta = 0.01,
no_of_days = 400
```

## Modified models

### Model with changed Beta2 parameter

```
S = 9900,
E = 100,
I = 0,
D = 0,
R = 0,
beta_bins = [20, 50],
beta_values = [0.75, 1.0, 0.25],
gamma = 0.1,
mi = 0.0002,
lambda_ = 0.0003,
epsilon = 0.2,
alpha = 0.01,
delta = 0.01,
no_of_days = 400
```

### Model with changed Beta1 parameter

```
S = 9900,
E = 100,
I = 0,
D = 0,
R = 0,
beta_bins = [20, 50],
beta_values = [0.1, 0.5, 0.25],
gamma = 0.1,
mi = 0.0002,
lambda_ = 0.0003,
epsilon = 0.2,
alpha = 0.01,
delta = 0.01,
no_of_days = 400
```

### Model with changed mi parameter

```
S = 9900,
E = 100,
I = 0,
D = 0,
R = 0,
beta_bins = [20, 50],
beta_values = [0.75, 0.5, 0.25],
gamma = 0.1,
mi = 0.001,
lambda_ = 0.0003,
epsilon = 0.2,
alpha = 0.01,
delta = 0.01,
no_of_days = 400
```
