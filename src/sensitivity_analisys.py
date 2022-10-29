from SALib.sample.sobol import sample
from SALib.analyze import sobol
import numpy as np
from seidr import SEIDR

def sa(N):
    '''
    manipulate only the parameters of the disease - playing with the starting numbers seems
    risky because it's hard to avoid situations where we mismatch the parameters and the starting 
    numbers in a way where the starting numbers would be unrealistic for the parameters. 
    For example, if the disease has barely any dormant stage and Exposed turn into Infected
    almost immediately, it wouldn't make sense to start with a big Exposed population.
    '''
    problem = {
    'num_vars': 9,
    'names': [
            'beta1', 'beta2', 'beta3', 'gamma', 'mi', 'lambda', 'epsilon', 'alpha', 'delta'
    ],
    'bounds': [
        [0.0,1.0]*9
    ]
    }

    param_values = sample(problem, N, calc_second_order=False)
    print(param_values.shape)

    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        model = SEIDR(beta_values=[X[0],X[1], X[2]],gamma=X[3],mi=X[4], lambda_=X[5], epsilon=X[6], alpha=X[7], delta=X[8])
        model.calculate()
        Y[i] = model.get_final_fatalities()

    Si = sobol.analyze(problem, Y, print_to_console=False,calc_second_order=False)
    return Si

if __name__=="__main__":
    Si = sa(10)
    print(Si.to_df())
