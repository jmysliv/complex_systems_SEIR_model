from seidr import SEIDR


def DirectOperator(params):
    seidr = seidr_from_params(params)
    return seidr.results.T

def seidr_from_params(params, name="test", no_of_days=79):
    seidr = SEIDR(
        name,
        beta_values = [params[i] for i in range(3)],
        gamma = params[3],
        epsilon = params[4],
        alpha = params[5],
        delta = 0,
        no_of_days=no_of_days
    )
    seidr.calculate()
    return seidr


