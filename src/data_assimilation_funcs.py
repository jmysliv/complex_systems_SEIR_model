from seidr import SEIDR


def DirectOperator(params):
    seidr = seidr_from_params(params)
    return seidr.results.T

def seidr_from_params(params, name="test"):
    seidr = SEIDR(
        name,
        beta_values = [params[i] for i in range(3)],
        gamma = params[3],
        epsilon = params[4],
        alpha = params[5],
        delta = params[6]
    )
    seidr.calculate()
    return seidr


