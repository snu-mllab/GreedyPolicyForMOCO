from setbench.metrics.r2 import r2_indicator_set
from setbench.metrics.hsr_indicator import HSR_Calculator
from setbench.tools.pareto_op import pareto_frontier
from pymoo.indicators.hv import HV
import numpy as np

def get_hv(rewards, return_val=True, **kwargs):
    """
        solutions are not need to be the pareto front.
    """
    # get pareto_targets
    _, pareto_targets = pareto_frontier(None, rewards, maximize=True)
    hv_indicator = HV(ref_point=kwargs["hv_ref"])
    if type(pareto_targets) == list:
        hv = [hv_indicator.do(-pareto_targets[i].cpu().numpy()) for i in range(len(pareto_targets))]
    else:
        hv = hv_indicator.do(-pareto_targets.cpu().numpy())
    if return_val:
        return hv
    else:
        return hv, pareto_targets

def get_all_metrics(solutions, eval_metrics, **kwargs):
    """
    This method assumes the solutions are already filtered to the pareto front
    """
    
    metrics = {}
    if "hypervolume" in eval_metrics and "hv_ref" in kwargs.keys():
        # hv_indicator = get_performance_indicator('hv', ref_point=kwargs["hv_ref"])
        hv_indicator = HV(ref_point=kwargs["hv_ref"])
        # `-` cause pymoo assumes minimization
        metrics["hypervolume"] = hv_indicator.do(-solutions)
    
    if "r2" in eval_metrics and "r2_prefs" in kwargs.keys() and "num_obj" in kwargs.keys():
        metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))
    
    if "hsri" in eval_metrics and "num_obj" in kwargs.keys():
        # class assumes minimization so transformer to negative problem
        hsr_class = HSR_Calculator(lower_bound=-np.ones(kwargs["num_obj"]) - 0.1, upper_bound=np.zeros(kwargs["num_obj"]) + 0.1)
        # try except cause hsri can run into divide by zero errors 
        try:
            metrics["hsri"], x = hsr_class.calculate_hsr(-solutions)
        except:
            metrics["hsri"] = 0.
        try:
            metrics["hsri"] = metrics["hsri"] if type(metrics["hsri"]) is float else metrics["hsri"][0]
        except:
            metrics["hsri"] = 0.
    return metrics