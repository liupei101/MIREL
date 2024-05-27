from .evaluator_clf import BinClf_Evaluator
from .evaluator_clf import MultiClf_Evaluator
from .evaluator_unc import Uncertainty_Evaluator

def load_evaluator(task, task_level, *args, **kws):
    if task == 'clf':
        if task_level == 'bag':
            print("[info] loading bag-level evaluator")
            if kws['binary_clf']:
                evaluator = BinClf_Evaluator(**kws)
            else:
                evaluator = MultiClf_Evaluator(**kws)
        elif task_level == 'instance':
            print("[info] loading instance-level evaluator")
            if kws['binary_clf']:
                kws['ins_output'] = True
                evaluator = BinClf_Evaluator(**kws)
            else:
                kws['ins_output'] = True
                evaluator = MultiClf_Evaluator(**kws)
        else:
            evaluator = None
    else:
        pass
    
    return evaluator

def load_uncertainty_evaluator(task, task_level, *args, **kws):
    if task == 'clf':
        if task_level == 'bag':
            print("[info] loading bag-level uncertainty evaluator")
            if kws['binary_clf']:
                evaluator = Uncertainty_Evaluator(**kws)
            else:
                evaluator = None
        elif task_level == 'instance':
            print("[info] loading instance-level uncertainty evaluator")
            kws['ins_output'] = True
            if kws['binary_clf']:
                evaluator = Uncertainty_Evaluator(**kws)
            else:
                evaluator = None
        else:
            evaluator = None
    else:
        pass
    
    return evaluator