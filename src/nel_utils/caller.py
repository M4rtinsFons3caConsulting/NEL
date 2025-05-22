from itertools import product
from slim_gsgp.main_gp import gp
from slim_gsgp.main_gsgp import gsgp
from slim_gsgp.main_slim import slim

model_dict = {
    "gp": gp,
    "gsgp": gsgp,
    "slim": slim
}

def call_model(
        fixed_params, 
        param_grid, 
        seed,
        set_max_depth = False
    ):
    
    # Copy and pop
    fixed_params = fixed_params.copy()
    model_key = fixed_params.pop('algorithm')

    models = []
    keys, values = zip(*param_grid.items())

    for combo in product(*values):

        dynamic_params = dict(zip(keys, combo))
        full_params = {**fixed_params, **dynamic_params}

        if set_max_depth:
            full_params.update({'max_depth': full_params['init_depth']+15})

        model = model_dict[model_key](**full_params, seed=seed)

        res = {'model': model}
        res.update({'rmse_train': model.fitness.item()})
        res.update({'rmse_test': model.test_fitness.item()})
        res.update({'dynamic_params': dynamic_params})

        models.append(res)

    return models      
