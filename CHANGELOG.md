# CHANGELOG:

## **Bugfix 1_20_05: Fixed log_settings(), in main_slim.py.**
<br>
FILE: slim_gsgp.main_slim.py<br>
FUNCTION: slim(...)<br>
LINE: 291 <br>
<br>
**`FROM`**
````
    log_settings(
        path=os.path.join(os.getcwd(), "log", "slim_settings.csv"),
        settings_dict=[slim_gsgp_solve_parameters,
                       slim_gsgp_parameters,
                       slim_gsgp_pi_init,
                       settings_dict],
        unique_run_id=UNIQUE_RUN_ID
    )
````

**`TO`**
````
log_settings(
    path=log_path[:-4] + "_settings.csv",
    settings_dict=[slim_gsgp_solve_parameters,
                    slim_gsgp_parameters,
                    slim_gsgp_pi_init,
                    settings_dict],
    unique_run_id=UNIQUE_RUN_ID
)
````
**REASON: **

Based on logging pattern identified in slim_gsgp.main_gp.py and slim_gsgp.main_gsgp.py for logger, at equivalent location in those respective files.
