# src/json_parser.py
#
# This utility script loads and validates configuration data from a JSON file.
# It ensures required fields are present, applies default values for optional fields,
# and verifies that all values match expected types according to a predefined schema.
#
# Usage:
#   Call `load_config(filepath)` with the path to a JSON config file.
#   The function returns a validated dictionary suitable for use in your application.
#
# Raises:
#   FileNotFoundError: If the JSON file cannot be found.
#   json.JSONDecodeError: If the file contains invalid JSON.
#   ValueError: If any mandatory config fields are missing or types are incorrect.
#
# This approach promotes reproducibility, portability, and early error detection
# when managing configuration parameters for machine learning experiments or similar projects.

import json

# ---------------- SCHEMA ---------------- #  

# Each key maps to a dict of expected fields:
#   field_name: (type, is_mandatory (bool), default_value or None)

GP_SCHEMA = {
    "system_params": {
        "random_seed": (int, False, 42),
        "verbose": (bool, False, True),
        "n_jobs": (int, False, 1),
        "train_color": (str, False, "blue"),
        "test_color": (str, False, "orange")
    },
    "data_params": {
        "dataset_path": (str, True, None),
        "target_variable": (str, True, None),
        "train_test_split": (float, False, 0.8),
        "outter_folds": (int, False, 5),
        "inner_folds": (int, False, 5)
    },
    "solver_params": {
        "dataset_name":(str, False, None),
        "initializer": (str, True, None),
        "elitism": (bool, False, True),
        "n_elites": (int, False, 1),
        "fitness_function": (str, True, None),
        "minimization": (bool, True, None),
        "algorithm": (str, True, None),
        "tree_constants": (list, True, None),
        "tree_functions": (list, True, None),
        "init_depth": (int, False, 3),
        "tournament_size": (int, False, 2),
        "n_iter": (int, False, 30)
    },
    "grid_params": {
        "pop_size": (list, True, None),
        "prob_const": (list, True, None),
        "p_xo": (list, True, None),
        "max_depth": (list, True, None)
    },
    "logging_params": {
        "log_path": (str, False, "./log/"),
        "log_file": (str, False, "gp_default.csv"),
        "log_level": (int, False, 2),
        "verbose": (bool, False, True)
    }
}

GSGP_SCHEMA = {
    "system_params": {
        "random_seed": (int, False, 42),
        "verbose": (bool, False, True),
        "n_jobs": (int, False, 1),
        "train_color": (str, False, "blue"),
        "test_color": (str, False, "orange")
    },
    "data_params": {
        "dataset_path": (str, True, None),
        "target_variable": (str, True, None),
        "train_test_split": (float, False, 0.8),
        "outter_folds": (int, False, 5),
        "inner_folds": (int, False, 5)
    },
    "solver_params": {
        "dataset_name":(str, False, None),
        "initializer": (str, True, None),
        "elitism": (bool, False, True),
        "n_elites": (int, False, 1),
        "fitness_function": (str, True, None),
        "minimization": (bool, True, None),
        "algorithm": (str, True, None),
        "tree_constants": (list, True, None),
        "tree_functions": (list, True, None),
        "init_depth": (int, False, 3),
        "tournament_size": (int, False, 2),
        "n_iter": (int, False, 30)
    },
    "grid_params": {
        "pop_size": (list, True, None),
        "prob_const": (list, True, None),
        "p_xo": (list, True, None),
        "ms_lower": (list, True, None),
        "ms_upper": (list, True, None)        
    },
    "logging_params": {
        "log_path": (str, False, "./log/"),
        "log_file": (str, False, "gsgp_default.csv"),
        "log_level": (int, False, 2),
        "verbose": (bool, False, True)
    }
}

SLIM_SCHEMA = {
    "system_params": {
        "random_seed": (int, False, 42),
        "verbose": (bool, False, True),
        "n_jobs": (int, False, 1),
        "train_color": (str, False, "blue"),
        "test_color": (str, False, "orange")
    },
    "data_params": {
        "dataset_path": (str, True, None),
        "target_variable": (str, True, None),
        "train_test_split": (float, False, 0.8),
        "outter_folds": (int, False, 5),
        "inner_folds": (int, False, 5)
    },
    "solver_params": {
        "dataset_name":(str, False, None),
        "initializer": (str, True, None),
        "elitism": (bool, False, True),
        "n_elites": (int, False, 1),
        "fitness_function": (str, True, None),
        "minimization": (bool, True, None),
        "algorithm": (str, True, None),
        "tree_constants": (list, True, None),
        "tree_functions": (list, True, None),
        "init_depth": (int, False, 3),
        "tournament_size": (int, False, 2),
        "n_iter": (int, False, 30),
        "reconstruct": (bool, False, False)
    },
    "grid_params": {
        "pop_size": (list, True, None),
        "prob_const": (list, True, None),
        "ms_lower": (list, True, None),
        "ms_upper": (list, True, None),
        "p_inflate": (list, True, None),
        "slim_version": (list, True, None)
    },
    "logging_params": {
        "log_path": (str, False, "./log/"),
        "log_file": (str, False, "gsgp_default.csv"),
        "log_level": (int, False, 2),
        "verbose": (bool, False, True)
    }
}

SCHEMA_DICT = {
    'gp' : GP_SCHEMA,
    'gsgp': GSGP_SCHEMA,
    'slim': SLIM_SCHEMA
}

def load_config(
    filepath, 
    algorithm_schema
):
    
    """
    Load and validate a JSON configuration file according to SCHEMA.
    
    Args:
        filepath (str): Path to the JSON config file.
    
    Returns:
        dict: Validated configuration dictionary, with default values applied.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        ValueError: If mandatory fields are missing or have wrong types.
    """

    with open(filepath, 'r') as f:
        config = json.load(f)
    
    validated_config = {}
    
    for section, fields in SCHEMA_DICT[algorithm_schema].items():
        section_data = config.get(section, {})
        if not isinstance(section_data, dict):
            raise ValueError(f"Section '{section}' must be a dictionary.")
        
        validated_section = {}
        for field, (expected_type, mandatory, default) in fields.items():
            if field in section_data:
                value = section_data[field]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Field '{field}' in section '{section}' must be of type {expected_type.__name__}, "
                        f"got {type(value).__name__} instead."
                    )
                validated_section[field] = value
            else:
                if mandatory:
                    raise ValueError(f"Mandatory field '{field}' missing in section '{section}'.")
                else:
                    validated_section[field] = default
        validated_config[section] = validated_section
    
    return validated_config

if __name__ == "__main__":

    # Example usage
    config_path = "./configs/sample_config.json"
    
    try:
        config = load_config(config_path)
        print("Loaded and validated config:")
        print(json.dumps(config, indent=4))
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON - {e}")
    except ValueError as e:
        print(f"Config validation error: {e}")
