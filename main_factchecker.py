import yaml

def load_config(filename = 'my_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration variables
my_config = load_config("my_config.yaml")
prompting = my_config['prompting']

if __name__ == "__main__":
    prompting = prompting.lower()
    if prompting == "hiss":
        import Python_scripts.HiSS_Prompt_Composer.factchecker
        print("HiSS prompting style selected.")
        Python_scripts.HiSS_Prompt_Composer.factchecker.main()
    elif prompting == "standard":
        import Python_scripts.standard_fact_checker.factchecker
        print("Standard prompting style selected.")
        Python_scripts.standard_fact_checker.factchecker.main()
    elif prompting == "react":
        #execute the react prompting
        print("ReAct prompting style selected.")
        pass
    elif prompting == "nosearch":
        import Python_scripts.nosearch_fact_checker.factchecker
        print("NoSearch prompting style selected.")
        Python_scripts.nosearch_fact_checker.factchecker.main()
    else:
        print("Prompting style not recognized. Please choose between 'nosearch', 'HiSS', 'Standard', or 'ReAct'.")
        exit(1)
