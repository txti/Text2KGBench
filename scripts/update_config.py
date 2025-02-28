import json
import sys

from kgbench.ontology import update_onto_list


def main():
    config_path = next((arg.split('=')[1] for arg in sys.argv
                       if arg.startswith('--config_path=')), None)

    if not config_path:
        print("Error: Please provide --config_path argument")
        return

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        updated_data = update_onto_list(data)

        if updated_data:
            with open(config_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            print("Successfully updated config file")

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        print("Invalid JSON format in config file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
