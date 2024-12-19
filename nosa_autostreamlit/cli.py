import argparse
import sys
from .generators.base_generator import BaseGenerator
from .utils.data_utils import DataProcessor

def main():
    """
    Command-line interface for Nosa-autoStreamlit
    """
    parser = argparse.ArgumentParser(description='Nosa-autoStreamlit: Automated Streamlit App Generator')
    
    parser.add_argument('--create', action='store_true', 
                        help='Create a new Streamlit application')
    parser.add_argument('--data', type=str, 
                        help='Path to data file for app generation')
    
    args = parser.parse_args()
    
    if args.create:
        generator = BaseGenerator()
        
        # Example of adding components
        generator.add_component('title', title='Nosa-autoStreamlit App')
        
        if args.data:
            try:
                df = DataProcessor.load_data(args.data)
                generator.add_component('dataframe', data=df)
                DataProcessor.display_data_summary(df)
            except Exception as e:
                print(f"Error processing data: {e}")
                sys.exit(1)
        
        # Generate the app (this would typically be run via Streamlit)
        generator.generate_app()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()