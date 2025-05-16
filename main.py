import time
from data_loader import load_all_csvs_from_folder 
from data_loader.data_loader import load_specific_csv_from_folder
from preprocessor import preprocess_data
from classifier import train_and_evaluate_xgboost, train_and_evaluate_catboost, train_and_evaluate_random_forest
from calibration import calibrate_model, get_calibrated_probabilities
from explainer import explain_model_with_shap
from genetic_algorithm import GeneticFeatureSelector

import pandas as pd
import yaml
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as print
from rich.progress import track


# Create a console instance
console = Console()


def main():
    console.print(Panel.fit("[bold blue]Avvio del pipeline di classificazione multiclasse...[/bold blue]", 
                           border_style="blue"))

    # Load configuration from YAML file
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configurazione caricata correttamente da [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        console.print(f"[bold red]ERRORE:[/bold red] File di configurazione '[bold]{config_path}[/bold]' non trovato. Assicurarsi che esista.", style="red")
        exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]ERRORE:[/bold red] nel parsing del file YAML '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]ERRORE:[/bold red] sconosciuto durante il caricamento della configurazione '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)
    
    # 1. Caricamento Dati
    data_folder = config['data_loader']['data_folder_path']
    specific_file = config['data_loader'].get('specific_file')

    console.print(Panel(f"[yellow]📂 Caricamento dati da: [bold]{data_folder}[/bold][/yellow]", 
                       border_style="yellow"))
    
    if specific_file:
        console.print(f"[yellow]File specifico: [bold]{specific_file}[/bold][/yellow]")
        dataframe = load_specific_csv_from_folder(data_folder, specific_file)
    else:
        console.print(f"[yellow]Caricamento di tutti i file CSV dalla cartella[/yellow]")
        dataframe = load_all_csvs_from_folder(data_folder)

    if dataframe is not None:
        console.print(Panel(f"[bold green]✓ Dati caricati: [/bold green][cyan]{len(dataframe)} righe, {len(dataframe.columns)} colonne[/cyan]"))
        
        console.print(Panel("[bold magenta]--- Inizio Preprocessing ---[/bold magenta]", border_style="magenta"))
        target_column = config['data_loader']['target_column'] 
        console.print(f"[magenta]Target column: [bold]{target_column}[/bold][/magenta]")
        
        with console.status("[bold magenta]Preprocessing dei dati...[/bold magenta]", spinner="dots"):
            X_train, X_test, y_train, y_test = preprocess_data(dataframe, target_column)
        
        console.print(Panel("[bold green]✓ Preprocessing completato[/bold green]", border_style="green"))
        if config["settings"].get("verbose", 0) > 1:
            console.print(f"[green]X_train:\n {X_train.head()}[/green]")
            console.print(f"[green]y_train:\n {y_train.head()}[/green]")
            console.print(f"[green]X_test:\n {X_test.head()}[/green]")
            console.print(f"[green]y_test:\n {y_test.head()}[/green]")

        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            console.print("[green]✓ Target convertiti in formato intero[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Attenzione: non è stato possibile convertire y_train/y_test in interi: [/yellow][dim]{e}[/dim]")

        if config["settings"].get("feature_selection", True):
            console.print(Panel("[bold yellow]--- Feature Selection with GA Enabled ---[/bold yellow]", border_style="yellow"))
            
            # Get GA parameters from config
            ga_params = config.get('genetic_algorithm', {}).get('params', {})
            
            # Initialize the genetic algorithm with parameters
            with console.status("[bold cyan]Inizializzazione del GA...[/bold cyan]", spinner="dots"):
                ga_selector = GeneticFeatureSelector(
                    X_train=X_train, 
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    population_size=ga_params.get('popolazione', 50),
                    generations=ga_params.get('generazioni', 30),
                    crossover_prob=ga_params.get('crossover_prob', 0.7),
                    mutation_prob=ga_params.get('mutation_prob', 0.2),
                    accuracy_weight=ga_params.get('accuracy_weight', 0.6),
                    feature_count_weight=ga_params.get('feature_count_weight', 0.2),
                    correlation_weight=ga_params.get('correlation_weight', 0.2)
                )

                console.print(Panel("[bold green]✓ GA inizializzato correttamente[/bold green]", border_style="green"))
            
            # Run the genetic algorithm
            with console.status("[bold cyan]Esecuzione del GA...[/bold cyan]", spinner="dots"):
                selected_features = ga_selector.run()
                console.print(Panel(f"[bold green]✓ Esecuzione del GA completata con successo![/bold green]", border_style="green"))
                console.print(f"[cyan]Feature selezionate: {len(selected_features)}[/cyan]")
            
            if config["settings"].get("verbose", 0) > 0:
                ga_selector.plot_fitness_history()
                
            # Filter the X_train and test based on the list of selected features
            X_train = X_train[selected_features]
            X_test = X_test[selected_features] 
            console.print(Panel(f"[bold green]✓ Filtered Dataset Features[/bold green]", border_style="green"))
            
        # Train and evaluate models
        console.print(Panel("[bold green]--- Training Model ---[/bold green]", border_style="green"))
        
        
        
    else:
        console.print(Panel(f"[bold red]Pipeline interrotto a causa di un errore nel caricamento dei dati dalla cartella '{data_folder}'.[/bold red]", 
                            border_style="red"))

    console.print(Panel("[bold blue]Pipeline di classificazione multiclasse completato.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Check if the config file exists
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERRORE:[/bold red] Il file di configurazione 'config.yaml' non esiste.", style="red")
        exit(1)
    
    main()