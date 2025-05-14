from data_loader import load_all_csvs_from_folder 
from data_loader.data_loader import load_specific_csv_from_folder
from preprocessor import preprocess_data
from classifier import train_and_evaluate_xgboost, train_and_evaluate_catboost, train_and_evaluate_random_forest
from calibration import calibrate_model, get_calibrated_probabilities
from explainer import explain_model_with_shap
from genetic_algorithm import GeneticFeatureSelector
from bayesian_methods.bayesian_methods import EsempioMetodoBayesiano 

import pandas as pd
import yaml
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
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
        
        console.print(Panel("[bold green]--- Fine Preprocessing ---[/bold green]", border_style="green"))
        console.print(f"[cyan]Set di training: [bold]{X_train.shape[0]} esempi, {X_train.shape[1]} features[/bold][/cyan]")
        console.print(f"[cyan]Set di testing: [bold]{X_test.shape[0]} esempi, {X_test.shape[1]} features[/bold][/cyan]")

        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            console.print("[green]✓ Target convertiti in formato intero[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Attenzione: non è stato possibile convertire y_train/y_test in interi: [/yellow][dim]{e}[/dim]")

        
        # print("\n--- Genetic Feature Selection ---")
        # ga_params = config.get('genetic_algorithm', {}).get('params', {})
        # ga_selector = GeneticFeatureSelector(
        #     X_train=X_train, 
        #     X_test=X_test,
        #     y_train=y_train,
        #     y_test=y_test,
        #     population_size=ga_params.get('popolazione', 50),
        #     generations=ga_params.get('generazioni', 30)
        # )
        
        # best_features, log = ga_selector.run()
        # results = ga_selector.analyze_results()
        # print(f"Selected {len(best_features)} features: {', '.join(best_features)}")
        # print("--- End Genetic Feature Selection ---")
        
        # # 2. Addestramento e Valutazione Modelli
        # print("\n--- Addestramento Modelli ---")
        # xgb_model, xgb_accuracy, xgb_report = train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)
        
        # catboost_model, catboost_accuracy, catboost_report = train_and_evaluate_catboost(X_train, y_train, X_test, y_test)
        
        # rf_model, rf_accuracy, rf_report = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)
        # print("--- Fine Addestramento Modelli ---")

        # # 3. Calibrazione (Esempio con il modello Random Forest)
        # print("\n--- Calibrazione Modello (Random Forest) ---")

        # calibrated_rf_model = calibrate_model(rf_model, X_train, y_train, X_test, method='isotonic')
        # if calibrated_rf_model is not None:
        #     calibrated_probs = get_calibrated_probabilities(calibrated_rf_model, X_test)
        # print("--- Fine Calibrazione Modello ---")

        # # 4. Spiegabilità (Esempio con il modello Random Forest)
        # print("\n--- Spiegabilità Modello (Random Forest con SHAP) ---")

        # shap_explainer, shap_values = explain_model_with_shap(rf_model, X_train, X_test, model_type='tree')
        # # Per visualizzare i plot SHAP (es. summary_plot), di solito serve un ambiente grafico
        # # o salvare i plot su file usando matplotlib. Ad esempio:
        # # import shap
        # # import matplotlib.pyplot as plt
        # # if shap_values is not None:
        # #     shap.summary_plot(shap_values, X_test, show=False)
        # #     plt.savefig('models/shap_summary_plot.png') # Salva il plot
        # #     plt.close()
        # #     print("Plot SHAP salvato in models/shap_summary_plot.png")
        # print("--- Fine Spiegabilità Modello ---")

        # # 5. Esempio di utilizzo del Modulo Algoritmo Genetico
        # print("\n--- Esempio Algoritmo Genetico ---")
        # parametri_ga = {'popolazione': 100, 'generazioni': 50}
        # ga_optimizer = EsempioAlgoritmoGenetico(parametri_ga)
        # risultato_ga = ga_optimizer.ottimizza(data='dati_per_ga_placeholder')
        # print(f"Risultato dell'ottimizzazione genetica: {risultato_ga}")
        # print("--- Fine Esempio Algoritmo Genetico ---")

        # # 6. Esempio di utilizzo del Modulo Metodi Bayesiani
        # print("\n--- Esempio Metodi Bayesiani ---")
        # iperparametri_bayes = {'alpha': 1.0, 'beta': 1.0}
        # bayes_model = EsempioMetodoBayesiano(iperparametri_bayes)
        # posteriore_bayes = bayes_model.inferenza(osservazioni='osservazioni_placeholder')
        # print(f"Risultato dell'inferenza bayesiana: {posteriore_bayes}")
        # print("--- Fine Esempio Metodi Bayesiani ---")

    else:
        console.print(Panel(f"[bold red]Pipeline interrotto a causa di un errore nel caricamento dei dati dalla cartella '{data_folder}'.[/bold red]", 
                            border_style="red"))

    console.print(Panel("[bold blue]Pipeline di classificazione multiclasse completato.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Check if the config file exists
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERRORE:[/bold red] Il file di configurazione 'config.yaml' non esiste.", style="red")
        exit(1)
    
    main()