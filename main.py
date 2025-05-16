import time
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

        # GENETIC ALGORITHM FEATURE SELECTION
        console.print(Panel("[bold cyan]--- Inizio Selezione Genetica delle Feature ---[/bold cyan]", border_style="cyan"))
        
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
                feature_importance_weight=ga_params.get('feature_importance_weight', 0.4),
                feature_count_weight=ga_params.get('feature_count_weight', 0.3),
                physics_correlation_weight=ga_params.get('physics_correlation_weight', 0.3)
            )
        
        try:
            # Run the genetic algorithm
            start_time = time.time()
            with console.status("[bold cyan]Esecuzione dell'algoritmo genetico in corso...[/bold cyan]", spinner="dots"):
                best_features, log = ga_selector.run()
            end_time = time.time()
            
            console.print(f"[green]✓ Algoritmo genetico completato in [bold]{end_time - start_time:.2f}[/bold] secondi[/green]")
            console.print(f"[green]Selezionate [bold]{len(best_features)}[/bold] feature su [bold]{X_train.shape[1]}[/bold] ({len(best_features)/X_train.shape[1]:.1%})[/green]")
            
            # Display selected features
            if len(best_features) <= 10:
                console.print(f"[yellow]Feature selezionate: [bold]{', '.join(best_features)}[/bold][/yellow]")
            else:
                console.print(f"[yellow]Prime 10 feature selezionate: [bold]{', '.join(best_features[:10])}[/bold]...[/yellow]")
            
            # Analyze results
            console.print(Panel("[bold cyan]Analisi dei risultati della selezione delle feature...[/bold cyan]", border_style="cyan"))
            with console.status("[bold cyan]Analisi in corso...[/bold cyan]", spinner="dots"):
                results = ga_selector.analyze_results()
            console.print("[green]✓ Analisi completata. Grafici salvati in: feature_importance.png, feature_correlation.png, fitness_evolution.png[/green]")
            
            # Physics-based analysis
            console.print(Panel("[bold cyan]Analisi fisica delle feature selezionate...[/bold cyan]", border_style="cyan"))
            with console.status("[bold cyan]Analisi fisica in corso...[/bold cyan]", spinner="dots"):
                physics_results = ga_selector.physics_based_analysis()
            console.print("[green]✓ Analisi fisica completata. Grafici salvati in: boxplot_*.png, feature_state_differentiation.png[/green]")
            
            # Find optimal feature subset
            console.print(Panel("[bold cyan]Ricerca del sottogruppo ottimale di feature...[/bold cyan]", border_style="cyan"))
            with console.status("[bold cyan]Ottimizzazione in corso...[/bold cyan]", spinner="dots"):
                optimal_subset = ga_selector.find_optimal_feature_subset()
            
            # Display optimal feature subset
            console.print(f"[green]Numero ottimale di feature: [bold]{optimal_subset['optimal_feature_count']}[/bold][/green]")
            if len(optimal_subset['optimal_features']) <= 10:
                console.print(f"[yellow]Feature ottimali: [bold]{', '.join(optimal_subset['optimal_features'])}[/bold][/yellow]")
            else:
                console.print(f"[yellow]Prime 10 feature ottimali: [bold]{', '.join(optimal_subset['optimal_features'][:10])}[/bold]...[/yellow]")
            
            console.print("[green]✓ Grafico salvato in: performance_vs_features.png[/green]")
            
            # Use the optimal features for subsequent modeling
            X_train_selected = X_train[optimal_subset['optimal_features']]
            X_test_selected = X_test[optimal_subset['optimal_features']]
            
            console.print(Panel("[bold green]--- Fine Selezione Genetica delle Feature ---[/bold green]", border_style="green"))
            console.print(f"[cyan]Utilizzo delle [bold]{len(optimal_subset['optimal_features'])}/{X_train.shape[1]}[/bold] feature ottimali per i modelli successivi[/cyan]")
            
        except Exception as e:
            console.print(f"[bold red]ERRORE:[/bold red] Durante la selezione genetica delle feature: {e}", style="red")
            console.print("[yellow]⚠️ Continuazione con tutte le feature originali...[/yellow]")
            X_train_selected = X_train
            X_test_selected = X_test
        
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