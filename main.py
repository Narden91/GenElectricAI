from data_loader import load_all_csvs_from_folder 
from preprocessor import preprocess_data
from classifier import train_and_evaluate_xgboost, train_and_evaluate_catboost, train_and_evaluate_random_forest
from calibration import calibrate_model, get_calibrated_probabilities
from explainer import explain_model_with_shap
from genetic_algorithm.genetic_algorithm import EsempioAlgoritmoGenetico
from bayesian_methods.bayesian_methods import EsempioMetodoBayesiano 

import pandas as pd
import yaml
import os
import warnings


def main():
    print("Avvio del pipeline di classificazione multiclasse...")

    # Load configuration from YAML file
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERRORE: File di configurazione '{config_path}' non trovato. Assicurarsi che esista.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ERRORE nel parsing del file YAML '{config_path}': {e}")
        exit(1)
    except Exception as e:
        print(f"ERRORE sconosciuto durante il caricamento della configurazione '{config_path}': {e}")
        exit(1)
    
    # 1. Caricamento Dati
    data_folder = config['data_loader']['data_folder_path'] 

    dataframe = load_all_csvs_from_folder(data_folder) 

    if dataframe is not None:
        print("\n--- Inizio Preprocessing ---")
        target_column = config['data_loader']['target_column'] 
        X_train, X_test, y_train, y_test = preprocess_data(dataframe, target_column)
        print("--- Fine Preprocessing ---")

        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except Exception as e:
            print(f"Attenzione: non è stato possibile convertire y_train/y_test in interi: {e}")

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
        print(f"Pipeline interrotto a causa di un errore nel caricamento dei dati dalla cartella '{data_folder}'.")

    print("\nPipeline di classificazione multiclasse completato.")


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Check if the config file exists
    if not os.path.exists('config.yaml'):
        print("ERRORE: Il file di configurazione 'config.yaml' non esiste.")
        exit(1)
    
    main()