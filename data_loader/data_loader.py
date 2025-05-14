import pandas as pd
import os
import glob

def load_csv(file_path):
    """Carica un file CSV in un DataFrame pandas."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dati caricati con successo da {file_path}")
        return df
    except FileNotFoundError:
        print(f"Errore: File non trovato in {file_path}")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento del file CSV {file_path}: {e}")
        return None


def load_specific_csv_from_folder(folder_path, file_name):
    """
    Carica un file CSV specifico da una cartella.
    
    Parameters:
    folder_path (str): Il percorso della cartella contenente i file CSV
    file_name (str): Il nome del file specifico da caricare
    
    Returns:
    DataFrame or None: Il DataFrame pandas caricato o None in caso di errore
    """
    if not os.path.isdir(folder_path):
        print(f"Errore: La cartella specificata '{folder_path}' non esiste o non è una directory.")
        return None
    
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_name}' non esiste nella cartella '{folder_path}'.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"File caricato con successo: {file_name} ({len(df)} righe)")
        return df
    except pd.errors.EmptyDataError:
        print(f"Attenzione: Il file CSV '{file_name}' è vuoto.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento del file CSV '{file_name}': {e}")
        return None


def load_all_csvs_from_folder(folder_path):
    """Carica tutti i file CSV da una cartella specificata e li concatena."""
    if not os.path.isdir(folder_path):
        print(f"Errore: La cartella specificata '{folder_path}' non esiste o non è una directory.")
        return None

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"Nessun file CSV trovato nella cartella '{folder_path}'.")
        return None
    
    all_dfs = []
    print(f"Trovati {len(csv_files)} file CSV nella cartella '{folder_path}'. Inizio caricamento...")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            print(f"  Caricato con successo: {os.path.basename(file_path)} ({len(df)} righe)")
        except FileNotFoundError:
            print(f"  Errore: File non trovato {file_path} (questo non dovrebbe accadere se glob lo ha trovato).")
        except pd.errors.EmptyDataError:
            print(f"  Attenzione: Il file CSV '{os.path.basename(file_path)}' è vuoto.")
        except Exception as e:
            print(f"  Errore durante il caricamento del file CSV '{os.path.basename(file_path)}': {e}")

    if not all_dfs:
        print("Nessun DataFrame è stato caricato con successo.")
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Tutti i file CSV sono stati caricati e concatenati. DataFrame combinato con {len(combined_df)} righe.")
    return combined_df