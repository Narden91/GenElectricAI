import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import numpy as np # For checking y_train unique values


def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, 
                             model_config: dict, cv_config: dict, console: Console):
    """
    Trains and evaluates a specified classifier.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
        model_config: Dictionary with model type and parameters.
                      Example: {'type': 'RandomForest', 
                                'RandomForest_params': {'n_estimators': 100},
                                'global_random_state': 42}
        cv_config: Dictionary with cross-validation parameters.
                   Example: {'n_splits': 5, 'shuffle': True}
        console: Rich Console object for printing.

    Returns:
        tuple: (trained_model, confusion_matrix, test_accuracy, classification_report_dict)
               Returns (None, None, None, None) if model type is unsupported or error occurs.
    """
    model_type = model_config.get('type', 'RandomForest').lower()
    # Use model-specific params if available, else empty dict
    model_params = model_config.get(f"{model_type}_params", {}) 
    
    global_random_state = model_config.get('global_random_state', 42)

    console.print(Panel(f"[bold cyan]Initializing Model: {model_type.capitalize()} [/bold cyan]", border_style="cyan"))

    # Ensure y_train and y_test are 1D
    if hasattr(y_train, 'ndim') and y_train.ndim > 1:
        y_train = y_train.squeeze()
    if hasattr(y_test, 'ndim') and y_test.ndim > 1:
        y_test = y_test.squeeze()

    # Check number of unique classes in y_train
    try:
        # Convert to numpy array first to handle potential mixed types if y_train is object dtype
        y_train_unique = np.unique(y_train.astype(str)) if y_train.dtype == 'object' else np.unique(y_train)
        num_class = len(y_train_unique)
        console.print(f"[cyan]Number of unique classes in y_train: {num_class}[/cyan]")
        if num_class <= 1:
            console.print(f"[bold red]ERRORE: Il target y_train contiene {num_class} classe/i unica/che. Necessarie almeno 2 classi per la classificazione.[/bold red]", style="red")
            return None, None, None, None
    except Exception as e:
        console.print(f"[bold red]ERRORE: Impossibile determinare il numero di classi da y_train: {e}[/bold red]", style="red")
        return None, None, None, None

    if model_type == 'randomforest':
        if 'random_state' not in model_params:
            model_params['random_state'] = global_random_state
        model = RandomForestClassifier(**model_params)
    elif model_type == 'xgboost':
        if 'random_state' not in model_params:
            model_params['random_state'] = global_random_state
        
        # XGBoost specific handling for multiclass
        if num_class > 2:
            if 'objective' not in model_params:
                model_params['objective'] = 'multi:softprob'
            if 'num_class' not in model_params: # Required for multi:softprob if not inferred
                 model_params['num_class'] = num_class
            if 'eval_metric' not in model_params:
                model_params['eval_metric'] = 'mlogloss'
             # Check if y_train labels are in [0, num_class-1]
            min_label, max_label = y_train.min(), y_train.max()
            if not (min_label == 0 and max_label == num_class - 1 and y_train.nunique() == num_class):
                 console.print(f"[yellow]⚠️  ATTENZIONE (XGBoost): y_train non sembra essere codificato correttamente nell'intervallo [0, num_class-1].[/yellow]")
                 console.print(f"[yellow]   Min label: {min_label}, Max label: {max_label}, Unique labels: {y_train.nunique()}, Expected num_class: {num_class}.[/yellow]")
                 console.print(f"[yellow]   Si raccomanda di usare LabelEncoder sulla colonna target nel preprocessor.[/yellow]")

        else: # Binary case
            if 'objective' not in model_params:
                model_params['objective'] = 'binary:logistic'
            if 'eval_metric' not in model_params:
                model_params['eval_metric'] = 'logloss'
        
        model_params['use_label_encoder'] = False # Recommended for modern XGBoost
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'catboost':
        if 'random_seed' not in model_params:
            model_params['random_seed'] = global_random_state
        if 'verbose' not in model_params:
            model_params['verbose'] = 0 
        if num_class > 2 and 'loss_function' not in model_params:
            model_params['loss_function'] = 'MultiClass'
        elif num_class == 2 and 'loss_function' not in model_params: # Binary
            model_params['loss_function'] = 'Logloss'
        model = cb.CatBoostClassifier(**model_params)
    else:
        console.print(f"[bold red]ERRORE: Tipo di modello '{model_type}' non supportato.[/bold red]", style="red")
        return None, None, None, None

    # Cross-validation
    k_folds = cv_config.get('n_splits', 5)
    shuffle_cv = cv_config.get('shuffle', True)
    random_state_cv = cv_config.get('random_state', global_random_state)
    
    console.print(f"[cyan]Esecuzione Cross-Validation ({k_folds}-fold) su dati di training...[/cyan]")
    try:
        # Ensure y_train is integer type for StratifiedKFold and some models
        y_train_cv = y_train.astype(int)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=shuffle_cv, random_state=random_state_cv)
        cv_scores = cross_val_score(model, X_train, y_train_cv, cv=skf, scoring='accuracy')
        console.print(f"[green]Accuratezza media Cross-Validation: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})[/green]")
    except ValueError as ve:
        console.print(f"[yellow]⚠️  Attenzione durante la Cross-Validation: {ve}[/yellow]")
        console.print(f"[yellow]   Possibile causa: y_train non è ancora in formato numerico/intero o ci sono problemi con le etichette delle classi.[/yellow]")
        console.print(f"[yellow]   Si procederà con il training sul set completo, ma la CV è stata saltata.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Errore imprevisto durante la Cross-Validation: {e}[/yellow]")
        console.print(f"[yellow]   Si procederà con il training sul set completo, ma la CV è stata saltata.[/yellow]")

    # Model Training
    console.print(f"[cyan]Training del modello {model_type.capitalize()} sul set di training completo...[/cyan]")
    try:
        model.fit(X_train, y_train)
        console.print(f"[green]✓ Modello {model_type.capitalize()} addestrato.[/green]")
    except Exception as e:
        console.print(f"[bold red]ERRORE durante l'addestramento del modello: {e}[/bold red]", style="red")
        return None, None, None, None

    # Evaluation on Test Set
    console.print(f"[cyan]Valutazione del modello {model_type.capitalize()} su Test Set...[/cyan]")
    y_pred = model.predict(X_test)
    
    # Ensure y_test and y_pred are of compatible types for metrics
    try:
        y_test_eval = y_test.astype(int)
        y_pred_eval = y_pred.astype(int)
    except ValueError:
        console.print(f"[yellow]⚠️ Attenzione: non è stato possibile convertire y_test/y_pred in interi per la valutazione. Tentativo con tipo originale.[/yellow]")
        y_test_eval = y_test
        y_pred_eval = y_pred
        # Fallback for CatBoost if it predicts string labels and y_test is numeric or vice-versa
        if isinstance(y_pred_eval, np.ndarray) and y_pred_eval.dtype == 'object':
            try:
                y_pred_eval = y_pred_eval.astype(y_test_eval.dtype)
            except ValueError:
                 # Try converting y_test_eval to string if y_pred_eval seems to be string class labels
                 if y_test_eval.dtype != 'object':
                    y_test_eval = y_test_eval.astype(str)


    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    conf_matrix = confusion_matrix(y_test_eval, y_pred_eval)
    # Ensure labels for classification report are consistent if y_test has few unique values
    unique_labels_test = np.unique(y_test_eval)
    unique_labels_pred = np.unique(y_pred_eval)
    report_labels = sorted(list(set(unique_labels_test) | set(unique_labels_pred)))


    class_report_str = classification_report(y_test_eval, y_pred_eval, zero_division=0, labels=report_labels if len(report_labels) > 1 else None)
    class_report_dict = classification_report(y_test_eval, y_pred_eval, output_dict=True, zero_division=0, labels=report_labels if len(report_labels) > 1 else None)

    console.print(Panel(f"[bold green]Risultati Valutazione su Test Set ({model_type.capitalize()})[/bold green]", border_style="green"))
    console.print(f"Accuratezza: [bold cyan]{accuracy:.4f}[/bold cyan]")
    
    console.print("\n[bold]Classification Report:[/bold]")
    console.print(Text(class_report_str))
    
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(conf_matrix)
    
    return model, conf_matrix, accuracy, class_report_dict