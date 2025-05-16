import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import pygad  # Replace DEAP with PyGad
import random
import warnings
from rich.console import Console
import traceback
import sys
import numpy as np
warnings.filterwarnings('ignore')


class GeneticFeatureSelector:
    def __init__(self, 
                 X_train=None,
                 X_test=None,
                 y_train=None,
                 y_test=None,
                 data_path=None,
                 target_column='GT',
                 test_size=0.2,
                 random_state=42,
                 population_size=50,
                 generations=30,
                 crossover_prob=0.7,
                 mutation_prob=0.2,
                 tournament_size=3,
                 feature_importance_weight=0.4,
                 feature_count_weight=0.3,
                 physics_correlation_weight=0.3):
        """
        Initialize the genetic feature selector.
        
        Parameters:
        - X_train, X_test, y_train, y_test: Preprocessed data (if already available)
        - data_path: Path to CSV file (alternative to providing preprocessed data)
        - target_column: Name of the target column
        - test_size: Proportion of data for testing
        - random_state: Random seed for reproducibility
        - population_size: Size of the GA population
        - generations: Number of GA generations
        - crossover_prob: Probability of crossover
        - mutation_prob: Probability of mutation
        - tournament_size: Size of tournament for selection
        - feature_importance_weight: Weight for feature importance in fitness
        - feature_count_weight: Weight for minimizing feature count in fitness
        - physics_correlation_weight: Weight for physics-based correlations in fitness
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.feature_importance_weight = feature_importance_weight
        self.feature_count_weight = feature_count_weight
        self.physics_correlation_weight = physics_correlation_weight
        
        # If preprocessed data is provided, use it
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.feature_names = list(X_train.columns)
            self.num_features = len(self.feature_names)
            self.preprocessed_data_provided = True
        else:
            self.preprocessed_data_provided = False
            # Load the data from file
            if data_path is not None:
                self.load_data()
            else:
                raise ValueError("Either preprocessed data (X_train, X_test, y_train, y_test) or data_path must be provided.")
        
        # Perform additional data preparation
        self.prepare_data()
        
        # Set up genetic algorithm components
        self.setup_ga()
    
    def load_data(self):
        """Load and preprocess the data."""
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        
        # Get feature names
        self.feature_names = list(self.X.columns)
        self.num_features = len(self.feature_names)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, 
            stratify=self.y if pd.api.types.is_categorical_dtype(self.y) or self.y.nunique() > 1 else None
        )
        
        print(f"Loaded dataset with {self.num_features} features and {len(self.y)} samples")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
    
    def prepare_data(self):
        """Prepare the data for feature selection."""
        # Create complete dataframe if we only have train/test data
        if self.preprocessed_data_provided:
            # Combine X_train and y_train to create a dataframe that looks like the original data
            train_df = self.X_train.copy()
            train_df[self.target_column] = self.y_train
            
            test_df = self.X_test.copy()
            test_df[self.target_column] = self.y_test
            
            self.df = pd.concat([train_df, test_df])
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column]
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Calculate feature correlations for physics insights
        self.feature_correlations = self.X.corr()
        
        # Calculate correlation of each feature with the target
        self.target_correlations = {}
        for feature in self.feature_names:
            correlation, _ = pearsonr(self.X[feature], self.y)
            self.target_correlations[feature] = abs(correlation)  # Use absolute correlation
            
        # Train a base model to get feature importances
        base_model = RandomForestClassifier(random_state=self.random_state)
        base_model.fit(self.X_train_scaled, self.y_train)
        self.feature_importances = dict(zip(self.feature_names, base_model.feature_importances_))
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Multi-objective fitness function for feature selection.
        
        Objectives:
        1. Maximize model performance (accuracy and F1 score)
        2. Minimize number of features
        3. Maximize feature importance and target correlation
        4. Minimize redundancy between selected features
        
        Higher fitness values are better for PyGAD.
        """
        console = Console()
        
        # Add diagnostic information
        if solution_idx % 10 == 0:  # Print only occasionally to avoid flooding
            console.print(f"[dim]Evaluating solution {solution_idx}...[/dim]")
        
        # Count selected features
        num_selected = sum(solution)
        
        # If no features are selected, return a small non-zero fitness
        if num_selected == 0:
            if solution_idx % 10 == 0:
                console.print(f"[yellow]⚠️ Solution {solution_idx} has no features selected[/yellow]")
            return 0.01  # Small positive value to guide search
        
        # Convert binary representation to feature indices
        selected_indices = [i for i, val in enumerate(solution) if val == 1]
        
        # Selected feature names (for correlation calculation)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        try:
            # Extract selected features from datasets
            X_train_selected = self.X_train_scaled[:, selected_indices]
            X_test_selected = self.X_test_scaled[:, selected_indices]
            
            # Train a model on selected features
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train_selected, self.y_train)
            y_pred = model.predict(X_test_selected)
            
            # OBJECTIVE 1: Model performance (higher is better)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            model_performance = 0.7 * f1 + 0.3 * accuracy  # Combined score (0-1 range)
            
            # OBJECTIVE 2: Feature count minimization
            # Normalize to 0-1 range where 1 means minimal features
            feature_ratio = 1.0 - (num_selected / self.num_features)
            
            # OBJECTIVE 3: Physics-based correlation insights
            
            # Average feature importance (higher is better)
            avg_importance = np.mean([self.feature_importances[feat] for feat in selected_features])
            
            # Average correlation with target (higher is better)
            avg_target_correlation = np.mean([self.target_correlations[feat] for feat in selected_features])
            
            # Calculate redundancy among selected features (lower is better)
            redundancy = 0.0
            if num_selected > 1:
                pairs = 0
                redundancy_sum = 0.0
                
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feat1 = selected_features[i]
                        feat2 = selected_features[j]
                        # Get absolute correlation
                        corr = abs(self.feature_correlations.loc[feat1, feat2])
                        redundancy_sum += corr
                        pairs += 1
                
                if pairs > 0:
                    redundancy = redundancy_sum / pairs
            
            # Convert redundancy to non-redundancy (so higher is better, like other metrics)
            non_redundancy = 1.0 - redundancy
            
            # Physics score (higher is better)
            physics_score = (0.4 * avg_importance + 
                            0.4 * avg_target_correlation + 
                            0.2 * non_redundancy)
            
            # Combine the objectives with weights (all objectives now have higher=better)
            fitness = (
                self.feature_importance_weight * model_performance + 
                self.feature_count_weight * feature_ratio +
                self.physics_correlation_weight * physics_score
            )
            
            # Ensure the fitness is always positive and non-zero
            fitness_value = max(0.01, fitness)
            
            # Add diagnostic for particularly good solutions
            if fitness_value > 0.8 and solution_idx % 10 == 0:
                console.print(f"[green]✓ Solution {solution_idx} has high fitness: {fitness_value:.4f} with {num_selected} features[/green]")
            
            return fitness_value
        
        except Exception as e:
            console.print(f"[red]Error in fitness evaluation for solution {solution_idx}: {str(e)}[/red]")
            return 0.01  # Small positive value for failed evaluations
    
    def setup_ga(self):
        """Set up PyGad genetic algorithm components."""
        # PyGad expects fitness function to return higher values for better solutions
        # (opposite of DEAP's minimize negative fitness)
        
        print("Setting up PyGad genetic algorithm...")
        
        # Configure the PyGad GA instance
        self.ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=int(self.population_size * 0.4),  # 40% of population becomes parents
            fitness_func=self.fitness_func,
            num_genes=self.num_features,
            sol_per_pop=self.population_size,
            gene_type=int,
            gene_space=[0, 1],  # Binary values only (0 or 1)
            parent_selection_type="tournament",
            K_tournament=self.tournament_size,
            crossover_type="two_points",
            crossover_probability=self.crossover_prob,
            mutation_type="random",
            mutation_probability=self.mutation_prob,
            keep_parents=1,  # Elitism parameter
            random_seed=self.random_state,
            save_best_solutions=True,  # Keep track of best solutions
            save_solutions=True,  # Keep all solutions for analysis
        )
        
        print("Using PyGad genetic algorithm for feature selection")
    
    def run(self):
        """Run the genetic algorithm optimization."""
        # Run the PyGad genetic algorithm
        print(f"Starting genetic algorithm with population size {self.population_size} for {self.generations} generations...")
        
        try:
            # Add more diagnostic information
            console = Console()
            
            # Initialize empty fitness arrays to prevent errors
            self.ga_instance.solutions_fitness = []
            
            # Run the algorithm with better error trapping
            with console.status("[bold cyan]Executing genetic algorithm...[/bold cyan]", spinner="dots"):
                self.ga_instance.run()
                console.print("[green]✓ Genetic algorithm execution completed[/green]")
            
            # Add detailed logging before getting the best solution
            console.print(f"[cyan]Generations completed: {self.ga_instance.generations_completed}[/cyan]")
            console.print(f"[cyan]Solutions fitness array shape: {np.array(self.ga_instance.solutions_fitness).shape}[/cyan]")
            
            # Get best solution with better error handling
            try:
                solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
                console.print(f"[green]✓ Best solution found with fitness: {solution_fitness}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error getting best solution: {str(e)}[/bold red]")
                # Create a fallback solution (select top 10% of features randomly)
                import random
                random.seed(self.random_state)
                num_to_select = max(1, int(self.num_features * 0.1))
                solution = np.zeros(self.num_features)
                selected_indices = random.sample(range(self.num_features), num_to_select)
                for idx in selected_indices:
                    solution[idx] = 1
                solution_fitness = 0.5  # Default placeholder fitness
                console.print(f"[yellow]⚠️ Using fallback solution with {num_to_select} random features[/yellow]")
            
            # Store results
            self.best_individual = solution
            self.best_features = [self.feature_names[i] for i, val in enumerate(solution) if val == 1]
            
            # More robust evolution log creation
            try:
                self.evolution_log = self._create_evolution_log()
            except Exception as e:
                console.print(f"[bold red]Error creating evolution log: {str(e)}[/bold red]")
                # Create a simple placeholder evolution log
                class SimpleEvolutionLog:
                    def __init__(self):
                        self.best = [0.5]
                        self.avg = [0.5]
                    def select(self, metric):
                        return self.best if metric == 'min' else self.avg
                self.evolution_log = SimpleEvolutionLog()
            
            console.print(f"[green]GA completed. Selected {len(self.best_features)} features with fitness {solution_fitness}[/green]")
            
            return self.best_features, self.evolution_log
            
        except Exception as e:
            console = Console()
            console.print(f"[bold red]Error during genetic algorithm execution: {str(e)}[/bold red]")
            console.print_exception()
            
            # Create fallback results
            self.best_individual = np.zeros(self.num_features)
            # Select top 5 features by importance
            importances = [(i, imp) for i, (feat, imp) in enumerate(self.feature_importances.items())]
            importances.sort(key=lambda x: x[1], reverse=True)
            for i, _ in importances[:5]:
                self.best_individual[i] = 1
            self.best_features = [self.feature_names[i] for i, val in enumerate(self.best_individual) if val == 1]
            
            # Create a simple placeholder evolution log
            class SimpleEvolutionLog:
                def __init__(self):
                    self.best = [0.5]
                    self.avg = [0.5]
                def select(self, metric):
                    return self.best if metric == 'min' else self.avg
            self.evolution_log = SimpleEvolutionLog()
            
            console.print(f"[yellow]⚠️ Using fallback solution with top 5 features by importance[/yellow]")
            return self.best_features, self.evolution_log

    def _create_evolution_log(self):
        """Create an evolution log similar to DEAP's for compatibility with existing code."""
        # Extract fitness data from PyGad with better error handling
        console = Console()
        console.print("[cyan]Creating evolution log...[/cyan]")
        
        best_fitness = []
        avg_fitness = []
        
        # Debug the solutions_fitness structure
        console.print(f"[cyan]Type of solutions_fitness: {type(self.ga_instance.solutions_fitness)}[/cyan]")
        
        # Handle the case where solutions_fitness is a single value instead of a list
        if isinstance(self.ga_instance.solutions_fitness, (float, np.float64, np.float32)):
            console.print("[yellow]⚠️ solutions_fitness is a single value, not an iterable[/yellow]")
            best_fitness = [float(self.ga_instance.solutions_fitness)]
            avg_fitness = [float(self.ga_instance.solutions_fitness)]
        else:
            try:
                # Try to iterate through generations
                for generation in range(self.ga_instance.num_generations):
                    try:
                        # Check if we have fitness data for this generation
                        if generation < len(self.ga_instance.solutions_fitness):
                            generation_fitnesses = self.ga_instance.solutions_fitness[generation]
                            
                            # Handle the case where generation_fitnesses might be a single value
                            if isinstance(generation_fitnesses, (float, np.float64, np.float32)):
                                console.print(f"[yellow]⚠️ Generation {generation} fitness is a single value[/yellow]")
                                best_fitness.append(float(generation_fitnesses))
                                avg_fitness.append(float(generation_fitnesses))
                            elif generation_fitnesses is not None and len(generation_fitnesses) > 0:
                                best_fitness.append(max(generation_fitnesses))
                                avg_fitness.append(sum(generation_fitnesses) / len(generation_fitnesses))
                            else:
                                # No fitness data for this generation
                                prev_best = best_fitness[-1] if best_fitness else 0.5
                                prev_avg = avg_fitness[-1] if avg_fitness else 0.5
                                best_fitness.append(prev_best)
                                avg_fitness.append(prev_avg)
                        else:
                            # We've run out of generations in the data
                            prev_best = best_fitness[-1] if best_fitness else 0.5
                            prev_avg = avg_fitness[-1] if avg_fitness else 0.5
                            best_fitness.append(prev_best)
                            avg_fitness.append(prev_avg)
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Error processing generation {generation}: {str(e)}[/yellow]")
                        # Use previous values or defaults
                        prev_best = best_fitness[-1] if best_fitness else 0.5
                        prev_avg = avg_fitness[-1] if avg_fitness else 0.5
                        best_fitness.append(prev_best)
                        avg_fitness.append(prev_avg)
            except Exception as e:
                console.print(f"[bold red]Error iterating through generations: {str(e)}[/bold red]")
                # Create at least one generation of data
                best_fitness = [0.5]
                avg_fitness = [0.5]
        
        # Create the evolution log class with better implementation
        class EvolutionLog:
            def __init__(self, best, avg):
                self.best = best
                self.avg = avg
            
            def select(self, metric):
                if metric == 'min':  # DEAP minimizes, so we return negative of best (which is maximized in PyGad)
                    return [-x for x in self.best]
                elif metric == 'avg':
                    return [-x for x in self.avg]
                elif metric == 'max':
                    return self.best
                return []
        
        console.print(f"[green]✓ Evolution log created with {len(best_fitness)} generations[/green]")
        return EvolutionLog(best_fitness, avg_fitness)
    
    def analyze_results(self):
        """Analyze the results of feature selection."""
        if not hasattr(self, 'best_individual'):
            raise ValueError("GA has not been run yet. Call run() first.")
        
        # Calculate number of selected features
        num_selected = sum(self.best_individual)
        
        # Get selected feature indices
        selected_indices = [i for i, val in enumerate(self.best_individual) if val == 1]
        
        # Extract selected features from datasets
        X_train_selected = self.X_train_scaled[:, selected_indices]
        X_test_selected = self.X_test_scaled[:, selected_indices]
        
        # Train a model on selected features
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_train_selected, self.y_train)
        y_pred = model.predict(X_test_selected)
        
        # Calculate model performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        # Create feature importance dataframe for the selected features
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.best_features,
            'Importance': feature_importances,
            'Correlation_with_Target': [self.target_correlations[feat] for feat in self.best_features]
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create correlation matrix for selected features
        selected_corr = self.feature_correlations.loc[self.best_features, self.best_features]
        
        print(f"\nGenetic Algorithm Results:")
        print(f"Selected {num_selected} features out of {self.num_features} ({num_selected/self.num_features:.1%})")
        print(f"\nSelected Features:")
        for i, feat in enumerate(self.best_features):
            print(f"{i+1}. {feat} (Importance: {importance_df[importance_df['Feature'] == feat]['Importance'].values[0]:.4f}, "
                  f"Target Correlation: {importance_df[importance_df['Feature'] == feat]['Correlation_with_Target'].values[0]:.4f})")
        
        print(f"\nModel Performance with Selected Features:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance of Selected Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(selected_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Features')
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        
        # Plot evolution of fitness
        gen = range(len(self.evolution_log.select('min')))
        fit_mins = self.evolution_log.select('min')
        fit_avgs = self.evolution_log.select('avg')
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness')
        plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.title('Evolution of Fitness Over Generations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('fitness_evolution.png')
        
        return {
            'selected_features': self.best_features,
            'feature_importances': importance_df,
            'correlation_matrix': selected_corr,
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def physics_based_analysis(self):
        """Perform physics-based analysis on the selected features."""
        if not hasattr(self, 'best_features'):
            raise ValueError("GA has not been run yet. Call run() first.")
        
        print("\n==== Physics-Based Analysis of Selected Features ====")
        
        # Create a DataFrame with only the selected features
        selected_df = self.df[self.best_features + [self.target_column]]
        
        # Analyze feature distribution by state/class
        states = selected_df[self.target_column].unique()
        
        # Create boxplots for each feature across different states
        for feature in self.best_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.target_column, y=feature, data=selected_df)
            plt.title(f'Distribution of {feature} by State')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'boxplot_{feature}.png')
        
        # Group data by state and get mean values for each feature
        state_means = selected_df.groupby(self.target_column).mean()
        
        print("\nMean Values of Selected Features by State:")
        print(state_means)
        
        # Calculate feature variability across states
        feature_variability = {}
        for feature in self.best_features:
            values_by_state = [selected_df[selected_df[self.target_column] == state][feature].values 
                               for state in states]
            
            # Calculate coefficient of variation for each state
            cv_by_state = [np.std(values) / np.mean(values) if np.mean(values) != 0 else 0 
                           for values in values_by_state]
            
            # Calculate difference between states (max mean - min mean) / overall mean
            means_by_state = [np.mean(values) for values in values_by_state]
            overall_mean = np.mean(selected_df[feature])
            state_difference = (max(means_by_state) - min(means_by_state)) / overall_mean if overall_mean != 0 else 0
            
            feature_variability[feature] = {
                'cv_by_state': dict(zip(states, cv_by_state)),
                'state_difference': state_difference
            }
        
        print("\nFeature Variability Analysis:")
        print("(Higher state_difference indicates the feature better distinguishes between states)")
        for feature, metrics in feature_variability.items():
            print(f"\n{feature}:")
            print(f"  State difference: {metrics['state_difference']:.4f}")
            print(f"  Coefficient of variation by state:")
            for state, cv in metrics['cv_by_state'].items():
                print(f"    State {state}: {cv:.4f}")
        
        # Create a feature variability DataFrame for visualization
        var_data = []
        for feature, metrics in feature_variability.items():
            var_data.append({
                'Feature': feature,
                'State_Difference': metrics['state_difference']
            })
        
        var_df = pd.DataFrame(var_data)
        var_df = var_df.sort_values('State_Difference', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='State_Difference', y='Feature', data=var_df)
        plt.title('Feature State Differentiation Power')
        plt.xlabel('State Difference (higher is better)')
        plt.tight_layout()
        plt.savefig('feature_state_differentiation.png')
        
        # Calculate pairwise correlations between selected features
        selected_corr = self.feature_correlations.loc[self.best_features, self.best_features]
        
        # Identify strongly correlated feature pairs (potential physical relationships)
        strong_correlations = []
        for i, feat1 in enumerate(self.best_features):
            for j, feat2 in enumerate(self.best_features[i+1:], i+1):
                corr = selected_corr.loc[feat1, feat2]
                if abs(corr) > 0.7:  # Threshold for strong correlation
                    strong_correlations.append((feat1, feat2, corr))
        
        print("\nStrong Feature Correlations (potential physical relationships):")
        for feat1, feat2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {feat1} ↔ {feat2}: {corr:.4f}")
        
        # Feature patterns across states
        patterns = []
        for feature in self.best_features:
            pattern = []
            for i, state1 in enumerate(states):
                for state2 in enumerate(states[i+1:], i+1):
                    mean1 = state_means.loc[state1, feature]
                    mean2 = state_means.loc[state2[1], feature]
                    if abs(mean1 - mean2) > 0.5 * np.mean([mean1, mean2]):  # Significant difference
                        pattern.append((state1, state2[1], feature, mean1, mean2))
            patterns.extend(pattern)
        
        if patterns:
            print("\nSignificant Feature Patterns Between States:")
            for state1, state2, feature, mean1, mean2 in patterns[:10]:  # Show top 10 patterns
                print(f"  {feature}: State {state1} ({mean1:.4f}) vs State {state2} ({mean2:.4f})")
        
        return {
            'state_means': state_means,
            'feature_variability': feature_variability,
            'strong_correlations': strong_correlations,
            'significant_patterns': patterns
        }

    def find_optimal_feature_subset(self):
        """Find the optimal number of features by evaluating performance with different feature counts."""
        if not hasattr(self, 'best_individual'):
            raise ValueError("GA has not been run yet. Call run() first.")
        
        # Get feature importance order based on the model trained on selected features
        selected_indices = [i for i, val in enumerate(self.best_individual) if val == 1]
        X_train_selected = self.X_train_scaled[:, selected_indices]
        X_test_selected = self.X_test_scaled[:, selected_indices]
        
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_train_selected, self.y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Selected feature names in order of importance
        ordered_features = [self.best_features[i] for i in indices]
        
        # Evaluate performance with increasing number of features
        accuracies = []
        f1_scores = []
        feature_counts = []
        
        step_size = max(1, len(ordered_features) // 10)  # Evaluate approximately 10 points
        
        for i in range(1, len(ordered_features) + 1, step_size):
            if i == 1:
                feature_subset = ordered_features[:i]
            else:
                feature_subset = ordered_features[:i]
            
            # Train and evaluate model with this subset
            X_train_subset = self.X_train[feature_subset]
            X_test_subset = self.X_test[feature_subset]
            
            # Scale features
            scaler = StandardScaler()
            X_train_subset_scaled = scaler.fit_transform(X_train_subset)
            X_test_subset_scaled = scaler.transform(X_test_subset)
            
            subset_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            subset_model.fit(X_train_subset_scaled, self.y_train)
            y_pred = subset_model.predict(X_test_subset_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            feature_counts.append(i)
            accuracies.append(accuracy)
            f1_scores.append(f1)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(feature_counts, accuracies, 'bo-', label='Accuracy')
        plt.plot(feature_counts, f1_scores, 'ro-', label='F1 Score')
        plt.xlabel('Number of Features')
        plt.ylabel('Score')
        plt.title('Performance vs Number of Features')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('performance_vs_features.png')
        
        # Find optimal feature count based on both metrics
        combined_score = [(cnt, 0.5*acc + 0.5*f1) 
                         for cnt, acc, f1 in zip(feature_counts, accuracies, f1_scores)]
        optimal_count, optimal_score = max(combined_score, key=lambda x: x[1])
        
        print(f"\nOptimal Feature Count Analysis:")
        print(f"Optimal number of features: {optimal_count} (Score: {optimal_score:.4f})")
        print(f"Selected features: {ordered_features[:optimal_count]}")
        
        # Create final optimal model
        optimal_features = ordered_features[:optimal_count]
        X_train_optimal = self.X_train[optimal_features]
        X_test_optimal = self.X_test[optimal_features]
        
        scaler = StandardScaler()
        X_train_optimal_scaled = scaler.fit_transform(X_train_optimal)
        X_test_optimal_scaled = scaler.transform(X_test_optimal)
        
        optimal_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        optimal_model.fit(X_train_optimal_scaled, self.y_train)
        y_pred_optimal = optimal_model.predict(X_test_optimal_scaled)
        
        print(f"\nOptimal Model Performance:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_optimal):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred_optimal, average='weighted'):.4f}")
        
        return {
            'optimal_feature_count': optimal_count,
            'optimal_features': optimal_features,
            'performance_data': {
                'feature_counts': feature_counts,
                'accuracies': accuracies,
                'f1_scores': f1_scores
            }
        }