import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

class GeneticFeatureSelector:
    def __init__(self, 
                 data_path,
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
        - data_path: Path to CSV file
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
        
        # Load the data
        self.load_data()
        
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
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        
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
            
        print(f"Loaded dataset with {self.num_features} features and {len(self.y)} samples")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
    
    def setup_ga(self):
        """Set up the genetic algorithm components."""
        # Create fitness function (we're minimizing negative fitness)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
        
        # Create Individual class
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Register attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Register individual creation
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_bool, n=self.num_features)
        
        # Register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate_features)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Set up parallel evaluation
        num_cores = multiprocessing.cpu_count()
        if num_cores > 1:
            pool = multiprocessing.Pool(processes=num_cores)
            self.toolbox.register("map", pool.map)
            print(f"Using {num_cores} CPU cores for parallel processing")
    
    def evaluate_features(self, individual):
        """
        Evaluate the fitness of an individual (feature subset).
        
        The fitness is a combination of:
        1. Model performance with selected features
        2. Number of selected features (fewer is better)
        3. Physics-based correlation insights
        """
        # Convert binary representation to feature indices
        selected_indices = [i for i, val in enumerate(individual) if val == 1]
        
        # If no features are selected, return worst fitness
        if sum(individual) == 0:
            return (float('inf'),)
        
        # Extract selected features from datasets
        X_train_selected = self.X_train_scaled[:, selected_indices]
        X_test_selected = self.X_test_scaled[:, selected_indices]
        
        # Train a model on selected features
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        try:
            model.fit(X_train_selected, self.y_train)
            y_pred = model.predict(X_test_selected)
            
            # Calculate model performance metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Calculate feature count penalty (normalized)
            feature_count_penalty = sum(individual) / self.num_features
            
            # Calculate physics-based correlation metric
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            # 1. Average importance of selected features
            avg_importance = np.mean([self.feature_importances[feat] for feat in selected_features])
            
            # 2. Average correlation with target for selected features
            avg_target_correlation = np.mean([self.target_correlations[feat] for feat in selected_features])
            
            # 3. Feature redundancy penalty (high correlation between selected features is penalized)
            redundancy = 0
            if len(selected_features) > 1:
                correlations = []
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        corr = abs(self.feature_correlations.loc[selected_features[i], selected_features[j]])
                        correlations.append(corr)
                redundancy = np.mean(correlations) if correlations else 0
            
            # Combine metrics into final fitness
            # Lower is better for GA optimization, so we use negative of beneficial metrics
            model_performance = -0.7*f1 - 0.3*accuracy  # We want to maximize these
            physics_metric = -0.5*avg_importance - 0.5*avg_target_correlation + 0.3*redundancy
            
            # Final weighted fitness
            fitness = (
                self.feature_importance_weight * model_performance +
                self.feature_count_weight * feature_count_penalty +
                self.physics_correlation_weight * physics_metric
            )
            
            return (fitness,)
            
        except Exception as e:
            # In case of any error (e.g., singular matrix), return worst fitness
            print(f"Error during evaluation: {e}")
            return (float('inf'),)
    
    def run(self):
        """Run the genetic algorithm optimization."""
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Track the best individuals
        hof = tools.HallOfFame(5)
        
        # Track statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the algorithm
        print(f"Starting genetic algorithm with population size {self.population_size} for {self.generations} generations...")
        pop, log = algorithms.eaSimple(pop, self.toolbox, 
                                       cxpb=self.crossover_prob, 
                                       mutpb=self.mutation_prob, 
                                       ngen=self.generations, 
                                       stats=stats, 
                                       halloffame=hof, 
                                       verbose=True)
        
        # Store results
        self.best_individual = hof[0]
        self.best_features = [self.feature_names[i] for i, val in enumerate(self.best_individual) if val == 1]
        self.evolution_log = log
        
        return self.best_features, log
    
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

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    data_path = "device1_data.csv"
    
    # Initialize and run the genetic feature selector
    selector = GeneticFeatureSelector(
        data_path=data_path,
        target_column='GT',
        population_size=50,
        generations=30,
        feature_importance_weight=0.4,
        feature_count_weight=0.3,
        physics_correlation_weight=0.3
    )
    
    # Run the genetic algorithm
    best_features, log = selector.run()
    
    # Analyze the results
    results = selector.analyze_results()
    
    # Perform physics-based analysis
    physics_analysis = selector.physics_based_analysis()
    
    # Find optimal feature subset
    optimal_subset = selector.find_optimal_feature_subset()
    
    print("\n=== Final Selected Features ===")
    print(f"Selected {len(best_features)} features:")
    for i, feature in enumerate(best_features):
        print(f"{i+1}. {feature}")
