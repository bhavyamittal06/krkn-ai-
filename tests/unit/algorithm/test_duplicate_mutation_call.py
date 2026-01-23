import pytest
from unittest.mock import Mock, patch
from krkn_ai.algorithm.genetic import GeneticAlgorithm

class TestDuplicateMutationCall:
    def test_verify_adapt_mutation_call_count(self, minimal_config, temp_output_dir):
        """
        Verify that adapt_mutation_rate is called exactly once per generation.
        """
        # Setup configuration to run for 1 generation
        minimal_config.generations = 1
        minimal_config.population_size = 4
        
        with patch("krkn_ai.algorithm.genetic.KrknRunner"), \
             patch("krkn_ai.algorithm.genetic.ScenarioFactory.generate_valid_scenarios") as mock_gen_valid, \
             patch("krkn_ai.algorithm.genetic.ScenarioFactory.generate_random_scenario") as mock_gen_rand:
            
            # Setup mocks
            mock_gen_valid.return_value = [("pod_scenarios", Mock)]
            mock_gen_rand.return_value = Mock()
            
            ga = GeneticAlgorithm(
                config=minimal_config, 
                output_dir=temp_output_dir, 
                format="yaml"
            )
            
            # Mock the method we want to track
            ga.adapt_mutation_rate = Mock()
            
            # Mock internal methods to avoid complex setup dependencies
            ga.create_population = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
            ga.calculate_fitness = Mock(return_value=Mock(fitness_result=Mock(fitness_score=10)))
            ga.select_parents = Mock(return_value=(Mock(), Mock()))
            ga.crossover = Mock(return_value=(Mock(), Mock()))
            ga.mutate = Mock(side_effect=lambda x: x)
            
            # Run simulation
            ga.simulate()
            
            # Verify call count
            # It should be called exactly once per generation
            assert ga.adapt_mutation_rate.call_count == 1, \
                f"adapt_mutation_rate was called {ga.adapt_mutation_rate.call_count} times, expected 1"
